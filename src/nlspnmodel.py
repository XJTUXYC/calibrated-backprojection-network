"""
    Non-Local Spatial Propagation Network for Depth Completion
    Jinsun Park, Kyungdon Joo, Zhe Hu, Chi-Kuei Liu and In So Kweon

    European Conference on Computer Vision (ECCV), Aug 2020

    Project Page : https://github.com/zzangjinsun/NLSPN_ECCV20
    Author : Jinsun Park (zzangjinsun@kaist.ac.kr)

    ======================================================================

    NLSPN implementation
"""


import os
import time
from common import *
from modulated_deform_conv_func import ModulatedDeformConvFunction
import torch
import torch.nn as nn
import torch.nn.functional as F
import log_utils, losses, networks, net_utils

EPSILON = 1e-8


class NLSPNModel(nn.Module):
    def __init__(self, min_predict_depth, max_predict_depth):
        super(NLSPNModel, self).__init__()
        
        self.prop_time8 = 1
        self.prop_time4 = 1
        self.prop_time2 = 1
        self.prop_time1 = 3
        self.num_feat4 = 96
        self.num_feat2 = 64
        self.prop_conf = False
        self.preserve_input = False
        self.always_clip = False
        self.min_predict_depth = min_predict_depth
        self.max_predict_depth = max_predict_depth
        
        self.device = torch.device('cuda')
        self.prop_kernel = 3
        assert (self.prop_kernel % 2) == 1, 'only odd kernel is supported but k_f = {}'.format(self.prop_kernel)
        self.num_neighbors = self.prop_kernel*self.prop_kernel - 1
        self.idx_ref = self.num_neighbors // 2

        # Encoder
        # 1/1
        self.enc11_rgb = conv_bn_relu(3, 32, kernel=3, stride=1, bn=False)
        self.S2D = S2D()

        backbone = get_resnet34(pretrained=True)

        self.enc11 = backbone.layer1
        
        # 1/2
        self.enc12 = backbone.layer2
        
        # 1/4
        self.enc24 = backbone.layer3
        del backbone

        # 1/8
        self.enc48 = conv_bn_relu(256, 256, kernel=3, stride=2)
        
        # Decoder
        # 1/8
        # self.dec88_dep = conv_bn_relu(256, 128, kernel=3, stride=1)
        if self.prop_time8 > 0:
            # self.dec88_aff = nn.Sequential(conv_bn_relu(256, 128, kernel=3, stride=1, relu=False), nn.Tanh())
            self.aff8_gen = conv_bn_relu(128, self.num_neighbors, kernel=3, stride=1, bn=False, relu=False, zero_init=True)
            self.GRU8 = ConvGRU(hidden=128, input=128)
        else:
            # self.dec88_aff = conv_bn_relu(256, 128, kernel=3, stride=1)
            pass

        # 1/4
        self.dec84_dep = convt_bn_relu(128+256, self.num_feat4, kernel=3, stride=2, padding=1, output_padding=1)
        if self.prop_time4 > 0:
            self.dec84_aff = nn.Sequential(convt_bn_relu(128+256, self.num_feat4, kernel=3, stride=2, padding=1, output_padding=1, relu=False), nn.Tanh())
            self.aff4_gen = conv_bn_relu(self.num_feat4, self.num_neighbors, kernel=3, stride=1, bn=False, relu=False, zero_init=True)
            self.GRU4 = ConvGRU(hidden=self.num_feat4, input=self.num_feat4)
        else:
            self.dec84_aff = convt_bn_relu(128+256, self.num_feat4, kernel=3, stride=2, padding=1, output_padding=1)
            
        # 1/2        
        self.dec42_dep = convt_bn_relu(self.num_feat4+256, self.num_feat2, kernel=3, stride=2, padding=1, output_padding=1)
        if self.prop_time2 > 0:
            self.dec42_aff = nn.Sequential(convt_bn_relu(self.num_feat4+256, self.num_feat2, kernel=3, stride=2, padding=1, output_padding=1, relu=False), nn.Tanh())
            self.aff2_gen = conv_bn_relu(self.num_feat2, self.num_neighbors, kernel=3, stride=1, bn=False, relu=False, zero_init=True)
            self.GRU2 = ConvGRU(hidden=self.num_feat2, input=self.num_feat2)
        else:
            self.dec42_aff = convt_bn_relu(self.num_feat4+256, self.num_feat2, kernel=3, stride=2, padding=1, output_padding=1)
            
        # 1/1
        self.dec21_dep = convt_bn_relu(self.num_feat2+128, 64, kernel=3, stride=2, padding=1, output_padding=1)
        self.dec21_aff = convt_bn_relu(self.num_feat2+128, 64, kernel=3, stride=2, padding=1, output_padding=1)
        
        self.dec11_dep = conv_bn_relu(64+64, 64, kernel=3, stride=1)
        self.dec11_pred = nn.Sequential(conv_bn_relu(64+64, 1, kernel=3, stride=1, bn=False, relu=False), nn.Sigmoid())
        if self.prop_conf:
            self.dec11_conf = nn.Sequential(conv_bn_relu(64+64, 1, kernel=3, stride=1, bn=False, relu=False), nn.Sigmoid())
        
        self.dec11_aff = conv_bn_relu(64+64, 64, kernel=3, stride=1)
        self.dec11_aff_off = conv_bn_relu(64+64, 3*self.num_neighbors, kernel=3, stride=1, bn=False, relu=False, zero_init=True)
        
        self.GRU1 = ConvGRU(hidden=self.num_neighbors, input=1, zero_init=True, tanh=False)

        # aff_scale_const
        if self.prop_time8 > 0:
            self.aff_scale_const8 = nn.Parameter(0.5 * self.num_neighbors * torch.ones(1))
        if self.prop_time4 > 0:
            self.aff_scale_const4 = nn.Parameter(0.5 * self.num_neighbors * torch.ones(1))
        if self.prop_time2 > 0:
            self.aff_scale_const2 = nn.Parameter(0.5 * self.num_neighbors * torch.ones(1))
        self.aff_scale_const1 = nn.Parameter(0.5 * self.num_neighbors * torch.ones(1))

        # DCN (Dummy parameters for gathering)
        self.ch_f = 1
        
        self.w = nn.Parameter(torch.ones((self.ch_f, 1, self.prop_kernel, self.prop_kernel)))
        self.b = nn.Parameter(torch.zeros(self.ch_f))

        self.w.requires_grad = False
        self.b.requires_grad = False

        self.w_conf = nn.Parameter(torch.ones((1, 1, 1, 1)))
        self.w_conf.requires_grad = False

        self.stride = 1
        self.padding = int((self.prop_kernel - 1) / 2)
        self.dilation = 1
        self.groups = self.ch_f
        self.deformable_groups = 1
        self.im2col_step = 64
        
        self.data_parallel()
        self.to(self.device)
    
    def _aff_norm_insert(self, aff, level):
        if level == 8:
            aff = torch.tanh(aff) / (self.aff_scale_const8 + 1e-8)
        elif level == 4:
            aff = torch.tanh(aff) / (self.aff_scale_const4 + 1e-8)
        elif level == 2:
            aff = torch.tanh(aff) / (self.aff_scale_const2 + 1e-8)
        else:
            aff = torch.tanh(aff) / (self.aff_scale_const1 + 1e-8)

        # Affinity normalization
        aff_abs = torch.abs(aff)
        aff_abs_sum = torch.sum(aff_abs, dim=1, keepdim=True) + 1e-4

        aff_abs_sum[aff_abs_sum < 1.0] = 1.0

        aff = aff / aff_abs_sum

        aff = self._aff_insert(aff)
        
        return aff
    
    def _propagate_once(self, feat, offset, aff):
        if offset is not None:
            feat = ModulatedDeformConvFunction.apply(
                feat, offset, aff, self.w, self.b, self.stride, self.padding,
                self.dilation, self.groups, self.deformable_groups, self.im2col_step
            )
            
            return feat
        
        else:
            # TODO: Faster!
            feat = F.pad(feat, (1,1,1,1), mode="replicate")
            _, _, H, W = feat.size()
            new_feat =  feat[:, :, 0:H-2, 0:W-2] * torch.unsqueeze(aff[:, 0, :, :], dim=1)
            new_feat += feat[:, :, 0:H-2, 1:W-1] * torch.unsqueeze(aff[:, 1, :, :], dim=1)
            new_feat += feat[:, :, 0:H-2, 2:W-0] * torch.unsqueeze(aff[:, 2, :, :], dim=1)
            new_feat += feat[:, :, 1:H-1, 0:W-2] * torch.unsqueeze(aff[:, 3, :, :], dim=1)
            new_feat += feat[:, :, 1:H-1, 1:W-1] * torch.unsqueeze(aff[:, 4, :, :], dim=1)
            new_feat += feat[:, :, 1:H-1, 2:W-0] * torch.unsqueeze(aff[:, 5, :, :], dim=1)
            new_feat += feat[:, :, 2:H-0, 0:W-2] * torch.unsqueeze(aff[:, 6, :, :], dim=1)
            new_feat += feat[:, :, 2:H-0, 1:W-1] * torch.unsqueeze(aff[:, 7, :, :], dim=1)
            new_feat += feat[:, :, 2:H-0, 2:W-0] * torch.unsqueeze(aff[:, 8, :, :], dim=1)
            
            return new_feat
    
    def _off_insert(self, offset):
        B, _, H, W = offset.shape
        offset = offset.view(B, self.num_neighbors, 2, H, W)
        list_offset = list(torch.chunk(offset, self.num_neighbors, dim=1))
        list_offset.insert(self.idx_ref, torch.zeros((B, 1, 2, H, W)).type_as(offset))
        offset = torch.cat(list_offset, dim=1).view(B, -1, H, W)
        
        return offset
    
    def _aff_insert(self, aff):
        aff_sum = torch.sum(aff, dim=1, keepdim=True)
        aff_ref = 1.0 - aff_sum

        list_aff = list(torch.chunk(aff, self.num_neighbors, dim=1))
        list_aff.insert(self.idx_ref, aff_ref)
        aff = torch.cat(list_aff, dim=1)
        
        return aff
    
    def _aff_pop(self, aff):
        list_aff = list(torch.chunk(aff, self.num_neighbors+1, dim=1))
        list_aff.pop(self.idx_ref)
        aff = torch.cat(list_aff, dim=1)
        
        return aff
    
    def compute_loss(self,
                     image0,
                     image1,
                     image2,
                     output_depth0,
                     sparse_depth0,
                     validity_map_depth0,
                     intrinsics,
                     pose01,
                     pose02,
                     w_color=0.15,
                     w_structure=0.95,
                     w_sparse_depth=0.60,
                     w_smoothness=0.04,
                     validity_map_image0=None):
        '''
        Computes loss function
        l = w_{ph}l_{ph} + w_{sz}l_{sz} + w_{sm}l_{sm}

        Arg(s):
            image0 : torch.Tensor[float32]
                N x 3 x H x W image at time step t
            image1 : torch.Tensor[float32]
                N x 3 x H x W image at time step t-1
            image2 : torch.Tensor[float32]
                N x 3 x H x W image at time step t+1
            output_depth0 : torch.Tensor[float32]
                N x 1 x H x W output depth at time t
            sparse_depth0 : torch.Tensor[float32]
                N x 1 x H x W sparse depth at time t
            validity_map_depth0 : torch.Tensor[float32]
                N x 1 x H x W validity map of sparse depth at time t
            intrinsics : torch.Tensor[float32]
                N x 3 x 3 camera intrinsics matrix
            pose01 : torch.Tensor[float32]
                N x 4 x 4 relative pose from image at time t to t-1
            pose02 : torch.Tensor[float32]
                N x 4 x 4 relative pose from image at time t to t+1
            w_color : float
                weight of color consistency term
            w_structure : float
                weight of structure consistency term (SSIM)
            w_sparse_depth : float
                weight of sparse depth consistency term
            w_smoothness : float
                weight of local smoothness term
        Returns:
            torch.Tensor[float32] : loss
            dict[str, torch.Tensor[float32]] : dictionary of loss related tensors
        '''

        shape = image0.shape
        if validity_map_image0 == None:
            validity_map_image0 = torch.ones_like(sparse_depth0)
            need_validity_map_image0 = False
        else:
            need_validity_map_image0 = True

        # Backproject points to 3D camera coordinates
        points = net_utils.backproject_to_camera(output_depth0, intrinsics, shape)

        # Reproject points onto image 1 and image 2
        target_xy01 = net_utils.project_to_pixel(points, pose01, intrinsics, shape)
        target_xy02 = net_utils.project_to_pixel(points, pose02, intrinsics, shape)

        # Reconstruct image0 from image1 and image2 by reprojection
        image01 = net_utils.grid_sample(image1, target_xy01, shape)
        image02 = net_utils.grid_sample(image2, target_xy02, shape)

        '''
        Essential loss terms
        '''
        # Color consistency loss function
        loss_color01 = losses.color_consistency_loss_func(
            src=image01,
            tgt=image0,
            w=validity_map_image0,
            need_w=need_validity_map_image0)
        loss_color02 = losses.color_consistency_loss_func(
            src=image02,
            tgt=image0,
            w=validity_map_image0,
            need_w=need_validity_map_image0)
        loss_color = loss_color01 + loss_color02

        # Structural consistency loss function
        loss_structure01 = losses.structural_consistency_loss_func(
            src=image01,
            tgt=image0,
            w=validity_map_image0,
            need_w=need_validity_map_image0)
        loss_structure02 = losses.structural_consistency_loss_func(
            src=image02,
            tgt=image0,
            w=validity_map_image0,
            need_w=need_validity_map_image0)
        loss_structure = loss_structure01 + loss_structure02

        # Sparse depth consistency loss function
        loss_sparse_depth = losses.sparse_depth_consistency_loss_func(
            src=output_depth0,
            tgt=sparse_depth0,
            w=validity_map_depth0)

        # Local smoothness loss function
        loss_smoothness = losses.smoothness_loss_func(
            predict=output_depth0,
            image=image0)

        # l = w_{ph}l_{ph} + w_{sz}l_{sz} + w_{sm}l_{sm}
        loss = w_color * loss_color + \
            w_structure * loss_structure + \
            w_sparse_depth * loss_sparse_depth + \
            w_smoothness * loss_smoothness

        loss_info = {
            'loss_color' : loss_color,
            'loss_structure' : loss_structure,
            'loss_sparse_depth' : loss_sparse_depth,
            'loss_smoothness' : loss_smoothness,
            'loss' : loss,
            'image01' : image01,
            'image02' : image02
        }

        return loss, loss_info
    
    def save_model(self, checkpoint_path, epoch, optimizer):
        '''
        Save weights of the model to checkpoint path

        Arg(s):
            checkpoint_path : str
                path to save checkpoint
            step : int
                current training step
            optimizer : torch.optim
                optimizer
        '''

        checkpoint = {}
        # Save training state
        checkpoint['epoch'] = epoch
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()

        # Save encoder and decoder weights
        checkpoint['state_dict'] = self.state_dict()

        torch.save(checkpoint, checkpoint_path)

    def restore_model(self, checkpoint_path, optimizer=None):
        '''
        Restore weights of the model

        Arg(s):
            checkpoint_path : str
                path to checkpoint
            optimizer : torch.optim
                optimizer
        Returns:
            int : current step in optimization
            torch.optim : optimizer with restored state
        '''

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Restore sparse to dense pool, encoder and decoder weights
        self.load_state_dict(checkpoint['state_dict'])

        if optimizer is not None:
            try:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            except Exception:
                pass

        # Return the current step and optimizer
        return checkpoint['epoch'], optimizer

    def data_parallel(self):
        '''
        Allows multi-gpu split along batch
        '''
        device_ids = [_ for _ in range(len(os.environ['CUDA_VISIBLE_DEVICES'].split(",")))]
        self = torch.nn.DataParallel(self, device_ids=device_ids)
    
    def log_summary(self,
                    summary_writer,
                    tag,
                    epoch,
                    image0=None,
                    image01=None,
                    image02=None,
                    output_depth0=None,
                    sparse_depth0=None,
                    validity_map0=None,
                    ground_truth0=None,
                    pose01=None,
                    pose02=None,
                    scalars={},
                    n_display=4):
        '''
        Logs summary to Tensorboard

        Arg(s):
            summary_writer : SummaryWriter
                Tensorboard summary writer
            tag : str
                tag that prefixes names to log
            step : int
                current step in training
            image0 : torch.Tensor[float32]
                image at time step t
            image01 : torch.Tensor[float32]
                image at time step t-1 warped to time step t
            image02 : torch.Tensor[float32]
                image at time step t+1 warped to time step t
            output_depth0 : torch.Tensor[float32]
                output depth at time t
            sparse_depth0 : torch.Tensor[float32]
                sparse_depth at time t
            validity_map0 : torch.Tensor[float32]
                validity map of sparse depth at time t
            ground_truth0 : torch.Tensor[float32]
                ground truth depth at time t
            pose01 : torch.Tensor[float32]
                4 x 4 relative pose from image at time t to t-1
            pose02 : torch.Tensor[float32]
                4 x 4 relative pose from image at time t to t+1
            scalars : dict[str, float]
                dictionary of scalars to log
            n_display : int
                number of images to display
        '''

        with torch.no_grad():

            display_summary_image = []
            display_summary_depth = []

            display_summary_image_text = tag
            display_summary_depth_text = tag

            if image0 is not None:
                image0_summary = image0[0:n_display, ...]

                display_summary_image_text += '_image0'
                display_summary_depth_text += '_image0'

                # Add to list of images to log
                display_summary_image.append(
                    torch.cat([
                        image0_summary.cpu(),
                        torch.zeros_like(image0_summary, device=torch.device('cpu'))],
                        dim=-1))

                display_summary_depth.append(display_summary_image[-1])

            if image0 is not None and image01 is not None:
                image01_summary = image01[0:n_display, ...]

                display_summary_image_text += '_image01-error'

                # Compute reconstruction error w.r.t. image 0
                image01_error_summary = torch.mean(
                    torch.abs(image0_summary - image01_summary),
                    dim=1,
                    keepdim=True)

                # Add to list of images to log
                image01_error_summary = log_utils.colorize(
                    (image01_error_summary / 0.10).cpu(),
                    colormap='inferno')

                display_summary_image.append(
                    torch.cat([
                        image01_summary.cpu(),
                        image01_error_summary],
                        dim=3))

            if image0 is not None and image02 is not None:
                image02_summary = image02[0:n_display, ...]

                display_summary_image_text += '_image02-error'

                # Compute reconstruction error w.r.t. image 0
                image02_error_summary = torch.mean(
                    torch.abs(image0_summary - image02_summary),
                    dim=1,
                    keepdim=True)

                # Add to list of images to log
                image02_error_summary = log_utils.colorize(
                    (image02_error_summary / 0.10).cpu(),
                    colormap='inferno')

                display_summary_image.append(
                    torch.cat([
                        image02_summary.cpu(),
                        image02_error_summary],
                        dim=3))

            if output_depth0 is not None:
                output_depth0_summary = output_depth0[0:n_display, ...]

                display_summary_depth_text += '_output0'

                # Add to list of images to log
                n_batch, _, n_height, n_width = output_depth0_summary.shape

                display_summary_depth.append(
                    torch.cat([
                        log_utils.colorize(
                            (output_depth0_summary / self.max_predict_depth).cpu(),
                            colormap='viridis'),
                        torch.zeros(n_batch, 3, n_height, n_width, device=torch.device('cpu'))],
                        dim=3))

                # Log distribution of output depth
                summary_writer.add_histogram(tag + '_output_depth0_distro', output_depth0, global_step=epoch)

            if output_depth0 is not None and sparse_depth0 is not None and validity_map0 is not None:
                sparse_depth0_summary = sparse_depth0[0:n_display]
                validity_map0_summary = validity_map0[0:n_display]

                display_summary_depth_text += '_sparse0-error'

                # Compute output error w.r.t. input sparse depth
                sparse_depth0_error_summary = \
                    torch.abs(output_depth0_summary - sparse_depth0_summary)

                sparse_depth0_error_summary = torch.where(
                    validity_map0_summary == 1.0,
                    (sparse_depth0_error_summary + EPSILON) / (sparse_depth0_summary + EPSILON),
                    validity_map0_summary)

                # Add to list of images to log
                sparse_depth0_summary = log_utils.colorize(
                    (sparse_depth0_summary / self.max_predict_depth).cpu(),
                    colormap='viridis')
                sparse_depth0_error_summary = log_utils.colorize(
                    (sparse_depth0_error_summary / 0.05).cpu(),
                    colormap='inferno')

                display_summary_depth.append(
                    torch.cat([
                        sparse_depth0_summary,
                        sparse_depth0_error_summary],
                        dim=3))

                # Log distribution of sparse depth
                summary_writer.add_histogram(tag + '_sparse_depth0_distro', sparse_depth0, global_step=epoch)

            if output_depth0 is not None and ground_truth0 is not None:
                validity_map0 = torch.unsqueeze(ground_truth0[:, 1, :, :], dim=1)
                ground_truth0 = torch.unsqueeze(ground_truth0[:, 0, :, :], dim=1)

                validity_map0_summary = validity_map0[0:n_display]
                ground_truth0_summary = ground_truth0[0:n_display]

                display_summary_depth_text += '_groundtruth0-error'

                # Compute output error w.r.t. ground truth
                ground_truth0_error_summary = \
                    torch.abs(output_depth0_summary - ground_truth0_summary)

                ground_truth0_error_summary = torch.where(
                    validity_map0_summary == 1.0,
                    (ground_truth0_error_summary + EPSILON) / (ground_truth0_summary + EPSILON),
                    validity_map0_summary)

                # Add to list of images to log
                ground_truth0_summary = log_utils.colorize(
                    (ground_truth0_summary / self.max_predict_depth).cpu(),
                    colormap='viridis')
                ground_truth0_error_summary = log_utils.colorize(
                    (ground_truth0_error_summary / 0.05).cpu(),
                    colormap='inferno')

                display_summary_depth.append(
                    torch.cat([
                        ground_truth0_summary,
                        ground_truth0_error_summary],
                        dim=3))

                # Log distribution of ground truth
                summary_writer.add_histogram(tag + '_ground_truth0_distro', ground_truth0, global_step=epoch)

            if pose01 is not None:
                # Log distribution of pose 1 to 0translation vector
                summary_writer.add_histogram(tag + '_tx01_distro', pose01[:, 0, 3], global_step=epoch)
                summary_writer.add_histogram(tag + '_ty01_distro', pose01[:, 1, 3], global_step=epoch)
                summary_writer.add_histogram(tag + '_tz01_distro', pose01[:, 2, 3], global_step=epoch)

            if pose02 is not None:
                # Log distribution of pose 2 to 0 translation vector
                summary_writer.add_histogram(tag + '_tx02_distro', pose02[:, 0, 3], global_step=epoch)
                summary_writer.add_histogram(tag + '_ty02_distro', pose02[:, 1, 3], global_step=epoch)
                summary_writer.add_histogram(tag + '_tz02_distro', pose02[:, 2, 3], global_step=epoch)

        # Log scalars to tensorboard
        for (name, value) in scalars.items():
            summary_writer.add_scalar(tag + '_' + name, value, global_step=epoch)

        # Log image summaries to tensorboard
        if len(display_summary_image) > 1:
            display_summary_image = torch.cat(display_summary_image, dim=2)

            summary_writer.add_image(
                display_summary_image_text,
                torchvision.utils.make_grid(display_summary_image, nrow=n_display),
                global_step=epoch)

        if len(display_summary_depth) > 1:
            display_summary_depth = torch.cat(display_summary_depth, dim=2)

            summary_writer.add_image(
                display_summary_depth_text,
                torchvision.utils.make_grid(display_summary_depth, nrow=n_display),
                global_step=epoch)

    def forward(self, rgb, dep):
        # Encoding
        # 1/1
        fe1_rgb = self.enc11_rgb(rgb) # b*32*H*W
        fe1_dep = self.S2D(dep) # b*32*H*W

        fe1 = torch.cat((fe1_rgb, fe1_dep), dim=1) # b*64*H*W
        
        fe1_mix = self.enc11(fe1) # b*64*H*W
        
        # 1/2
        fe2 = self.enc12(fe1_mix) # b*128*H/2*W/2
        
        # 1/4
        fe4 = self.enc24(fe2) # b*256*H/4*W/4
        
        # 1/8
        fe8 = self.enc48(fe4) # b*256*H/8*W/8
        
        # Decoding
        # 1/8
        # fe8_dep = self.dec88_dep(fe8) # b*128*H/8*W/8
        # fe8_aff = self.dec88_aff(fe8) # b*128*H/8*W/8
        fe8_dep = fe8[:, :128, :, :]
        fe8_aff = fe8[:, 128:, :, :]
        
        # time_start = time.time()
        for _ in range(self.prop_time8):
            aff8 = self.aff8_gen(fe8_aff)
            aff8 = self._aff_norm_insert(aff8, 8)
            
            fe8_dep = self._propagate_once(fe8_dep, None, aff8)
            fe8_dep = F.relu(fe8_dep)
            
            fe8_aff = self.GRU8(h=fe8_aff, x=fe8_dep)
        # time_end=time.time()
        # print('time cost for GRU8',1000*(time_end-time_start),'ms')
        
        # 1/4
        fd4_dep = self.dec84_dep(torch.cat([fe8_dep, fe8], dim=1)) # b*(128+256)*H/8*W/8 -> b*128*H/4*W/4
        fd4_aff = self.dec84_aff(torch.cat([fe8_aff, fe8], dim=1)) # b*(128+256)*H/8*W/8 -> b*128*H/4*W/4
        
        # time_start = time.time()
        for _ in range(self.prop_time4):
            aff4 = self.aff4_gen(fd4_aff)
            aff4 = self._aff_norm_insert(aff4, 4)
            
            fd4_dep = self._propagate_once(fd4_dep, None, aff4)
            fd4_dep = F.relu(fd4_dep)
            
            fd4_aff = self.GRU4(h=fd4_aff, x=fd4_dep)
        # time_end=time.time()
        # print('time cost for GRU4',1000*(time_end-time_start),'ms')
        
        # 1/2
        fd2_dep = self.dec42_dep(concat(fd4_dep, fe4)) # b*(128+256)*H/4*W/4 -> b*64*H/2*W/2
        fd2_aff = self.dec42_aff(concat(fd4_aff, fe4)) # b*(128+256)*H/4*W/4 -> b*64*H/2*W/2
        
        # time_start = time.time()
        for _ in range(self.prop_time2):
            aff2 = self.aff2_gen(fd2_aff)
            aff2 = self._aff_norm_insert(aff2, 2)
            
            fd2_dep = self._propagate_once(fd2_dep, None, aff2)
            fd2_dep = F.relu(fd2_dep)
            
            fd2_aff = self.GRU2(h=fd2_aff, x=fd2_dep)
        # time_end=time.time()
        # print('time cost for GRU2',1000*(time_end-time_start),'ms')
        
        # 1/1
        fd1_dep = self.dec21_dep(concat(fd2_dep, fe2)) # b*(64+128)*H/2*W/2 -> b*64*H*W
        fd1_aff = self.dec21_aff(concat(fd2_aff, fe2)) # b*(64+128)*H/2*W/2 -> b*64*H*W
        
        # pred_conf
        fd1_pred_conf = self.dec11_dep(concat(fd1_dep, fe1_mix)) # b*(64+64)*H*W -> b*64*H*W
        pred = self.dec11_pred(concat(fd1_pred_conf, fe1)) # b*(64+64)*H*W -> b*1*H*W
        
        if self.preserve_input:
            mask_fix = torch.sum((dep >= self.min_predict_depth) & (dep <= self.max_predict_depth), dim=1, keepdim=True).detach()
            mask_fix = (mask_fix > 0.0).type_as(dep)
            pred = (1.0 - mask_fix) * pred + mask_fix * (self.min_predict_depth / (dep + 1e-8) - self.min_predict_depth / self.max_predict_depth)
        if self.always_clip:
            pred = torch.clamp(pred, min=0, max=1-self.min_predict_depth/self.max_predict_depth)
            
        if self.prop_conf:
            conf = self.dec11_conf(concat(fd1_pred_conf, fe1)) # b*(64+64)*H*W -> b*1*H*W
            if self.preserve_input:
                conf = (1.0 - mask_fix) * conf + mask_fix
        
        # aff_off
        fd1_aff_off = self.dec11_aff(concat(fd1_aff, fe1_mix)) # b*(64+64)*H*W -> b*64*H*W
        aff_off = self.dec11_aff_off(concat(fd1_aff_off, fe1)) # b*(64+64)*H*W -> b*24*H*W
        aff = aff_off[:, :self.num_neighbors, :, :] # b*8*H*W
        off = aff_off[:, self.num_neighbors:, :, :] # b*16*H*W
        aff = self._aff_norm_insert(aff, 1) # b*9*H*W
        off = self._off_insert(off) # b*18*H*W

        # DCSPN
        # time_start = time.time()
        for k in range(self.prop_time1):
            if self.prop_conf:
                pred = self._propagate_once(pred*conf, off, aff)
            else:
                pred = self._propagate_once(pred, off, aff)

            if self.preserve_input:
                pred = (1.0 - mask_fix) * pred + mask_fix * (self.min_predict_depth / (dep + 1e-8) - self.min_predict_depth / self.max_predict_depth)
            if self.always_clip:
                pred = torch.clamp(pred, min=0, max=1-self.min_predict_depth/self.max_predict_depth)
                                                     
            if k < self.prop_time1 - 1:
                aff = self._aff_pop(aff) # b*8*H*W
                aff = self.GRU1(h=aff, x=pred) # b*8*H*W
                aff = self._aff_norm_insert(aff, 1) # b*9*H*W
        # time_end=time.time()
        # print('time cost for GRU1',1000*(time_end-time_start),'ms')
        
        # output
        pred = self.min_predict_depth / (pred + self.min_predict_depth / self.max_predict_depth)
        pred = torch.clamp(pred, min=self.min_predict_depth, max=self.max_predict_depth)

        return pred
    
    
class ConvGRU(nn.Module):
    def __init__(self, hidden, input, zero_init=False, tanh=True):
        super(ConvGRU, self).__init__()
        self.tanh = tanh
        self.convz = nn.Conv2d(hidden+input, hidden, 3, padding=1)
        self.convr = nn.Conv2d(hidden+input, hidden, 3, padding=1)
        
        self.convq = nn.Conv2d(hidden+input, hidden, 3, padding=1)
        if zero_init:
            self.convq.weight.data.zero_()
            self.convq.bias.data.zero_()

    def forward(self, h, x):
        hx = torch.cat([h, x], dim=1)

        z = torch.sigmoid(self.convz(hx))
        r = torch.sigmoid(self.convr(hx))
        
        if self.tanh:
            q = torch.tanh(self.convq(torch.cat([r*h, x], dim=1)))
        else:
            q = self.convq(torch.cat([r*h, x], dim=1))
            
        h = (1-z) * h + z * q
        return h
    
    
class S2D(nn.Module):
    def __init__(self):
        super(S2D, self).__init__()

        self.min_pool_sizes = [5,7,9,11,13]
        self.max_pool_sizes = [15,17]

        # Construct min pools
        self.min_pools = []
        for s in self.min_pool_sizes:
            padding = s // 2
            pool = nn.MaxPool2d(kernel_size=s, stride=1, padding=padding)
            self.min_pools.append(pool)

        # Construct max pools
        self.max_pools = []
        for s in self.max_pool_sizes:
            padding = s // 2
            pool = nn.MaxPool2d(kernel_size=s, stride=1, padding=padding)
            self.max_pools.append(pool)

        in_channels = len(self.min_pool_sizes) + len(self.max_pool_sizes)

        self.pool_convs = nn.Sequential(
            conv_bn_relu(in_channels, 8, kernel=1, stride=1, bn=False),
            conv_bn_relu(8, 16, kernel=1, stride=1, bn=False),
            conv_bn_relu(16, 32-1, kernel=1, stride=1, bn=False)
        )

        self.conv = conv_bn_relu(32, 32, kernel=3, stride=1, bn=False)

    def forward(self, dep):
        pool_pyramid = []

        for pool, s in zip(self.min_pools, self.min_pool_sizes):
            z_pool = -pool(torch.where(dep == 0, -999 * torch.ones_like(dep), -dep))
            z_pool = torch.where(z_pool == 999, torch.zeros_like(dep), z_pool)

            pool_pyramid.append(z_pool)

        for pool, s in zip(self.max_pools, self.max_pool_sizes):
            z_pool = pool(dep)

            pool_pyramid.append(z_pool)

        pool_pyramid = torch.cat(pool_pyramid, dim=1)
        pool_pyramid = self.pool_convs(pool_pyramid)

        pool_pyramid = torch.cat([dep, pool_pyramid], dim=1)
        pool_pyramid = self.conv(pool_pyramid)

        return pool_pyramid
