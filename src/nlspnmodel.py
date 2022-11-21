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
from common import *
from modulated_deform_conv_func import ModulatedDeformConvFunction
import torch
import torch.nn as nn
import log_utils, losses, networks, net_utils

EPSILON = 1e-8


class NLSPN(nn.Module):
    def __init__(
        self,
        ch_g,
        ch_f,
        k_g,
        k_f,
        conf_prop=True,
        prop_time=18,
        affinity='TGASS',
        affinity_gamma=0.5,
        legacy=False,
        preserve_input=False
    ):
        super(NLSPN, self).__init__()

        # Guidance : [B x ch_g x H x W]
        # Feature : [B x ch_f x H x W]

        # Currently only support ch_f == 1
        assert ch_f == 1, 'only tested with ch_f == 1 but {}'.format(ch_f)

        assert (k_g % 2) == 1, \
            'only odd kernel is supported but k_g = {}'.format(k_g)
        pad_g = int((k_g - 1) / 2)
        assert (k_f % 2) == 1, \
            'only odd kernel is supported but k_f = {}'.format(k_f)
        pad_f = int((k_f - 1) / 2)

        self.prop_time = prop_time
        self.affinity = affinity
        self.affinity_gamma = affinity_gamma
        self.conf_prop = conf_prop
        self.legacy = legacy
        self.preserve_input = preserve_input
        self.ch_g = ch_g
        self.ch_f = ch_f
        self.k_g = k_g
        self.k_f = k_f
        # Assume zero offset for center pixels
        self.num = self.k_f * self.k_f - 1
        self.idx_ref = self.num // 2

        if self.affinity in ['AS', 'ASS', 'TC', 'TGASS']:
            self.conv_offset_aff = nn.Conv2d(
                self.ch_g, 3 * self.num, kernel_size=self.k_g, stride=1,
                padding=pad_g, bias=True
            )
            self.conv_offset_aff.weight.data.zero_()
            self.conv_offset_aff.bias.data.zero_()

            if self.affinity == 'TC':
                self.aff_scale_const = nn.Parameter(self.num * torch.ones(1))
                self.aff_scale_const.requires_grad = False
            elif self.affinity == 'TGASS':
                self.aff_scale_const = nn.Parameter(
                    self.affinity_gamma * self.num * torch.ones(1))
            else:
                self.aff_scale_const = nn.Parameter(torch.ones(1))
                self.aff_scale_const.requires_grad = False
        else:
            raise NotImplementedError

        # Dummy parameters for gathering
        self.w = nn.Parameter(torch.ones((self.ch_f, 1, self.k_f, self.k_f)))
        self.b = nn.Parameter(torch.zeros(self.ch_f))

        self.w.requires_grad = False
        self.b.requires_grad = False

        self.w_conf = nn.Parameter(torch.ones((1, 1, 1, 1)))
        self.w_conf.requires_grad = False

        self.stride = 1
        self.padding = pad_f
        self.dilation = 1
        self.groups = self.ch_f
        self.deformable_groups = 1
        self.im2col_step = 64

    def _get_offset_affinity(self, guidance, confidence=None, rgb=None):
        B, _, H, W = guidance.shape

        if self.affinity in ['AS', 'ASS', 'TC', 'TGASS']:
            offset_aff = self.conv_offset_aff(guidance)
            o1, o2, aff = torch.chunk(offset_aff, 3, dim=1)

            # Add zero reference offset
            offset = torch.cat((o1, o2), dim=1).view(B, self.num, 2, H, W)
            list_offset = list(torch.chunk(offset, self.num, dim=1))
            list_offset.insert(self.idx_ref,
                               torch.zeros((B, 1, 2, H, W)).type_as(offset))
            offset = torch.cat(list_offset, dim=1).view(B, -1, H, W)

            if self.affinity in ['AS', 'ASS']:
                pass
            elif self.affinity == 'TC':
                aff = torch.tanh(aff) / self.aff_scale_const
            elif self.affinity == 'TGASS':
                aff = torch.tanh(aff) / (self.aff_scale_const + 1e-8)
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

        # # Apply confidence
        # # TODO : Need more efficient way
        # if self.conf_prop:
        #     list_conf = []
        #     offset_each = torch.chunk(offset, self.num + 1, dim=1)

        #     modulation_dummy = torch.ones((B, 1, H, W)).type_as(offset).detach()

        #     for idx_off in range(0, self.num + 1):
        #         ww = idx_off % self.k_f
        #         hh = idx_off // self.k_f

        #         if ww == (self.k_f - 1) / 2 and hh == (self.k_f - 1) / 2:
        #             continue

        #         offset_tmp = offset_each[idx_off].detach()

        #         # NOTE : Use --legacy option ONLY for the pre-trained models
        #         # for ECCV20 results.
        #         if self.legacy:
        #             offset_tmp[:, 0, :, :] = \
        #                 offset_tmp[:, 0, :, :] + hh - (self.k_f - 1) / 2
        #             offset_tmp[:, 1, :, :] = \
        #                 offset_tmp[:, 1, :, :] + ww - (self.k_f - 1) / 2

        #         conf_tmp = ModulatedDeformConvFunction.apply(
        #             confidence, offset_tmp, modulation_dummy, self.w_conf,
        #             self.b, self.stride, 0, self.dilation, self.groups,
        #             self.deformable_groups, self.im2col_step)
        #         list_conf.append(conf_tmp)

        #     conf_aff = torch.cat(list_conf, dim=1)
        #     aff = aff * conf_aff.contiguous()

        # Affinity normalization
        aff_abs = torch.abs(aff)
        aff_abs_sum = torch.sum(aff_abs, dim=1, keepdim=True) + 1e-4

        if self.affinity in ['ASS', 'TGASS']:
            aff_abs_sum[aff_abs_sum < 1.0] = 1.0

        if self.affinity in ['AS', 'ASS', 'TGASS']:
            aff = aff / aff_abs_sum

        aff_sum = torch.sum(aff, dim=1, keepdim=True)
        aff_ref = 1.0 - aff_sum

        list_aff = list(torch.chunk(aff, self.num, dim=1))
        list_aff.insert(self.idx_ref, aff_ref)
        aff = torch.cat(list_aff, dim=1)

        return offset, aff

    def _propagate_once(self, feat, offset, aff):
        feat = ModulatedDeformConvFunction.apply(
            feat, offset, aff, self.w, self.b, self.stride, self.padding,
            self.dilation, self.groups, self.deformable_groups, self.im2col_step
        )

        return feat

    def forward(self, feat_init, guidance, confidence=None, feat_fix=None,
                rgb=None):
        assert self.ch_g == guidance.shape[1]
        assert self.ch_f == feat_init.shape[1]

        if self.conf_prop:
            assert confidence is not None

        if self.conf_prop:
            offset, aff = self._get_offset_affinity(guidance, confidence, rgb)
        else:
            offset, aff = self._get_offset_affinity(guidance, None, rgb)

        # Propagation
        if self.preserve_input:
            assert feat_init.shape == feat_fix.shape
            mask_fix = torch.sum(feat_fix > 0.0, dim=1, keepdim=True).detach()
            mask_fix = (mask_fix > 0.0).type_as(feat_fix)

        feat_result = feat_init

        list_feat = []

        for k in range(1, self.prop_time + 1):
            # Input preservation for each iteration
            if self.preserve_input:
                feat_result = (1.0 - mask_fix) * feat_result \
                              + mask_fix * feat_fix

            # feat_result = self._propagate_once(feat_result, offset, aff)
            feat_result = self._propagate_once(feat_result*confidence, offset, aff)

            list_feat.append(feat_result)

        return feat_result, list_feat, offset, aff, self.aff_scale_const.data


class NLSPNModel(nn.Module):
    def __init__(
        self,
        prop_kernel=3,
        network='resnet34',
        from_scratch=False,
        conf_prop=True,
        prop_time=18,
        affinity='TGASS',
        affinity_gamma=0.5,
        legacy=False,
        preserve_input=False,
        device=torch.device('cuda'),
        min_predict_depth=1.5
    ):
        super(NLSPNModel, self).__init__()

        self.prop_kernel=prop_kernel
        self.network=network
        self.from_scratch=from_scratch
        self.conf_prop=conf_prop
        self.prop_time=prop_time
        self.affinity=affinity
        self.affinity_gamma=affinity_gamma
        self.legacy=legacy
        self.preserve_input=preserve_input
        self.device = device
        self.min_predict_depth = min_predict_depth

        self.num_neighbors = self.prop_kernel*self.prop_kernel - 1

        # Encoder
        self.conv1_rgb = conv_bn_relu(3, 48, kernel=3, stride=1, padding=1,
                                      bn=False)
        self.conv1_dep = conv_bn_relu(1, 16, kernel=3, stride=1, padding=1,
                                      bn=False)

        if self.network == 'resnet18':
            net = get_resnet18(not self.from_scratch)
        elif self.network == 'resnet34':
            net = get_resnet34(not self.from_scratch)
        else:
            raise NotImplementedError

        # 1/1
        self.conv2 = net.layer1
        # 1/2
        self.conv3 = net.layer2
        # 1/4
        self.conv4 = net.layer3
        # 1/8
        self.conv5 = net.layer4

        del net

        # 1/16
        self.conv6 = conv_bn_relu(512, 512, kernel=3, stride=2, padding=1)

        # Shared Decoder
        # 1/8
        self.dec5 = convt_bn_relu(512, 256, kernel=3, stride=2,
                                  padding=1, output_padding=1)
        # 1/4
        self.dec4 = convt_bn_relu(256+512, 128, kernel=3, stride=2,
                                  padding=1, output_padding=1)
        # 1/2
        self.dec3 = convt_bn_relu(128+256, 64, kernel=3, stride=2,
                                  padding=1, output_padding=1)

        # 1/1
        self.dec2 = convt_bn_relu(64+128, 64, kernel=3, stride=2,
                                  padding=1, output_padding=1)

        # Init Depth Branch
        # 1/1
        self.id_dec1 = conv_bn_relu(64+64, 64, kernel=3, stride=1,
                                    padding=1)
        self.id_dec0 = conv_bn_relu(64+64, 1, kernel=3, stride=1,
                                    padding=1, bn=False, relu=True)

        # Guidance Branch
        # 1/1
        self.gd_dec1 = conv_bn_relu(64+64, 64, kernel=3, stride=1,
                                    padding=1)
        self.gd_dec0 = conv_bn_relu(64+64, self.num_neighbors, kernel=3, stride=1,
                                    padding=1, bn=False, relu=False)

        if self.conf_prop:
            # Confidence Branch
            # Confidence is shared for propagation and mask generation
            # 1/1
            self.cf_dec1 = conv_bn_relu(64+64, 32, kernel=3, stride=1,
                                        padding=1)
            self.cf_dec0 = nn.Sequential(
                nn.Conv2d(32+64, 1, kernel_size=3, stride=1, padding=1),
                nn.Sigmoid()
            )

        self.prop_layer = NLSPN(
            self.num_neighbors, 
            1,
            3,
            self.prop_kernel,
            self.conf_prop,
            self.prop_time,
            self.affinity,
            self.affinity_gamma,
            self.legacy,
            self.preserve_input
            )

        # Set parameter groups
        params = []
        for param in self.named_parameters():
            if param[1].requires_grad:
                params.append(param[1])

        params = nn.ParameterList(params)

        # self.param_groups = [
        #     {'params': params, 'lr': self.args.lr}
        # ]
        
        # Move to device
        self.data_parallel()
        self.to(self.device)

    def _concat(self, fd, fe, dim=1):
        # Decoder feature may have additional padding
        _, _, Hd, Wd = fd.shape
        _, _, He, We = fe.shape

        # Remove additional padding
        if Hd > He:
            h = Hd - He
            fd = fd[:, :, :-h, :]

        if Wd > We:
            w = Wd - We
            fd = fd[:, :, :, :-w]

        f = torch.cat((fd, fe), dim=dim)

        return f

    def forward(self, image, depth):
        rgb = image
        dep = depth

        # Encoding
        fe1_rgb = self.conv1_rgb(rgb)
        fe1_dep = self.conv1_dep(dep)

        fe1 = torch.cat((fe1_rgb, fe1_dep), dim=1)

        fe2 = self.conv2(fe1)
        fe3 = self.conv3(fe2)
        fe4 = self.conv4(fe3)
        fe5 = self.conv5(fe4)
        fe6 = self.conv6(fe5)

        # Shared Decoding
        fd5 = self.dec5(fe6)
        fd4 = self.dec4(self._concat(fd5, fe5))
        fd3 = self.dec3(self._concat(fd4, fe4))
        fd2 = self.dec2(self._concat(fd3, fe3))

        # Init Depth Decoding
        id_fd1 = self.id_dec1(self._concat(fd2, fe2))
        pred_init = self.id_dec0(self._concat(id_fd1, fe1))

        # Guidance Decoding
        gd_fd1 = self.gd_dec1(self._concat(fd2, fe2))
        guide = self.gd_dec0(self._concat(gd_fd1, fe1))

        if self.conf_prop:
            # Confidence Decoding
            cf_fd1 = self.cf_dec1(self._concat(fd2, fe2))
            confidence = self.cf_dec0(self._concat(cf_fd1, fe1))
        else:
            confidence = None

        # Diffusion
        y, y_inter, offset, aff, aff_const = \
            self.prop_layer(pred_init, guide, confidence, dep, rgb)

        # Remove negative depth
        y = torch.clamp(y, min=self.min_predict_depth)

        output = {'pred': y, 'pred_init': pred_init, 'pred_inter': y_inter,
                  'guidance': guide, 'offset': offset, 'aff': aff,
                  'gamma': aff_const, 'confidence': confidence}

        return output
    
    def data_parallel(self):
        '''
        Allows multi-gpu split along batch
        '''
        device_ids = [_ for _ in range(len(os.environ['CUDA_VISIBLE_DEVICES'].split(",")))]
        self = torch.nn.DataParallel(self, device_ids=device_ids)

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
        checkpoint['model_state_dict'] = self.state_dict()

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

        # Restore encoder and decoder weights
        self.load_state_dict(checkpoint['model_state_dict'])

        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Return the current step and optimizer
        return checkpoint['train_step'], optimizer
    
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
                            # (output_depth0_summary / self.max_predict_depth).cpu(),
                            (output_depth0_summary).cpu(),
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
                    # (sparse_depth0_summary / self.max_predict_depth).cpu(),
                    (sparse_depth0_summary).cpu(),
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