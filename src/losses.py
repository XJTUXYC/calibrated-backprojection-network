'''
Author: Alex Wong <alexw@cs.ucla.edu>

If you use this code, please cite the following paper:

A. Wong, and S. Soatto. Unsupervised Depth Completion with Calibrated Backprojection Layers.
https://arxiv.org/pdf/2108.10531.pdf

@inproceedings{wong2021unsupervised,
  title={Unsupervised Depth Completion with Calibrated Backprojection Layers},
  author={Wong, Alex and Soatto, Stefano},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={12747--12756},
  year={2021}
}
'''
import torch
import log_utils, losses, networks, net_utils

EPSILON = 1e-8

def compute_loss(   image0,
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

def color_consistency_loss_func(src, tgt, w, need_w, use_pytorch_impl=False):
    '''
    Computes the color consistency loss

    Arg(s):
        src : torch.Tensor[float32]
            N x 3 x H x W source image
        tgt : torch.Tensor[float32]
            N x 3 x H x W target image
        w : torch.Tensor[float32]
            N x 1 x H x W weights
    Returns:
        torch.Tensor[float32] : mean absolute difference between source and target images
    '''

    if need_w == False:
        loss = torch.sum(w * torch.abs(tgt - src), dim=[1, 2, 3])
        loss = torch.mean(loss / torch.sum(w, dim=[1, 2, 3]))
    else:
        loss = torch.mean((1-w) * torch.abs(tgt - src) + w)
        
    return loss

def structural_consistency_loss_func(src, tgt, w, need_w):
    '''
    Computes the structural consistency loss using SSIM

    Arg(s):
        src : torch.Tensor[float32]
            N x 3 x H x W source image
        tgt : torch.Tensor[float32]
            N x 3 x H x W target image
        w : torch.Tensor[float32]
            N x 3 x H x W weights
    Returns:
        torch.Tensor[float32] : mean 1 - SSIM scores between source and target images
    '''

    scores_loss = ssim(src, tgt)
    scores_loss = torch.nn.functional.interpolate(scores_loss, size=w.shape[2:4], mode='nearest')
    
    if need_w == False:
        loss = torch.sum(w * scores_loss, dim=[1, 2, 3])
        loss = torch.mean(loss / torch.sum(w, dim=[1, 2, 3]))
    else:
        loss = torch.mean((1-w) * scores_loss + w)
    
    return loss

def sparse_depth_consistency_loss_func(src, tgt, w):
    '''
    Computes the sparse depth consistency loss

    Arg(s):
        src : torch.Tensor[float32]
            N x 1 x H x W source depth
        tgt : torch.Tensor[float32]
            N x 1 x H x W target depth
        w : torch.Tensor[float32]
            N x 1 x H x W weights
    Returns:
        torch.Tensor[float32] : mean absolute difference between source and target depth
    '''

    delta = torch.abs(tgt - src)
    loss = torch.sum(w * delta, dim=[1, 2, 3])

    return torch.mean(loss / torch.sum(w, dim=[1, 2, 3]))

def smoothness_loss_func(predict, image):
    '''
    Computes the local smoothness loss

    Arg(s):
        predict : torch.Tensor[float32]
            N x 1 x H x W predictions
        image : torch.Tensor[float32]
            N x 3 x H x W RGB image
    Returns:
        torch.Tensor[float32] : mean SSIM distance between source and target images
    '''

    predict_dy, predict_dx = gradient_yx(predict)
    image_dy, image_dx = gradient_yx(image)

    # Create edge awareness weights
    weights_x = torch.exp(-torch.mean(torch.abs(image_dx), dim=1, keepdim=True))
    weights_y = torch.exp(-torch.mean(torch.abs(image_dy), dim=1, keepdim=True))

    smoothness_x = torch.mean(weights_x * torch.abs(predict_dx))
    smoothness_y = torch.mean(weights_y * torch.abs(predict_dy))

    return smoothness_x + smoothness_y


'''
Helper functions for constructing loss functions
'''
def gradient_yx(T):
    '''
    Computes gradients in the y and x directions

    Arg(s):
        T : torch.Tensor[float32]
            N x C x H x W tensor
    Returns:
        torch.Tensor[float32] : gradients in y direction
        torch.Tensor[float32] : gradients in x direction
    '''

    dx = T[:, :, :, :-1] - T[:, :, :, 1:]
    dy = T[:, :, :-1, :] - T[:, :, 1:, :]
    return dy, dx

def ssim(x, y):
    '''
    Computes Structural Similarity Index distance between two images

    Arg(s):
        x : torch.Tensor[float32]
            N x 3 x H x W RGB image
        y : torch.Tensor[float32]
            N x 3 x H x W RGB image
    Returns:
        torch.Tensor[float32] : SSIM distance between two images
    '''

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu_x = torch.nn.AvgPool2d(3, 1)(x)
    mu_y = torch.nn.AvgPool2d(3, 1)(y)
    mu_xy = mu_x * mu_y
    mu_xx = mu_x ** 2
    mu_yy = mu_y ** 2

    sigma_x = torch.nn.AvgPool2d(3, 1)(x ** 2) - mu_xx
    sigma_y = torch.nn.AvgPool2d(3, 1)(y ** 2) - mu_yy
    sigma_xy = torch.nn.AvgPool2d(3, 1)(x * y) - mu_xy

    numer = (2 * mu_xy + C1)*(2 * sigma_xy + C2)
    denom = (mu_xx + mu_yy + C1) * (sigma_x + sigma_y + C2)
    score = numer / denom

    return torch.clamp((1.0 - score) / 2.0, 0.0, 1.0)
