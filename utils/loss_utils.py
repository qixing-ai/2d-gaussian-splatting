#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
import numpy as np

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

def ms_ssim_loss(img1, img2):
    """
    计算多尺度结构相似性损失
    Args:
        img1: 预测图像 [C, H, W]
        img2: 真实图像 [C, H, W]
    Returns:
        MS-SSIM 损失值
    """
    from pytorch_msssim import ms_ssim
    # 添加批次维度
    img1 = img1.unsqueeze(0)  # [1, C, H, W]
    img2 = img2.unsqueeze(0)  # [1, C, H, W]
    return 1 - ms_ssim(img1, img2, data_range=1.0, size_average=True)

def compute_training_losses(render_pkg, gt_image, viewpoint_cam, opt, iteration, scene_radius=None):
    """
    计算训练过程中的所有损失（包含深度校正）
    Args:
        render_pkg: 渲染结果包
        gt_image: 真实图像
        viewpoint_cam: 视角相机
        opt: 优化参数
        iteration: 当前迭代次数
        scene_radius: 场景半径，用于深度收敛损失
    Returns:
        dict: 包含各种损失的字典
    """
    # 基础重建损失
    image = render_pkg["render"]
    Ll1 = l1_loss(image, gt_image)
    ms_ssim_loss_val = ms_ssim_loss(image, gt_image)
    loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * ms_ssim_loss_val
    
    # 计算lambda_normal
    if iteration <= opt.normal_decay_start_iter or opt.normal_decay_start_iter >= opt.iterations:
        lambda_normal = opt.lambda_normal
    else:
        progress = (iteration - opt.normal_decay_start_iter) / (opt.iterations - opt.normal_decay_start_iter)
        lambda_normal = opt.lambda_normal * np.exp(-5 * progress)
        
    lambda_alpha = opt.lambda_alpha if iteration > 100 else 0.0
    lambda_converge = getattr(opt, 'lambda_converge', 7.0)
    
    # 法线损失
    rend_normal = render_pkg['rend_normal']
    surf_normal = render_pkg['surf_normal']
    normal_error = (1 - (rend_normal * surf_normal).sum(dim=0))[None]
    normal_loss = lambda_normal * normal_error.mean()
    
    # Alpha损失
    alpha_loss = torch.tensor(0.0, device="cuda")
    if hasattr(viewpoint_cam, 'gt_alpha_mask') and lambda_alpha > 0 and viewpoint_cam.gt_alpha_mask is not None:
        gt_alpha = viewpoint_cam.gt_alpha_mask
        rend_alpha = render_pkg['rend_alpha']
        bg_region = (1.0 - gt_alpha)
        alpha_loss = lambda_alpha * (rend_alpha * bg_region).mean()
    
    # 深度收敛损失
    depth_convergence_loss_val = torch.tensor(0.0, device="cuda")
    if 'surf_depth' in render_pkg:
        depth_map = render_pkg['surf_depth']  # [1, H, W]
        # 去除通道维度，得到 [H, W]
        if depth_map.dim() == 3 and depth_map.size(0) == 1:
            depth_map = depth_map.squeeze(0)  # [H, W]
        
        # 计算水平和垂直方向的深度梯度
        depth_grad_x = torch.abs(depth_map[:, 1:] - depth_map[:, :-1])  # [H, W-1]
        depth_grad_y = torch.abs(depth_map[1:, :] - depth_map[:-1, :])  # [H-1, W]
        depth_convergence_loss_val = lambda_converge * (depth_grad_x.mean() + depth_grad_y.mean())
    
    total_loss = loss + normal_loss + alpha_loss + depth_convergence_loss_val
    
    return {
        'total_loss': total_loss,
        'l1_loss': Ll1,
        'ms_ssim_loss': ms_ssim_loss_val,
        'normal_loss': normal_loss,
        'alpha_loss': alpha_loss,
        'depth_convergence_loss': depth_convergence_loss_val,
        'reconstruction_loss': loss,
        'lambda_normal': lambda_normal,
        'lambda_alpha': lambda_alpha,
        'lambda_converge': lambda_converge,
    }

def compute_training_losses_with_depth_correction(render_pkg, gt_image, viewpoint_cam, opt, iteration, scene_radius=None):
    """
    兼容性函数：计算包含深度校正的训练损失
    这个函数是为了保持向后兼容性，实际上调用的是compute_training_losses
    """
    return compute_training_losses(render_pkg, gt_image, viewpoint_cam, opt, iteration, scene_radius)

