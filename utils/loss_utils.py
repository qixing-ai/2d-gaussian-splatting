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

def compute_flatness_weight(gt_image, kernel_size=5, flat_weight=2.0, texture_weight=0.5):
    """
    计算图像平坦度权重图
    Args:
        gt_image: 真实图像 [C, H, W]
        kernel_size: 用于计算梯度的卷积核大小
        flat_weight: 平坦区域的权重
        texture_weight: 纹理区域的权重
    Returns:
        weight_map: 权重图 [1, H, W]
    """
    # 转换为灰度图
    if gt_image.shape[0] == 3:
        gray = 0.299 * gt_image[0] + 0.587 * gt_image[1] + 0.114 * gt_image[2]
    else:
        gray = gt_image[0]
    
    # 计算图像梯度 (Sobel算子)
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=gt_image.device).unsqueeze(0).unsqueeze(0)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=gt_image.device).unsqueeze(0).unsqueeze(0)
    
    gray_padded = F.pad(gray.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode='reflect')
    grad_x = F.conv2d(gray_padded, sobel_x)
    grad_y = F.conv2d(gray_padded, sobel_y)
    
    # 计算梯度幅值
    gradient_magnitude = torch.sqrt(grad_x**2 + grad_y**2 + 1e-8)
    
    # 使用高斯滤波平滑梯度图
    gaussian_kernel = torch.ones(1, 1, kernel_size, kernel_size, device=gt_image.device) / (kernel_size * kernel_size)
    padding = kernel_size // 2
    gradient_smooth = F.conv2d(F.pad(gradient_magnitude, (padding, padding, padding, padding), mode='reflect'), gaussian_kernel)
    
    # 归一化梯度值到[0,1]
    grad_min = gradient_smooth.min()
    grad_max = gradient_smooth.max()
    if grad_max > grad_min:
        gradient_norm = (gradient_smooth - grad_min) / (grad_max - grad_min)
    else:
        gradient_norm = torch.zeros_like(gradient_smooth)
    
    # 计算权重：平坦区域(低梯度)高权重，纹理区域(高梯度)低权重
    # 使用指数函数使权重变化更平滑
    weight_map = flat_weight * torch.exp(-3.0 * gradient_norm) + texture_weight
    
    return weight_map.squeeze(0)  # [1, H, W]

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
    
    # 动态调整深度收敛损失权重 - 避免主导总损失
    base_lambda_converge = getattr(opt, 'lambda_converge', 0.5)
    
    # 自适应法线损失 - 根据图像平坦度调整权重
    rend_normal = render_pkg['rend_normal']
    surf_normal = render_pkg['surf_normal']
    normal_error = (1 - (rend_normal * surf_normal).sum(dim=0))[None]
    
    # 计算自适应权重
    use_adaptive_normal = getattr(opt, 'use_adaptive_normal', True)
    if use_adaptive_normal:
        # 获取平坦度权重图
        flat_weight = getattr(opt, 'normal_flat_weight', 2.0)
        texture_weight = getattr(opt, 'normal_texture_weight', 0.5)
        flatness_weight = compute_flatness_weight(gt_image, flat_weight=flat_weight, texture_weight=texture_weight)
        
        # 应用自适应权重到法线误差
        weighted_normal_error = normal_error * flatness_weight
        normal_loss = lambda_normal * weighted_normal_error.mean()
    else:
        # 原始法线损失
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
    if 'convergence_map' in render_pkg:
        # 使用CUDA计算的真正深度收敛损失 - 按照论文公式实现
        convergence_map = render_pkg['convergence_map']
        raw_depth_loss = convergence_map.mean()
        
        # 动态权重平衡：确保深度损失不会主导总损失
        # 如果深度损失过大，自动降低权重
        reconstruction_magnitude = loss.item()
        depth_magnitude = raw_depth_loss.item()
        
        if depth_magnitude > 0 and reconstruction_magnitude > 0:
            # 计算自适应权重，使深度损失贡献不超过重建损失的50%
            max_depth_contribution = 0.5 * reconstruction_magnitude
            if base_lambda_converge * depth_magnitude > max_depth_contribution:
                adaptive_lambda_converge = max_depth_contribution / depth_magnitude
                adaptive_lambda_converge = min(adaptive_lambda_converge, base_lambda_converge)
            else:
                adaptive_lambda_converge = base_lambda_converge
        else:
            adaptive_lambda_converge = base_lambda_converge
            
        depth_convergence_loss_val = adaptive_lambda_converge * raw_depth_loss
        lambda_converge = adaptive_lambda_converge
        
    elif 'surf_depth' in render_pkg:
        # 备用方案：如果CUDA版本不可用，使用简化的深度梯度版本
        depth_map = render_pkg['surf_depth']
        depth_grad_x = torch.abs(depth_map[:, :, 1:] - depth_map[:, :, :-1])
        depth_grad_y = torch.abs(depth_map[:, 1:, :] - depth_map[:-1, :, :])
        raw_depth_loss = depth_grad_x.mean() + depth_grad_y.mean()
        
        # 同样的自适应权重机制
        reconstruction_magnitude = loss.item()
        depth_magnitude = raw_depth_loss.item()
        
        if depth_magnitude > 0 and reconstruction_magnitude > 0:
            max_depth_contribution = 0.5 * reconstruction_magnitude
            if base_lambda_converge * depth_magnitude > max_depth_contribution:
                adaptive_lambda_converge = max_depth_contribution / depth_magnitude
                adaptive_lambda_converge = min(adaptive_lambda_converge, base_lambda_converge)
            else:
                adaptive_lambda_converge = base_lambda_converge
        else:
            adaptive_lambda_converge = base_lambda_converge
            
        depth_convergence_loss_val = adaptive_lambda_converge * raw_depth_loss
        lambda_converge = adaptive_lambda_converge
    else:
        lambda_converge = base_lambda_converge
    
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
        'lambda_converge': lambda_converge,  # 返回实际使用的权重
    }

def compute_training_losses_with_depth_correction(render_pkg, gt_image, viewpoint_cam, opt, iteration, scene_radius=None):
    """
    兼容性函数：计算包含深度校正的训练损失
    这个函数是为了保持向后兼容性，实际上调用的是compute_training_losses
    """
    return compute_training_losses(render_pkg, gt_image, viewpoint_cam, opt, iteration, scene_radius)

