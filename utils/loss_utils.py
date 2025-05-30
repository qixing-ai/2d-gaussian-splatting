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

def compute_adaptive_normal_weights(image, flat_weight=0.1, edge_weight=1.0, threshold=0.1):
    """
    基于图像梯度的自适应法线一致性算法
    识别图像中平坦区域和纹理丰富/边缘区域，使用不同权重值
    
    Args:
        image: 输入图像 [C, H, W]
        flat_weight: 平坦区域的权重
        edge_weight: 边缘/纹理区域的权重
        threshold: 梯度阈值，用于区分平坦和边缘区域
    Returns:
        权重图 [1, H, W]
    """
    # 确保图像有正确的维度
    if image.dim() == 3:
        image = image.unsqueeze(0)  # [1, C, H, W]
    
    # 转换为灰度图像用于梯度计算
    if image.size(1) == 3:  # RGB图像
        gray = 0.299 * image[:, 0:1] + 0.587 * image[:, 1:2] + 0.114 * image[:, 2:3]
    else:
        gray = image[:, 0:1]  # 已经是单通道
    
    # 计算Sobel梯度
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                          dtype=torch.float32, device=image.device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                          dtype=torch.float32, device=image.device).view(1, 1, 3, 3)
    
    # 计算梯度
    grad_x = F.conv2d(gray, sobel_x, padding=1)
    grad_y = F.conv2d(gray, sobel_y, padding=1)
    
    # 计算梯度幅值
    gradient_magnitude = torch.sqrt(grad_x.pow(2) + grad_y.pow(2) + 1e-8)
    
    # 应用高斯平滑减少噪声
    gaussian_kernel = torch.tensor([[1, 2, 1], [2, 4, 2], [1, 2, 1]], 
                                  dtype=torch.float32, device=image.device).view(1, 1, 3, 3) / 16.0
    gradient_magnitude = F.conv2d(gradient_magnitude, gaussian_kernel, padding=1)
    
    # 自适应阈值：使用梯度的百分位数
    adaptive_threshold = torch.quantile(gradient_magnitude.flatten(), 0.7)
    threshold = max(threshold, adaptive_threshold.item())
    
    # 创建权重图：高梯度区域使用edge_weight，低梯度区域使用flat_weight
    weight_map = torch.where(gradient_magnitude > threshold, 
                           torch.tensor(edge_weight, device=image.device),
                           torch.tensor(flat_weight, device=image.device))
    
    # 平滑权重过渡，避免突变
    weight_map = F.conv2d(weight_map, gaussian_kernel, padding=1)
    
    return weight_map.squeeze(0)  # 返回 [1, H, W]

def compute_training_losses(render_pkg, gt_image, viewpoint_cam, opt, iteration):
    """
    计算训练过程中的所有损失
    Args:
        render_pkg: 渲染结果包
        gt_image: 真实图像
        viewpoint_cam: 视角相机
        opt: 优化参数
        iteration: 当前迭代次数
    Returns:
        dict: 包含各种损失的字典
    """
    # 基础重建损失
    image = render_pkg["render"]
    Ll1 = l1_loss(image, gt_image)
    ms_ssim_loss_val = ms_ssim_loss(image, gt_image)
    loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * ms_ssim_loss_val
    
    # 计算自适应权重（始终计算，用于监控）
    adaptive_weights = compute_adaptive_normal_weights(
        gt_image, 
        flat_weight=getattr(opt, 'normal_flat_weight', 0.1),
        edge_weight=getattr(opt, 'normal_edge_weight', 1.0),
        threshold=getattr(opt, 'normal_gradient_threshold', 0.1)
    )
    avg_adaptive_weight = adaptive_weights.mean().item()
    
    # 2阶段法线一致性策略：完全去除时间衰减机制
    adaptive_start_iter = getattr(opt, 'adaptive_normal_start_iter', getattr(opt, 'normal_decay_start_iter', 1000))
    
    if iteration <= adaptive_start_iter:
        # 阶段1：使用固定的法线一致性权重
        lambda_normal = opt.lambda_normal
        use_adaptive_weights = False
    else:
        # 阶段2：lambda_normal设为0，完全使用自适应权重
        lambda_normal = 0.0
        use_adaptive_weights = True
        
    lambda_alpha = opt.lambda_alpha if iteration > 100 else 0.0
    
    # 法线损失
    rend_normal = render_pkg['rend_normal']
    surf_normal = render_pkg['surf_normal']
    normal_error = (1 - (rend_normal * surf_normal).sum(dim=0))[None]
    
    if use_adaptive_weights:
        # 阶段2：使用自适应权重进行像素级调节，使用专门的权重参数
        lambda_adaptive_normal = opt.lambda_adaptive_normal
        weighted_normal_error = normal_error * adaptive_weights
        normal_loss = lambda_adaptive_normal * weighted_normal_error.mean()
    else:
        # 阶段1：使用固定权重
        normal_loss = lambda_normal * normal_error.mean()
    
    # Alpha损失
    alpha_loss = torch.tensor(0.0, device="cuda")
    if hasattr(viewpoint_cam, 'gt_alpha_mask') and lambda_alpha > 0 and viewpoint_cam.gt_alpha_mask is not None:
        gt_alpha = viewpoint_cam.gt_alpha_mask
        rend_alpha = render_pkg['rend_alpha']
        bg_region = (1.0 - gt_alpha)
        alpha_loss = lambda_alpha * (rend_alpha * bg_region).mean()
    
    # 深度收敛损失
    convergence_loss = torch.tensor(0.0, device="cuda")
    if opt.lambda_converge > 0:
        convergence_loss = opt.lambda_converge * compute_depth_convergence_loss(render_pkg, viewpoint_cam)
    
    total_loss = loss + normal_loss + alpha_loss + convergence_loss
    
    return {
        'total_loss': total_loss,
        'l1_loss': Ll1,
        'ms_ssim_loss': ms_ssim_loss_val,
        'reconstruction_loss': loss,
        'normal_loss': normal_loss,
        'alpha_loss': alpha_loss,
        'convergence_loss': convergence_loss,
        'lambda_normal': lambda_normal,
        'lambda_alpha': lambda_alpha,
        'adaptive_normal_weight': avg_adaptive_weight,
        'adaptive_stage': 2 if use_adaptive_weights else 1
    }

def compute_depth_convergence_loss(render_pkg, viewpoint_cam, k=1.25):
    """
    计算深度收敛损失 - 简化稳定版本
    基于论文公式：L_converge = Σ min(Ĝ_i(x), Ĝ_{i-1}(x)) * D_i
    其中 D_i = (d_i - d_{i-1})^2
    
    Args:
        render_pkg: 渲染结果包
        viewpoint_cam: 视角相机
        k: 梯度缩放因子
    Returns:
        深度收敛损失值
    """
    # 获取表面深度和透明度
    surf_depth = render_pkg['surf_depth']  # [1, H, W]
    rend_alpha = render_pkg['rend_alpha']  # [1, H, W]
    
    # 处理无效深度值
    surf_depth = torch.nan_to_num(surf_depth, nan=0.0, posinf=0.0, neginf=0.0)
    
    # 简单的深度平滑
    kernel = torch.ones(1, 1, 3, 3, device=surf_depth.device) / 9.0
    surf_depth_smooth = torch.nn.functional.conv2d(surf_depth, kernel, padding=1)
    
    # 计算相邻像素的深度差
    depth_diff_x = surf_depth_smooth[:, :, 1:] - surf_depth_smooth[:, :, :-1]  # [1, H, W-1]
    depth_diff_y = surf_depth_smooth[:, 1:, :] - surf_depth_smooth[:, :-1, :]  # [1, H-1, W]
    
    # 使用Huber损失来减少极值的影响
    delta = 0.1  # Huber损失的阈值
    
    def huber_loss(x, delta):
        abs_x = torch.abs(x)
        return torch.where(abs_x <= delta, 0.5 * x.pow(2), delta * (abs_x - 0.5 * delta))
    
    depth_loss_x = huber_loss(depth_diff_x, delta)
    depth_loss_y = huber_loss(depth_diff_y, delta)
    
    # 计算权重（使用alpha作为Gaussian值的近似）
    alpha_weight_x = torch.min(rend_alpha[:, :, 1:], rend_alpha[:, :, :-1]).detach()
    alpha_weight_y = torch.min(rend_alpha[:, 1:, :], rend_alpha[:, :-1, :]).detach()
    
    # 使用sigmoid来平滑权重，避免突然的0值
    alpha_weight_x = torch.sigmoid(alpha_weight_x * 10)  # 放大后应用sigmoid
    alpha_weight_y = torch.sigmoid(alpha_weight_y * 10)
    
    # 计算加权损失
    convergence_loss_x = (alpha_weight_x * depth_loss_x).mean()
    convergence_loss_y = (alpha_weight_y * depth_loss_y).mean()
    
    # 总的深度收敛损失
    convergence_loss = k * (convergence_loss_x + convergence_loss_y)
    
    # 数值稳定性检查
    if torch.isnan(convergence_loss) or torch.isinf(convergence_loss):
        convergence_loss = torch.tensor(0.0, device=convergence_loss.device)
    
    return convergence_loss

