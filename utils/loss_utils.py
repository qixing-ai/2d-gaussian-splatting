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

def create_gaussian_kernel_2d(kernel_size, sigma=1.0):
    """
    创建真正的2D高斯滤波核
    Args:
        kernel_size: 核大小（奇数）
        sigma: 高斯分布的标准差
    Returns:
        gaussian_kernel: [1, 1, kernel_size, kernel_size] 的高斯核
    """
    # 确保kernel_size是奇数
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    # 创建坐标网格
    coords = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
    x, y = torch.meshgrid(coords, coords, indexing='ij')
    
    # 计算高斯分布
    gaussian_2d = torch.exp(-(x**2 + y**2) / (2 * sigma**2))
    
    # 归一化使总和为1
    gaussian_2d = gaussian_2d / gaussian_2d.sum()
    
    # 添加batch和channel维度
    return gaussian_2d.unsqueeze(0).unsqueeze(0)

def compute_flatness_weight(gt_image, kernel_size=5, flat_weight=0.1, edge_weight=0.02):
    """
    计算图像平坦度权重图 - 使用多方向梯度检测
    Args:
        gt_image: 真实图像 [C, H, W]
        kernel_size: 计算梯度的核大小
        flat_weight: 平坦区域的权重（强权重）
        edge_weight: 边缘/纹理区域的权重（弱权重）
    Returns:
        weight_map: 权重图 [1, H, W]
    """
    device = gt_image.device  # 获取输入图像的设备
    
    # 转换为灰度图
    if gt_image.shape[0] == 3:
        gray = 0.299 * gt_image[0] + 0.587 * gt_image[1] + 0.114 * gt_image[2]
    else:
        gray = gt_image[0]
    
    gray_expanded = gray.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    
    # 改进1: 使用多方向梯度检测，包括对角线方向
    # Sobel算子 - 水平和垂直
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
    
    # 对角线方向的边缘检测
    diagonal_1 = torch.tensor([[-2, -1, 0], [-1, 0, 1], [0, 1, 2]], dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
    diagonal_2 = torch.tensor([[0, 1, 2], [-1, 0, 1], [-2, -1, 0]], dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
    
    # 计算多方向梯度
    grad_x = F.conv2d(gray_expanded, sobel_x, padding=1)
    grad_y = F.conv2d(gray_expanded, sobel_y, padding=1)
    grad_d1 = F.conv2d(gray_expanded, diagonal_1, padding=1)
    grad_d2 = F.conv2d(gray_expanded, diagonal_2, padding=1)
    
    # 改进2: 计算综合梯度幅值，考虑所有方向
    gradient_magnitude = torch.sqrt(grad_x**2 + grad_y**2 + 0.5 * (grad_d1**2 + grad_d2**2)).squeeze(0)  # [1, H, W]
    
    # 改进3: 使用更大的高斯核进行平滑，减少噪声影响
    gaussian_kernel = create_gaussian_kernel_2d(kernel_size, sigma=kernel_size/3.0).to(device)
    gradient_magnitude = F.conv2d(gradient_magnitude.unsqueeze(0), gaussian_kernel, padding=kernel_size//2).squeeze(0)
    
    # 改进4: 使用自适应阈值，基于图像的梯度分布
    grad_mean = gradient_magnitude.mean()
    grad_std = gradient_magnitude.std()
    
    # 使用统计信息进行更智能的归一化
    # 设置动态阈值：均值 + 0.5*标准差作为边缘阈值
    edge_threshold = grad_mean + 0.5 * grad_std
    flat_threshold = grad_mean - 0.5 * grad_std
    
    # 改进5: 使用分段函数进行更清晰的区域划分
    # 创建三个区域：明确平坦、明确边缘、过渡区域
    flatness_score = torch.zeros_like(gradient_magnitude)
    
    # 明确平坦区域 (梯度 < flat_threshold)
    flat_mask = gradient_magnitude < flat_threshold
    flatness_score[flat_mask] = 1.0
    
    # 明确边缘区域 (梯度 > edge_threshold)  
    edge_mask = gradient_magnitude > edge_threshold
    flatness_score[edge_mask] = 0.0
    
    # 过渡区域 (flat_threshold <= 梯度 <= edge_threshold)
    transition_mask = ~(flat_mask | edge_mask)
    if transition_mask.any():
        transition_values = gradient_magnitude[transition_mask]
        # 在过渡区域使用平滑插值
        normalized_transition = (edge_threshold - transition_values) / (edge_threshold - flat_threshold)
        flatness_score[transition_mask] = torch.clamp(normalized_transition, 0.0, 1.0)
    
    # 改进6: 应用形态学操作，进一步清理边缘检测结果
    # 使用小的形态学核来平滑权重图
    morph_kernel = torch.ones(3, 3, device=device).unsqueeze(0).unsqueeze(0) / 9.0
    flatness_score = F.conv2d(flatness_score.unsqueeze(0), morph_kernel, padding=1).squeeze(0)
    
    # 计算最终权重
    weight_map = edge_weight + (flat_weight - edge_weight) * flatness_score
    
    return weight_map

# 添加全局缓存字典
_flatness_weight_cache = {}

def precompute_flatness_weights(viewpoint_stack, kernel_size=7, flat_weight=0.1, edge_weight=0.02):
    """
    预计算所有视角的平坦区域权重
    Args:
        viewpoint_stack: 视角列表
        kernel_size: 计算梯度的核大小
        flat_weight: 平坦区域的权重
        edge_weight: 边缘区域的权重
    """
    global _flatness_weight_cache
    _flatness_weight_cache.clear()
    
    print("预计算平坦区域权重...")
    for i, viewpoint_cam in enumerate(viewpoint_stack):
        # 确保视角有正确的ID
        viewpoint_cam.id = i
        gt_image = viewpoint_cam.original_image
        # 确保图像在CUDA设备上
        if not gt_image.is_cuda:
            gt_image = gt_image.cuda()
        weight = compute_flatness_weight(
            gt_image,
            kernel_size=kernel_size,
            flat_weight=flat_weight,
            edge_weight=edge_weight
        )
        _flatness_weight_cache[i] = weight
    print(f"完成 {len(viewpoint_stack)} 个视角的权重预计算")

def get_cached_flatness_weight(viewpoint_idx):
    """
    获取预计算的平坦区域权重
    Args:
        viewpoint_idx: 视角索引
    Returns:
        weight_map: 预计算的权重图 [1, H, W]
    """
    global _flatness_weight_cache
    if viewpoint_idx not in _flatness_weight_cache:
        raise ValueError(f"视角 {viewpoint_idx} 的平坦区域权重未预计算")
    return _flatness_weight_cache[viewpoint_idx]

def compute_training_losses(render_pkg, gt_image, viewpoint_cam, opt, iteration, scene_radius=None):
    """
    计算训练过程中的所有损失（包含自适应法线一致性损失）
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
        
    lambda_alpha = opt.lambda_alpha if iteration > 100 else 0.0
    
    # 动态调整深度收敛损失权重 - 避免主导总损失
    base_lambda_converge = getattr(opt, 'lambda_converge', 0.5)
    
    # 自适应法线损失 - 替代原有的全局lambda_normal
    rend_normal = render_pkg['rend_normal']
    surf_normal = render_pkg['surf_normal']
    
    # 计算基础法线误差
    normal_error = (1 - (rend_normal * surf_normal).sum(dim=0))  # [H, W]
    
    # 使用预计算的权重图
    try:
        if not hasattr(viewpoint_cam, 'id'):
            raise ValueError("视角没有ID属性")
        adaptive_weights = get_cached_flatness_weight(viewpoint_cam.id)
    except (ValueError, AttributeError) as e:
        # 如果缓存未命中，回退到实时计算
        print(f"警告：视角 {getattr(viewpoint_cam, 'id', 'unknown')} 的权重未预计算，使用实时计算")
        adaptive_weights = compute_flatness_weight(
            gt_image, 
            kernel_size=getattr(opt, 'flatness_kernel_size', 7),
            flat_weight=getattr(opt, 'flat_normal_weight', 0.1),
            edge_weight=getattr(opt, 'edge_normal_weight', 0.02)
        )
        # 将新计算的权重添加到缓存中
        if hasattr(viewpoint_cam, 'id'):
            _flatness_weight_cache[viewpoint_cam.id] = adaptive_weights
    
    # 应用自适应权重
    weighted_normal_error = normal_error * adaptive_weights.squeeze(0)  # [H, W]
    normal_loss = weighted_normal_error.mean()
    
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
        'lambda_alpha': lambda_alpha,
        'lambda_converge': lambda_converge,  # 返回实际使用的权重
        'adaptive_normal_weights': adaptive_weights.mean().item(),  # 返回平均自适应权重用于监控
    }

def compute_training_losses_with_depth_correction(render_pkg, gt_image, viewpoint_cam, opt, iteration, scene_radius=None):
    """
    兼容性函数：计算包含深度校正的训练损失
    这个函数是为了保持向后兼容性，实际上调用的是compute_training_losses
    """
    return compute_training_losses(render_pkg, gt_image, viewpoint_cam, opt, iteration, scene_radius)

