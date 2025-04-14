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
from pytorch_msssim import ms_ssim  # 导入多尺度SSIM
from utils.image_utils import gradient_map # 导入图像梯度计算函数

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def smooth_loss(disp, img):
    grad_disp_x = torch.abs(disp[:,1:-1, :-2] + disp[:,1:-1,2:] - 2 * disp[:,1:-1,1:-1])
    grad_disp_y = torch.abs(disp[:,:-2, 1:-1] + disp[:,2:,1:-1] - 2 * disp[:,1:-1,1:-1])
    grad_img_x = torch.mean(torch.abs(img[:, 1:-1, :-2] - img[:, 1:-1, 2:]), 0, keepdim=True) * 0.5
    grad_img_y = torch.mean(torch.abs(img[:, :-2, 1:-1] - img[:, 2:, 1:-1]), 0, keepdim=True) * 0.5
    grad_disp_x *= torch.exp(-grad_img_x)
    grad_disp_y *= torch.exp(-grad_img_y)
    return grad_disp_x.mean() + grad_disp_y.mean()

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

def compute_color_loss(rendered_img, gt_img, lambda_dssim=0.2, use_ms_ssim=True):
    """
    计算颜色损失，支持普通SSIM和多尺度SSIM
    
    Args:
        rendered_img: 渲染图像
        gt_img: 真值图像
        lambda_dssim: SSIM在损失中的权重
        use_ms_ssim: 是否使用多尺度SSIM
        
    Returns:
        color_loss: 颜色损失
    """
    l1_loss_val = l1_loss(rendered_img, gt_img)
    
    if use_ms_ssim:
        # 确保图像尺寸符合MS-SSIM要求（至少大于48×48）
        if rendered_img.shape[1] >= 48 and rendered_img.shape[2] >= 48:
            # 5个尺度的权重，推荐值
            weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
            ms_ssim_loss = 1.0 - ms_ssim(rendered_img.unsqueeze(0), 
                                        gt_img.unsqueeze(0), 
                                        data_range=1.0, 
                                        weights=weights)
            ssim_loss = ms_ssim_loss
        else:
            # 如果图像太小，回退到普通SSIM
            ssim_loss = 1.0 - ssim(rendered_img, gt_img)
    else:
        # 使用普通SSIM
        ssim_loss = 1.0 - ssim(rendered_img, gt_img)
    
    return (1.0 - lambda_dssim) * l1_loss_val + lambda_dssim * ssim_loss

def edge_aware_normal_loss(rendered_normal, gt_rgb, surf_normal, q=4, lambda_consistency=0.5):
    """
    边缘感知法向损失
    
    Args:
        rendered_normal: 渲染的法线图 [3, H, W]
        gt_rgb: 真值RGB图像 [3, H, W]
        surf_normal: 表面法线 [3, H, W]
        q: 边缘权重指数
        lambda_consistency: 原始法线一致性的权重
        
    Returns:
        loss: 边缘感知法向损失
    """
    # 确保输入格式正确
    if len(gt_rgb.shape) == 3:  # [3, H, W]
        gt_rgb = gt_rgb.unsqueeze(0)  # 变为 [1, 3, H, W]
    
    # 计算边缘图
    grad_rgb = gradient_map(gt_rgb)  # [1, 1, H, W]
    
    # 确保法线格式正确
    if len(rendered_normal.shape) == 3:  # [3, H, W]
        rendered_normal_norm = ((rendered_normal + 1) / 2).unsqueeze(0)  # 变为 [1, 3, H, W]，并归一化到[0,1]
    else:
        rendered_normal_norm = (rendered_normal + 1) / 2
    
    # 计算法线曲率 |∇N|
    grad_normal = gradient_map(rendered_normal_norm)  # [1, 1, H, W]
    
    # 确保权重正确
    weight = torch.clamp(grad_rgb.squeeze(), 0.0, 1.0) ** q  # [H, W]
    
    # 加权损失：边缘区域（权重小）放松约束，平坦区域（权重大）强约束
    edge_loss = (grad_normal.squeeze() * (1.0 - weight)).mean()
    
    # 原始法线一致性损失
    consistency_loss = (1.0 - (rendered_normal * surf_normal).sum(dim=0)).mean()
    
    # 组合损失
    loss = edge_loss + lambda_consistency * consistency_loss
    
    return loss

def depth_convergence_loss(render_depth, render_alpha):
    """
    深度收敛损失：强制相邻高斯基元深度接近
    L_converge = ∑min(G_i, G_(i-1))⋅(d_i - d_(i-1))^2
    
    Args:
        render_depth: 渲染深度图 [1, H, W]
        render_alpha: 不透明度/权重图 [1, H, W]
        
    Returns:
        loss: 深度收敛损失
    """
    # 确保输入格式正确
    if len(render_depth.shape) != 3 or render_depth.shape[0] != 1:
        raise ValueError(f"深度图应形如 [1, H, W]，但得到 {render_depth.shape}")
    
    # 计算深度的水平和垂直梯度
    # 水平方向深度差
    depth_grad_x = render_depth[:, :, 1:] - render_depth[:, :, :-1]
    # 垂直方向深度差
    depth_grad_y = render_depth[:, 1:, :] - render_depth[:, :-1, :]
    
    # 计算对应的权重（取相邻像素不透明度的最小值）
    # 水平方向权重
    weight_x = torch.min(render_alpha[:, :, 1:], render_alpha[:, :, :-1])
    # 垂直方向权重
    weight_y = torch.min(render_alpha[:, 1:, :], render_alpha[:, :-1, :])
    
    # 计算加权平方差
    loss_x = (weight_x * depth_grad_x.pow(2)).sum()
    loss_y = (weight_y * depth_grad_y.pow(2)).sum()
    
    # 防止除零问题（当像素总数为0时）
    num_pixels_x = weight_x.sum()
    num_pixels_y = weight_y.sum()
    
    # 取平均值
    if num_pixels_x > 0:
        loss_x = loss_x / num_pixels_x
    if num_pixels_y > 0:
        loss_y = loss_y / num_pixels_y
    
    # 组合两个方向的损失
    loss = loss_x + loss_y
    
    return loss

def background_loss(render_alpha, gt_image, threshold=0.1, debug=False):
    """
    背景损失：强制透明背景区域的高斯不透明度趋近于0
    L_bg = (1/hw) * ∑[A_i * (1-M_i)]
    
    Args:
        render_alpha: 渲染的alpha通道 [1, H, W]
        gt_image: 真实图像 [3, H, W]
        threshold: 前景/背景阈值，默认0.1
        debug: 是否返回调试信息
        
    Returns:
        loss: 背景损失，如果debug=True则返回额外的调试信息
    """
    # 确保输入格式正确
    if len(render_alpha.shape) != 3 or render_alpha.shape[0] != 1:
        raise ValueError(f"Alpha图应形如 [1, H, W]，但得到 {render_alpha.shape}")
    
    # 计算真实图像的平均亮度
    image_brightness = gt_image.mean(dim=0, keepdim=True)
    
    # 使用自适应阈值方法
    # 首先尝试使用固定阈值
    foreground_mask = (image_brightness > threshold).float()
    bg_pixel_ratio = (1.0 - foreground_mask).sum() / (foreground_mask.shape[1] * foreground_mask.shape[2])
    
    # 准备调试信息字典
    adaptive_threshold = None
    
    # 如果背景像素比例太高（大于90%），调整阈值以确保至少有10%的前景像素
    if bg_pixel_ratio > 0.90:  # 降低阈值，使背景区域更小
        # 使用图像亮度的百分位数作为自适应阈值
        flat_brightness = image_brightness.view(-1)
        sorted_brightness, _ = torch.sort(flat_brightness)
        adaptive_idx = int(0.10 * len(sorted_brightness))  # 增加到10%的前景像素
        adaptive_threshold = sorted_brightness[adaptive_idx]
        
        # 应用自适应阈值
        foreground_mask = (image_brightness > adaptive_threshold).float()
    
    # 背景区域为 (1-M_i)
    background_mask = 1.0 - foreground_mask
    
    # 计算背景区域的高斯不透明度损失
    # 使用指数惩罚，使背景区域的alpha更接近0
    bg_alpha = render_alpha * background_mask
    bg_loss = torch.exp(bg_alpha).mean() - 1.0  # 指数惩罚
    
    if debug:
        debug_info = {
            'foreground_mask': foreground_mask,
            'background_mask': background_mask,
            'bg_alpha_avg': (render_alpha * background_mask).sum() / (background_mask.sum() + 1e-6),
            'fg_alpha_avg': (render_alpha * foreground_mask).sum() / (foreground_mask.sum() + 1e-6),
            'bg_pixel_ratio': background_mask.sum() / (background_mask.shape[1] * background_mask.shape[2])
        }
        
        # 如果使用了自适应阈值，添加到调试信息中
        if adaptive_threshold is not None:
            debug_info['adaptive_threshold'] = adaptive_threshold
        
        return bg_loss, debug_info
    
    return bg_loss

