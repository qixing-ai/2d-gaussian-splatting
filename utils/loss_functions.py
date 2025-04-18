import torch
from utils.loss_utils import l1_loss, ssim, compute_color_loss, edge_aware_normal_loss, depth_convergence_loss

def calculate_color_loss(image, gt_image, mask, opt):
    """
    计算颜色损失，支持有无mask的情况
    
    Args:
        image: 渲染图像
        gt_image: 真值图像
        mask: 掩码，可以为None
        opt: 优化参数对象
        
    Returns:
        loss: 颜色损失
        Ll1_report: 用于报告的L1损失（未掩码）
    """
    # 计算未掩码的L1损失，用于报告
    Ll1_report = l1_loss(image, gt_image)
    
    # 颜色损失计算
    if mask is not None:
        # Apply mask to L1 calculation
        pixel_count = mask.sum()
        if pixel_count > 0:
            masked_l1 = (torch.abs(image - gt_image) * mask).sum() / (pixel_count * image.shape[0] + 1e-6) # Mean L1 over foreground pixels
        else:
            masked_l1 = torch.tensor(0.0, device=image.device) # Avoid division by zero if mask is empty

        # SSIM calculation (on unmasked images for now)
        if hasattr(opt, 'use_ms_ssim') and opt.use_ms_ssim:
             # Need to ensure ms_ssim is available and imported if used.
             # Assuming ssim function from loss_utils handles both cases based on internal logic or we'd need separate calls.
             # For simplicity, using the standard ssim function here. Modify if ms_ssim is intended.
             ssim_val = ssim(image, gt_image) # Calculate SSIM on full image
             loss = (1.0 - opt.lambda_dssim) * masked_l1 + opt.lambda_dssim * (1.0 - ssim_val)
        else:
             ssim_val = ssim(image, gt_image) # Calculate SSIM on full image
             loss = (1.0 - opt.lambda_dssim) * masked_l1 + opt.lambda_dssim * (1.0 - ssim_val)
    else:
        # Original loss calculation if no mask is available
        # 确定使用哪种SSIM
        use_fused = getattr(opt, 'use_fused_ssim', False)
        use_ms = getattr(opt, 'use_ms_ssim', True) and not use_fused # 只有当fused未启用时，ms才可能生效

        loss = compute_color_loss(image, gt_image,
                                lambda_dssim=opt.lambda_dssim,
                                use_ms_ssim=use_ms,
                                use_fused_ssim=use_fused)
    
    return loss, Ll1_report

def calculate_normal_loss(rend_normal, surf_normal, gt_image, lambda_normal, opt):
    """
    计算法线损失
    
    Args:
        rend_normal: 渲染的法线图
        surf_normal: 表面法线
        gt_image: 真值图像（用于边缘感知法向损失）
        lambda_normal: 法线损失权重
        opt: 优化参数对象
        
    Returns:
        normal_loss: 法线损失
    """
    if lambda_normal > 0:
        # 检查是否启用了边缘感知法向损失
        if hasattr(opt, 'use_edge_aware_normal') and opt.use_edge_aware_normal:
            # 使用边缘感知法向损失
            edge_weight_exp = opt.edge_weight_exponent if hasattr(opt, 'edge_weight_exponent') else 4.0
            lambda_cons = opt.lambda_consistency if hasattr(opt, 'lambda_consistency') else 0.5
            
            normal_loss = lambda_normal * edge_aware_normal_loss(
                rendered_normal=rend_normal,
                gt_rgb=gt_image,
                surf_normal=surf_normal,
                q=edge_weight_exp,
                lambda_consistency=lambda_cons
            )
        else:
            # 使用原始法线一致性损失
            normal_error = (1 - (rend_normal * surf_normal).sum(dim=0))[None]
            normal_loss = lambda_normal * (normal_error).mean()
    else:
        normal_loss = torch.tensor(0.0, device='cuda')
        
    return normal_loss

def calculate_convergence_loss(surf_depth, render_alpha, lambda_conv):
    """
    计算深度收敛损失
    
    Args:
        surf_depth: 表面深度图
        render_alpha: 渲染的透明度
        lambda_conv: 深度收敛损失权重
        
    Returns:
        convergence_loss: 深度收敛损失
    """
    if lambda_conv > 0:
        convergence_loss = lambda_conv * depth_convergence_loss(surf_depth, render_alpha)
    else:
        convergence_loss = torch.tensor(0.0, device='cuda')
        
    return convergence_loss

def calculate_background_opacity_loss(render_alpha, mask, lambda_bg_opacity):
    """
    计算背景透明度损失
    
    Args:
        render_alpha: 渲染的透明度
        mask: 前景掩码
        lambda_bg_opacity: 背景透明度损失权重
        
    Returns:
        background_opacity_loss: 背景透明度损失
    """
    background_opacity_loss = torch.tensor(0.0, device='cuda')
    
    if mask is not None and lambda_bg_opacity > 0:
        background_mask = (1 - mask)
        num_background_pixels = background_mask.sum()
        if num_background_pixels > 0:
            # 计算背景区域 alpha 的 L1 损失均值
            background_opacity_loss = (torch.abs(render_alpha * background_mask)).sum() / num_background_pixels
    
    return background_opacity_loss

def get_lambda_normal(iteration, opt):
    """
    获取法线损失权重，根据迭代次数动态调整
    
    Args:
        iteration: 当前迭代次数
        opt: 优化参数对象
        
    Returns:
        lambda_normal: 法线损失权重
    """
    if iteration <= 3000:
        lambda_normal = 0.0
    elif iteration <= 5000:
        lambda_normal = opt.lambda_normal * (iteration - 3000) / 2000  # 线性增加
    else:
        lambda_normal = opt.lambda_normal
    return lambda_normal

def get_lambda_dist(iteration, opt):
    """
    获取距离损失权重
    
    Args:
        iteration: 当前迭代次数
        opt: 优化参数对象
        
    Returns:
        lambda_dist: 距离损失权重
    """
    return opt.lambda_dist if iteration > 3000 else 0.0

def get_lambda_convergence(iteration, opt):
    """
    获取深度收敛损失权重
    
    Args:
        iteration: 当前迭代次数
        opt: 优化参数对象
        
    Returns:
        lambda_conv: 深度收敛损失权重
    """
    return opt.lambda_depth_convergence if hasattr(opt, 'use_depth_convergence') and opt.use_depth_convergence and iteration > opt.conv_start_iter else 0.0 