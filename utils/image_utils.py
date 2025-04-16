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
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np

def mse(img1, img2):
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)

def psnr(img1, img2):
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def gradient_map(image):
    """
    计算图像的梯度图，支持多通道输入
    
    Args:
        image: 输入图像 [B, C, H, W]
        
    Returns:
        magnitude: 梯度幅度 [B, 1, H, W]
    """
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).float().unsqueeze(0).unsqueeze(0).cuda()/4
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).float().unsqueeze(0).unsqueeze(0).cuda()/4
    
    B, C, H, W = image.shape
    
    # 如果是多通道图像（如RGB），先转为灰度
    if C > 1:
        # 使用RGB到灰度的标准转换公式
        gray = image[:, 0, ...] * 0.299 + image[:, 1, ...] * 0.587 + image[:, 2, ...] * 0.114
        image_gray = gray.unsqueeze(1)  # [B, 1, H, W]
    else:
        image_gray = image
    
    # 使用灰度图计算梯度
    grad_x = torch.cat([F.conv2d(image_gray[i].unsqueeze(0), sobel_x, padding=1) for i in range(B)])
    grad_y = torch.cat([F.conv2d(image_gray[i].unsqueeze(0), sobel_y, padding=1) for i in range(B)])
    
    magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2)
    magnitude = magnitude.norm(dim=0, keepdim=True)

    return magnitude

def colormap(map_np, cmap="turbo", device=None):
    """Applies a colormap to a numpy array and returns a PyTorch tensor."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Default device

    # 获取 matplotlib colormap 颜色，并将其转换为 PyTorch 张量
    try:
        # 使用 get_cmap 获取 colormap 对象
        cmap_object = plt.cm.get_cmap(cmap)
        # 从 colormap 对象获取颜色，这通常是一个 (N, 3) 或 (N, 4) 的 numpy 数组
        cmap_colors_np = cmap_object(np.linspace(0, 1, 256))[:, :3] # 取 RGB 通道
        # 将 numpy 颜色数组转换为 PyTorch 张量，并移动到目标设备
        colors = torch.tensor(cmap_colors_np, dtype=torch.float32).to(device)
    except Exception as e:
        print(f"Error getting/converting colormap '{cmap}': {e}")
        # 回退到简单的灰度或返回错误
        # 这里我们返回一个灰度表示，或者你可以抛出异常
        map_tensor = torch.tensor(map_np, dtype=torch.float32, device=device)
        map_tensor = (map_tensor - map_tensor.min()) / (map_tensor.max() - map_tensor.min() + 1e-6)
        return map_tensor.unsqueeze(0).repeat(3, 1, 1) # [3, H, W]

    # 归一化输入 numpy 数组
    map_normalized = (map_np - map_np.min()) / (map_np.max() - map_np.min() + 1e-6) # 添加 epsilon 防止除以零
    map_indices = (map_normalized * (colors.shape[0] - 1)).round().astype(np.int64) # 确保索引是整数

    # 检查索引是否在范围内
    map_indices = np.clip(map_indices, 0, colors.shape[0] - 1)

    # 使用索引从颜色张量中选取颜色
    # map_indices 可能是 [H, W]，colors 是 [N, 3]
    # 我们需要将 map_indices 变平，应用索引，然后重塑
    # 注意：Numpy 索引在 PyTorch 张量上通常很慢，但这里 colormap 张量不大
    mapped_colors = colors[torch.tensor(map_indices, device=device)] # [H, W, 3]

    # 调整维度顺序为 [C, H, W]
    map_colored_tensor = mapped_colors.permute(2, 0, 1) # [3, H, W]

    return map_colored_tensor

def render_net_image(render_pkg, render_items, render_mode, camera):
    output = render_items[render_mode].lower()
    if output == 'alpha':
        net_image = render_pkg["rend_alpha"]
    elif output == 'normal':
        net_image = render_pkg["rend_normal"]
        net_image = (net_image+1)/2
    elif output == 'depth':
        net_image = render_pkg["surf_depth"]
    elif output == 'edge':
        net_image = gradient_map(render_pkg["render"])
    elif output == 'curvature':
        net_image = render_pkg["rend_normal"]
        net_image = (net_image+1)/2
        net_image = gradient_map(net_image)
    else:
        net_image = render_pkg["render"]

    if net_image.shape[0]==1:
        # 将张量移动到 CPU 并转换为 NumPy 以便 colormap 函数使用
        # 传递原始设备信息
        net_image_np = net_image.squeeze(0).cpu().numpy()
        net_image = colormap(net_image_np, device=net_image.device)
    return net_image