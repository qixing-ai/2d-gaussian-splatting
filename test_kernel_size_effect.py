#!/usr/bin/env python3
"""
测试 flatness_kernel_size 的精细作用
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from utils.loss_utils import compute_flatness_weight

def create_detailed_test_image():
    """创建包含多种细节层次的测试图像"""
    H, W = 512, 512
    image = torch.zeros(3, H, W, device='cuda')
    
    # 1. 大面积平坦区域 (左上角)
    image[:, 0:H//3, 0:W//3] = 0.5
    
    # 2. 粗纹理区域 (右上角) - 大周期纹理
    h_region, w_region = H//3, W//3
    x_coarse = torch.linspace(0, 4*np.pi, w_region, device='cuda')
    y_coarse = torch.linspace(0, 4*np.pi, h_region, device='cuda')
    X_coarse, Y_coarse = torch.meshgrid(x_coarse, y_coarse, indexing='ij')
    coarse_texture = 0.5 + 0.3 * torch.sin(X_coarse) * torch.cos(Y_coarse)
    image[:, 0:h_region, W//3:W//3+w_region] = coarse_texture.permute(1, 0).unsqueeze(0).repeat(3, 1, 1)
    
    # 3. 细纹理区域 (左下角) - 小周期纹理
    x_fine = torch.linspace(0, 20*np.pi, w_region, device='cuda')
    y_fine = torch.linspace(0, 20*np.pi, h_region, device='cuda')
    X_fine, Y_fine = torch.meshgrid(x_fine, y_fine, indexing='ij')
    fine_texture = 0.5 + 0.2 * torch.sin(X_fine) * torch.cos(Y_fine)
    image[:, H//3:H//3+h_region, 0:w_region] = fine_texture.permute(1, 0).unsqueeze(0).repeat(3, 1, 1)
    
    # 4. 极细纹理区域 (右下角) - 非常小的周期
    x_ultra = torch.linspace(0, 50*np.pi, w_region, device='cuda')
    y_ultra = torch.linspace(0, 50*np.pi, h_region, device='cuda')
    X_ultra, Y_ultra = torch.meshgrid(x_ultra, y_ultra, indexing='ij')
    ultra_texture = 0.5 + 0.1 * torch.sin(X_ultra) * torch.cos(Y_ultra)
    image[:, H//3:H//3+h_region, W//3:W//3+w_region] = ultra_texture.permute(1, 0).unsqueeze(0).repeat(3, 1, 1)
    
    # 5. 添加各种边缘
    # 垂直强边缘
    image[:, H//4:3*H//4, W//6] = 1.0
    # 水平强边缘  
    image[:, H//6, W//4:3*W//4] = 0.1
    # 对角线边缘
    for i in range(min(H//3, W//3)):
        if i < H and i < W:
            image[:, i, i] = 0.8
    
    # 6. 渐变区域 (下半部分)
    gradient_h = H - 2*H//3
    gradient = torch.linspace(0, 1, W, device='cuda').unsqueeze(0).repeat(gradient_h, 1)
    image[:, 2*H//3:, :] = gradient.unsqueeze(0).repeat(3, 1, 1)
    
    return image

def analyze_kernel_size_effects():
    """分析不同核大小的精细效果"""
    print("分析 flatness_kernel_size 的精细作用...")
    
    # 创建详细测试图像
    test_image = create_detailed_test_image()
    print(f"测试图像尺寸: {test_image.shape}")
    
    # 测试不同的核大小
    kernel_sizes = [1, 3, 5, 7, 9, 11, 15, 21, 31]
    
    results = {}
    
    for kernel_size in kernel_sizes:
        print(f"\n=== 核大小: {kernel_size} ===")
        
        # 计算权重图
        weight_map = compute_flatness_weight(
            test_image,
            kernel_size=kernel_size,
            flat_weight=0.1,
            edge_weight=0.02
        )
        
        # 分析不同区域的权重
        H, W = test_image.shape[1], test_image.shape[2]
        
        # 平坦区域 (左上角)
        flat_region = weight_map[0, 0:H//3, 0:W//3]
        flat_mean = flat_region.mean().item()
        flat_std = flat_region.std().item()
        
        # 粗纹理区域 (右上角)
        coarse_region = weight_map[0, 0:H//3, W//3:2*W//3]
        coarse_mean = coarse_region.mean().item()
        coarse_std = coarse_region.std().item()
        
        # 细纹理区域 (左下角)
        fine_region = weight_map[0, H//3:2*H//3, 0:W//3]
        fine_mean = fine_region.mean().item()
        fine_std = fine_region.std().item()
        
        # 极细纹理区域 (右下角)
        ultra_region = weight_map[0, H//3:2*H//3, W//3:2*W//3]
        ultra_mean = ultra_region.mean().item()
        ultra_std = ultra_region.std().item()
        
        # 渐变区域 (下半部分)
        gradient_region = weight_map[0, 2*H//3:, :]
        gradient_mean = gradient_region.mean().item()
        gradient_std = gradient_region.std().item()
        
        # 整体统计
        overall_mean = weight_map.mean().item()
        overall_std = weight_map.std().item()
        overall_min = weight_map.min().item()
        overall_max = weight_map.max().item()
        
        print(f"整体统计: 均值={overall_mean:.4f}, 标准差={overall_std:.4f}, 范围=[{overall_min:.4f}, {overall_max:.4f}]")
        print(f"平坦区域: 均值={flat_mean:.4f}, 标准差={flat_std:.4f}")
        print(f"粗纹理区域: 均值={coarse_mean:.4f}, 标准差={coarse_std:.4f}")
        print(f"细纹理区域: 均值={fine_mean:.4f}, 标准差={fine_std:.4f}")
        print(f"极细纹理区域: 均值={ultra_mean:.4f}, 标准差={ultra_std:.4f}")
        print(f"渐变区域: 均值={gradient_mean:.4f}, 标准差={gradient_std:.4f}")
        
        # 计算区域间的权重对比
        flat_vs_coarse = flat_mean / coarse_mean if coarse_mean > 0 else float('inf')
        flat_vs_fine = flat_mean / fine_mean if fine_mean > 0 else float('inf')
        flat_vs_ultra = flat_mean / ultra_mean if ultra_mean > 0 else float('inf')
        
        print(f"权重对比 - 平坦/粗纹理: {flat_vs_coarse:.2f}, 平坦/细纹理: {flat_vs_fine:.2f}, 平坦/极细纹理: {flat_vs_ultra:.2f}")
        
        results[kernel_size] = {
            'overall_std': overall_std,
            'flat_mean': flat_mean,
            'coarse_mean': coarse_mean,
            'fine_mean': fine_mean,
            'ultra_mean': ultra_mean,
            'gradient_mean': gradient_mean,
            'flat_vs_fine': flat_vs_fine,
            'flat_vs_ultra': flat_vs_ultra
        }
    
    return results

def analyze_edge_sensitivity():
    """分析不同核大小对边缘的敏感性"""
    print("\n=== 边缘敏感性分析 ===")
    
    # 创建包含不同强度边缘的图像
    H, W = 256, 256
    edge_image = torch.ones(3, H, W, device='cuda') * 0.5
    
    # 添加不同强度的边缘
    # 强边缘 (对比度 0.5)
    edge_image[:, :, W//4] = 1.0
    # 中等边缘 (对比度 0.3)
    edge_image[:, :, W//2] = 0.8
    # 弱边缘 (对比度 0.1)
    edge_image[:, :, 3*W//4] = 0.6
    
    kernel_sizes = [1, 3, 5, 9, 15]
    
    for kernel_size in kernel_sizes:
        weight_map = compute_flatness_weight(edge_image, kernel_size=kernel_size)
        
        # 分析边缘附近的权重变化
        strong_edge_weights = weight_map[0, :, W//4-2:W//4+3].mean(dim=1)
        medium_edge_weights = weight_map[0, :, W//2-2:W//2+3].mean(dim=1)
        weak_edge_weights = weight_map[0, :, 3*W//4-2:3*W//4+3].mean(dim=1)
        
        # 计算边缘检测的锐度 (权重变化的标准差)
        strong_sharpness = strong_edge_weights.std().item()
        medium_sharpness = medium_edge_weights.std().item()
        weak_sharpness = weak_edge_weights.std().item()
        
        print(f"核大小 {kernel_size}: 强边缘锐度={strong_sharpness:.4f}, 中边缘锐度={medium_sharpness:.4f}, 弱边缘锐度={weak_sharpness:.4f}")

def recommend_kernel_size():
    """推荐不同应用场景的核大小"""
    print("\n=== 核大小推荐 ===")
    
    recommendations = {
        1: "极精细检测 - 保留所有细节，但可能过于敏感噪声",
        3: "精细检测 - 适合高分辨率图像，保留大部分细节",
        5: "标准检测 - 平衡细节保留和噪声抑制 (默认推荐)",
        7: "中等平滑 - 适合中等分辨率，轻微平滑噪声",
        9: "平滑检测 - 忽略小细节，关注主要结构",
        11: "粗糙检测 - 只检测明显的纹理变化",
        15: "很粗糙 - 只区分大面积的平坦和纹理区域",
        21: "极粗糙 - 几乎只区分大块区域",
        31: "超粗糙 - 可能过度平滑，丢失重要边缘信息"
    }
    
    for size, desc in recommendations.items():
        print(f"kernel_size = {size:2d}: {desc}")
    
    print("\n应用场景建议:")
    print("• 高分辨率图像 (>1024): 使用 3-5")
    print("• 标准分辨率 (512-1024): 使用 5-7") 
    print("• 低分辨率 (<512): 使用 7-9")
    print("• 噪声较多: 增大核大小")
    print("• 细节丰富: 减小核大小")

if __name__ == "__main__":
    if torch.cuda.is_available():
        results = analyze_kernel_size_effects()
        analyze_edge_sensitivity()
        recommend_kernel_size()
        
        print("\n=== 核大小效果总结 ===")
        print("核大小  整体标准差  平坦/细纹理比  平坦/极细纹理比")
        print("-" * 50)
        for kernel_size, data in results.items():
            print(f"{kernel_size:6d}  {data['overall_std']:10.4f}  {data['flat_vs_fine']:12.2f}  {data['flat_vs_ultra']:14.2f}")
        
        print("\n✅ 核大小精细作用分析完成！")
    else:
        print("❌ 需要CUDA支持来运行测试") 