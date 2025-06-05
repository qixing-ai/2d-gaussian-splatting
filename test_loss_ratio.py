#!/usr/bin/env python3
"""
测试自适应法线损失在总损失中的占比
"""

import torch
import numpy as np
from utils.loss_utils import compute_training_losses, compute_flatness_weight

class MockOpt:
    """模拟优化参数"""
    def __init__(self):
        self.lambda_dssim = 0.3
        self.lambda_alpha = 0.1
        self.lambda_converge = 0.5
        self.flat_normal_weight = 0.1
        self.edge_normal_weight = 0.02
        self.flatness_kernel_size = 5

def create_mock_render_pkg(H=256, W=256):
    """创建模拟的渲染包"""
    # 模拟渲染图像
    render_image = torch.rand(3, H, W, device='cuda')
    
    # 模拟渲染法线
    rend_normal = torch.randn(3, H, W, device='cuda')
    rend_normal = torch.nn.functional.normalize(rend_normal, dim=0)
    
    # 模拟表面法线（与渲染法线有一定差异）
    surf_normal = rend_normal + 0.1 * torch.randn(3, H, W, device='cuda')
    surf_normal = torch.nn.functional.normalize(surf_normal, dim=0)
    
    # 模拟alpha通道
    rend_alpha = torch.rand(1, H, W, device='cuda')
    
    # 模拟深度收敛图
    convergence_map = torch.rand(1, H, W, device='cuda') * 0.01
    
    return {
        'render': render_image,
        'rend_normal': rend_normal,
        'surf_normal': surf_normal,
        'rend_alpha': rend_alpha,
        'convergence_map': convergence_map
    }

def create_test_images():
    """创建不同类型的测试图像"""
    H, W = 256, 256
    
    # 1. 平坦图像
    flat_image = torch.ones(3, H, W, device='cuda') * 0.5
    
    # 2. 纹理丰富图像
    x = torch.linspace(0, 20*np.pi, W, device='cuda')
    y = torch.linspace(0, 20*np.pi, H, device='cuda')
    X, Y = torch.meshgrid(x, y, indexing='ij')
    texture_image = torch.stack([
        0.5 + 0.3 * torch.sin(X) * torch.cos(Y),
        0.5 + 0.3 * torch.cos(X) * torch.sin(Y),
        0.5 + 0.3 * torch.sin(X + Y)
    ]).T.permute(2, 0, 1)
    
    # 3. 混合图像（左半平坦，右半纹理）
    mixed_image = torch.zeros(3, H, W, device='cuda')
    mixed_image[:, :, :W//2] = 0.5  # 左半平坦
    mixed_image[:, :, W//2:] = texture_image[:, :, W//2:]  # 右半纹理
    
    return {
        'flat': flat_image,
        'texture': texture_image,
        'mixed': mixed_image
    }

def analyze_loss_ratios():
    """分析不同图像类型下的损失占比"""
    print("分析自适应法线损失占比...")
    
    opt = MockOpt()
    test_images = create_test_images()
    
    results = {}
    
    for image_type, gt_image in test_images.items():
        print(f"\n=== {image_type.upper()} 图像分析 ===")
        
        # 创建模拟渲染包
        render_pkg = create_mock_render_pkg()
        
        # 计算损失
        loss_dict = compute_training_losses(
            render_pkg, gt_image, None, opt, iteration=1000
        )
        
        # 分析各损失组件
        total_loss = loss_dict['total_loss'].item()
        reconstruction_loss = loss_dict['reconstruction_loss'].item()
        normal_loss = loss_dict['normal_loss'].item()
        alpha_loss = loss_dict['alpha_loss'].item()
        depth_loss = loss_dict['depth_convergence_loss'].item()
        
        # 计算占比
        normal_ratio = (normal_loss / total_loss) * 100
        reconstruction_ratio = (reconstruction_loss / total_loss) * 100
        alpha_ratio = (alpha_loss / total_loss) * 100
        depth_ratio = (depth_loss / total_loss) * 100
        
        print(f"总损失: {total_loss:.6f}")
        print(f"重建损失: {reconstruction_loss:.6f} ({reconstruction_ratio:.2f}%)")
        print(f"自适应法线损失: {normal_loss:.6f} ({normal_ratio:.2f}%)")
        print(f"Alpha损失: {alpha_loss:.6f} ({alpha_ratio:.2f}%)")
        print(f"深度收敛损失: {depth_loss:.6f} ({depth_ratio:.2f}%)")
        
        # 分析自适应权重
        adaptive_weights = compute_flatness_weight(gt_image)
        avg_weight = adaptive_weights.mean().item()
        min_weight = adaptive_weights.min().item()
        max_weight = adaptive_weights.max().item()
        
        print(f"自适应权重 - 平均: {avg_weight:.4f}, 范围: [{min_weight:.4f}, {max_weight:.4f}]")
        
        results[image_type] = {
            'normal_ratio': normal_ratio,
            'normal_loss': normal_loss,
            'avg_weight': avg_weight,
            'weight_range': (min_weight, max_weight)
        }
    
    return results

def compare_with_uniform_weights():
    """比较自适应权重与统一权重的差异"""
    print("\n=== 自适应权重 vs 统一权重对比 ===")
    
    # 使用混合图像进行对比
    gt_image = create_test_images()['mixed']
    render_pkg = create_mock_render_pkg()
    
    # 计算基础法线误差
    rend_normal = render_pkg['rend_normal']
    surf_normal = render_pkg['surf_normal']
    normal_error = (1 - (rend_normal * surf_normal).sum(dim=0))
    
    # 1. 自适应权重
    adaptive_weights = compute_flatness_weight(gt_image)
    adaptive_loss = (normal_error * adaptive_weights.squeeze(0)).mean()
    
    # 2. 统一权重（使用平均权重）
    uniform_weight = adaptive_weights.mean()
    uniform_loss = (normal_error * uniform_weight).mean()
    
    # 3. 原始2DGS权重（假设为0.05）
    original_weight = 0.05
    original_loss = (normal_error * original_weight).mean()
    
    print(f"基础法线误差平均值: {normal_error.mean().item():.6f}")
    print(f"自适应权重损失: {adaptive_loss.item():.6f}")
    print(f"统一权重损失: {uniform_loss.item():.6f}")
    print(f"原始2DGS权重损失: {original_loss.item():.6f}")
    
    print(f"\n相对变化:")
    print(f"自适应 vs 统一: {((adaptive_loss - uniform_loss) / uniform_loss * 100).item():.2f}%")
    print(f"自适应 vs 原始: {((adaptive_loss - original_loss) / original_loss * 100).item():.2f}%")

if __name__ == "__main__":
    if torch.cuda.is_available():
        results = analyze_loss_ratios()
        compare_with_uniform_weights()
        
        print("\n=== 总结 ===")
        for image_type, data in results.items():
            print(f"{image_type}: 法线损失占比 {data['normal_ratio']:.2f}%, 平均权重 {data['avg_weight']:.4f}")
        
        print("\n✅ 损失占比分析完成！")
    else:
        print("❌ 需要CUDA支持来运行测试") 