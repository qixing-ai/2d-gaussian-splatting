#!/usr/bin/env python3
"""
测试基于图像梯度的自适应法线一致性算法
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from utils.loss_utils import compute_adaptive_normal_weights, compute_training_losses

def create_test_image():
    """创建一个测试图像，包含平坦区域和边缘区域"""
    # 创建一个简单的测试图像
    H, W = 128, 128
    image = torch.zeros(3, H, W)
    
    # 添加一些平坦区域（低梯度）
    image[:, 20:60, 20:60] = 0.5
    
    # 添加一些边缘（高梯度）
    image[:, 70:110, 70:110] = 1.0
    image[:, 75:105, 75:105] = 0.0
    
    # 添加一些纹理区域
    x = torch.linspace(0, 4*np.pi, W)
    y = torch.linspace(0, 4*np.pi, H)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    texture = 0.5 + 0.3 * torch.sin(X) * torch.cos(Y)
    image[:, :40, 80:] = texture[:40, 80:].unsqueeze(0).repeat(3, 1, 1)
    
    return image

def test_adaptive_weights():
    """测试自适应权重计算"""
    print("测试基于图像梯度的自适应法线一致性算法...")
    
    # 创建测试图像
    test_image = create_test_image()
    print(f"测试图像形状: {test_image.shape}")
    
    # 计算自适应权重
    weights = compute_adaptive_normal_weights(
        test_image,
        flat_weight=0.1,
        edge_weight=1.0,
        threshold=0.1
    )
    
    print(f"权重图形状: {weights.shape}")
    print(f"权重范围: [{weights.min().item():.3f}, {weights.max().item():.3f}]")
    print(f"平均权重: {weights.mean().item():.3f}")
    
    # 统计不同权重区域
    flat_regions = (weights < 0.3).sum().item()
    edge_regions = (weights > 0.7).sum().item()
    total_pixels = weights.numel()
    
    print(f"平坦区域像素数: {flat_regions} ({flat_regions/total_pixels*100:.1f}%)")
    print(f"边缘区域像素数: {edge_regions} ({edge_regions/total_pixels*100:.1f}%)")
    
    return test_image, weights

def test_with_different_parameters():
    """测试不同参数设置"""
    print("\n测试不同参数设置...")
    
    test_image = create_test_image()
    
    # 测试不同的权重设置
    configs = [
        {"flat_weight": 0.1, "edge_weight": 1.0, "threshold": 0.1},
        {"flat_weight": 0.05, "edge_weight": 1.5, "threshold": 0.05},
        {"flat_weight": 0.2, "edge_weight": 0.8, "threshold": 0.2},
    ]
    
    for i, config in enumerate(configs):
        weights = compute_adaptive_normal_weights(test_image, **config)
        avg_weight = weights.mean().item()
        print(f"配置 {i+1}: flat={config['flat_weight']}, edge={config['edge_weight']}, "
              f"threshold={config['threshold']} -> 平均权重: {avg_weight:.3f}")

class MockOpt:
    """模拟训练参数"""
    def __init__(self):
        self.lambda_normal = 0.05
        self.lambda_dssim = 0.2
        self.lambda_alpha = 0.01
        self.lambda_converge = 0.0
        self.adaptive_normal_start_iter = 1000  # 使用新的参数名
        self.iterations = 30000
        # 自适应法线参数
        self.lambda_adaptive_normal = 0.03  # 第二阶段专用的自适应法线权重
        self.normal_flat_weight = 0.1
        self.normal_edge_weight = 1.0
        self.normal_gradient_threshold = 0.1

def create_mock_render_pkg():
    """创建模拟的渲染结果包"""
    H, W = 256, 256  # 增大图像尺寸以满足MS-SSIM要求
    return {
        'render': torch.rand(3, H, W, device='cuda'),
        'rend_normal': torch.rand(3, H, W, device='cuda'),
        'surf_normal': torch.rand(3, H, W, device='cuda'),
        'rend_alpha': torch.rand(1, H, W, device='cuda'),
        'surf_depth': torch.rand(1, H, W, device='cuda')
    }

def test_training_losses_logic():
    """测试训练损失计算的2阶段逻辑"""
    print("\n测试训练损失计算的2阶段逻辑（完全去除时间衰减）...")
    
    if not torch.cuda.is_available():
        print("⚠️  CUDA不可用，跳过训练损失测试")
        return
    
    # 创建更大的测试图像以满足MS-SSIM要求
    H, W = 256, 256
    test_image = torch.zeros(3, H, W)
    
    # 添加一些平坦区域（低梯度）
    test_image[:, 50:150, 50:150] = 0.5
    
    # 添加一些边缘（高梯度）
    test_image[:, 180:230, 180:230] = 1.0
    test_image[:, 190:220, 190:220] = 0.0
    
    # 添加一些纹理区域
    x = torch.linspace(0, 4*np.pi, W)
    y = torch.linspace(0, 4*np.pi, H)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    texture = 0.5 + 0.3 * torch.sin(X) * torch.cos(Y)
    test_image[:, :100, 150:] = texture[:100, 150:].unsqueeze(0).repeat(3, 1, 1)
    
    test_image = test_image.cuda()
    opt = MockOpt()
    
    # 模拟渲染结果包
    render_pkg = create_mock_render_pkg()
    
    # 测试不同迭代阶段
    test_iterations = [500, 1000, 1500, 5000, 15000]
    
    print("迭代次数 | 阶段 | lambda_normal | adaptive_weight | adaptive_stage | 说明")
    print("-" * 80)
    
    for iteration in test_iterations:
        try:
            losses = compute_training_losses(render_pkg, test_image, None, opt, iteration)
            
            if iteration <= opt.adaptive_normal_start_iter:
                stage_name = "固定权重"
                explanation = "使用固定法线一致性权重"
            else:
                stage_name = "自适应权重"
                explanation = "使用自适应法线一致性算法"
            
            print(f"{iteration:8d} | {stage_name:8s} | {losses['lambda_normal']:.6f} | "
                  f"{losses['adaptive_normal_weight']:.6f} | {losses['adaptive_stage']:13d} | {explanation}")
                  
        except Exception as e:
            print(f"迭代 {iteration} 测试失败: {e}")

if __name__ == "__main__":
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 运行测试
    test_image, weights = test_adaptive_weights()
    test_with_different_parameters()
    test_training_losses_logic()
    
    print("\n✅ 2阶段自适应法线一致性算法测试完成！")
    print("\n算法特点:")
    print("1. 完全去除了原有的时间衰减机制")
    print("2. 阶段1（≤ adaptive_normal_start_iter）：使用固定法线一致性权重 lambda_normal")
    print("3. 阶段2（> adaptive_normal_start_iter）：使用自适应法线一致性算法")
    print("4. 阶段2使用专门的权重参数 lambda_adaptive_normal 控制损失强度")
    print("5. 始终计算自适应权重用于监控")
    print("6. 自动识别图像中的平坦区域和边缘/纹理区域")
    print("7. 对平坦区域使用较低的法线一致性权重")
    print("8. 对边缘/纹理区域使用较高的法线一致性权重")
    print("\n参数说明:")
    print("- adaptive_normal_start_iter: 切换到自适应算法的迭代次数")
    print("- lambda_adaptive_normal: 第二阶段自适应法线损失的权重")
    print("- normal_flat_weight: 平坦区域的权重")
    print("- normal_edge_weight: 边缘/纹理区域的权重")
    print("- normal_gradient_threshold: 梯度阈值") 