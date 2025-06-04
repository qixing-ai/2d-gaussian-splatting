#!/usr/bin/env python3
"""
测试自适应法线损失功能
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from utils.loss_utils import compute_flatness_weight

def create_test_image():
    """创建测试图像：包含平坦区域和纹理区域"""
    H, W = 256, 256
    image = torch.zeros(3, H, W, device='cuda')
    
    # 左半部分：平坦区域（纯色）
    image[:, :, :W//2] = 0.5
    
    # 右半部分：纹理区域（棋盘格）
    for i in range(H//16):
        for j in range(W//32, W//16):
            if (i + j) % 2 == 0:
                image[:, i*16:(i+1)*16, j*16:(j+1)*16] = 0.8
            else:
                image[:, i*16:(i+1)*16, j*16:(j+1)*16] = 0.2
    
    return image

def visualize_adaptive_weights():
    """可视化自适应权重的效果"""
    # 创建测试图像
    test_image = create_test_image()
    
    # 计算平坦度权重
    weight_map = compute_flatness_weight(
        test_image, 
        flat_weight=2.0, 
        texture_weight=0.5
    )
    
    # 转换为numpy用于可视化
    image_np = test_image.cpu().numpy().transpose(1, 2, 0)
    weight_np = weight_map.cpu().numpy().squeeze()
    
    # 创建可视化
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 原始图像
    axes[0].imshow(image_np)
    axes[0].set_title('原始测试图像\n(左：平坦区域，右：纹理区域)')
    axes[0].axis('off')
    
    # 权重图
    im = axes[1].imshow(weight_np, cmap='hot', vmin=0.5, vmax=2.0)
    axes[1].set_title('自适应权重图\n(红色：高权重，蓝色：低权重)')
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1])
    
    # 权重分布直方图
    axes[2].hist(weight_np.flatten(), bins=50, alpha=0.7)
    axes[2].set_xlabel('权重值')
    axes[2].set_ylabel('像素数量')
    axes[2].set_title('权重分布直方图')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('adaptive_normal_weights_test.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # 打印统计信息
    print(f"权重统计信息:")
    print(f"  最小权重: {weight_np.min():.3f}")
    print(f"  最大权重: {weight_np.max():.3f}")
    print(f"  平均权重: {weight_np.mean():.3f}")
    print(f"  平坦区域平均权重: {weight_np[:, :128].mean():.3f}")
    print(f"  纹理区域平均权重: {weight_np[:, 128:].mean():.3f}")

def test_different_parameters():
    """测试不同参数设置的效果"""
    test_image = create_test_image()
    
    # 测试不同的权重设置
    configs = [
        {"flat_weight": 1.0, "texture_weight": 1.0, "name": "均匀权重"},
        {"flat_weight": 2.0, "texture_weight": 0.5, "name": "默认设置"},
        {"flat_weight": 3.0, "texture_weight": 0.3, "name": "强对比"},
        {"flat_weight": 1.5, "texture_weight": 0.8, "name": "弱对比"},
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for i, config in enumerate(configs):
        weight_map = compute_flatness_weight(
            test_image,
            flat_weight=config["flat_weight"],
            texture_weight=config["texture_weight"]
        )
        weight_np = weight_map.cpu().numpy().squeeze()
        
        im = axes[i].imshow(weight_np, cmap='hot', vmin=0.3, vmax=3.0)
        axes[i].set_title(f'{config["name"]}\n平坦:{config["flat_weight"]}, 纹理:{config["texture_weight"]}')
        axes[i].axis('off')
        plt.colorbar(im, ax=axes[i])
    
    plt.tight_layout()
    plt.savefig('adaptive_normal_parameters_test.png', dpi=150, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    print("测试自适应法线损失功能...")
    
    # 检查CUDA可用性
    if not torch.cuda.is_available():
        print("警告: CUDA不可用，使用CPU进行测试")
        device = 'cpu'
    else:
        device = 'cuda'
        print(f"使用设备: {device}")
    
    try:
        # 可视化自适应权重
        print("\n1. 可视化自适应权重效果...")
        visualize_adaptive_weights()
        
        # 测试不同参数
        print("\n2. 测试不同参数设置...")
        test_different_parameters()
        
        print("\n✅ 测试完成！生成的图像:")
        print("  - adaptive_normal_weights_test.png: 权重效果展示")
        print("  - adaptive_normal_parameters_test.png: 不同参数对比")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc() 