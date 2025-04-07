import os
import torch
import random
import numpy as np
from diff_surfel_rasterization import GaussianRasterizationSettings, GaussianRasterizer
import matplotlib.pyplot as plt

# 设置随机种子以确保结果可复现
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

def test_rasterization_params():
    """测试CUDA光栅化器参数对渲染质量的影响"""
    
    # 创建一些随机的高斯点
    num_points = 1000
    means3D = torch.randn(num_points, 3).cuda() * 2.0  # 随机3D点
    
    # 随机颜色
    colors = torch.rand(num_points, 3).cuda()
    
    # 随机尺度
    scales = torch.rand(num_points, 2).cuda() * 0.05 + 0.01
    
    # 随机旋转
    rotations = torch.cat([torch.randn(num_points, 3), 
                          torch.ones(num_points, 1)], dim=1).cuda()
    rotations = rotations / rotations.norm(dim=1, keepdim=True)
    
    # 不透明度
    opacity = torch.rand(num_points).cuda() * 0.8 + 0.2
    
    # 设置相机参数
    image_height = 800
    image_width = 800
    tanfovx = 0.5
    tanfovy = 0.5
    bg_color = torch.ones(3).cuda()  # 白色背景
    
    # 设置视图和投影矩阵
    viewmatrix = torch.tensor([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 5.0],  # 相机在z=5处
        [0.0, 0.0, 0.0, 1.0]
    ]).cuda()
    
    projmatrix = torch.tensor([
        [2.0, 0.0, 0.0, 0.0],
        [0.0, 2.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0],
        [0.0, 0.0, -1.0, 0.0]
    ]).cuda()
    
    # 相机位置
    campos = torch.tensor([0.0, 0.0, 5.0]).cuda()
    
    # 光栅化设置
    raster_settings = GaussianRasterizationSettings(
        image_height=image_height,
        image_width=image_width,
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=1.0,
        viewmatrix=viewmatrix,
        projmatrix=projmatrix,
        sh_degree=0,
        campos=campos,
        prefiltered=False,
        debug=True  # 启用调试输出
    )
    
    # 创建光栅化器
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    
    # 执行渲染
    rendered_image, _, _ = rasterizer(
        means3D=means3D,
        means2D=torch.zeros_like(means3D),
        shs=None,
        colors_precomp=colors,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=None
    )
    
    # 将图像转换为numpy数组并显示
    rendered_image_np = rendered_image.detach().cpu().numpy().transpose(1, 2, 0)
    
    # 保存结果
    plt.figure(figsize=(10, 10))
    plt.imshow(rendered_image_np)
    plt.title("使用优化CUDA参数的渲染结果")
    plt.savefig("optimized_cuda_rendering.png")
    print(f"已保存渲染结果到 optimized_cuda_rendering.png")
    
    return rendered_image_np

if __name__ == "__main__":
    print("测试CUDA光栅化器参数优化效果...")
    result = test_rasterization_params()
    print("测试完成!") 