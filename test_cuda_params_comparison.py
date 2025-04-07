import os
import torch
import random
import numpy as np
from diff_surfel_rasterization import GaussianRasterizationSettings, GaussianRasterizer
import matplotlib.pyplot as plt
from PIL import Image
import shutil
import subprocess

# 设置随机种子以确保结果可复现
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

def backup_and_modify_config(backup=True, restore=False):
    """备份和修改config.h文件"""
    config_path = "/workspace/2dgs/2d-gaussian-splatting/submodules/diff-surfel-rasterization/cuda_rasterizer/config.h"
    backup_path = "/workspace/2dgs/2d-gaussian-splatting/config_backup.h"
    
    if backup:
        shutil.copy(config_path, backup_path)
        print(f"已备份配置文件到 {backup_path}")
    
    if restore:
        shutil.copy(backup_path, config_path)
        print(f"已从 {backup_path} 还原配置文件")
        return
    
    # 读取原始内容
    with open(config_path, 'r') as f:
        content = f.read()
    
    # 替换为原始参数
    content = content.replace("FILTER_SIZE 0.75f", "FILTER_SIZE 0.5f")
    content = content.replace("FILTER_INV_SQUARE 1.77777778f", "FILTER_INV_SQUARE 4.0f")
    content = content.replace("SUPERSAMPLE_FACTOR 2", "SUPERSAMPLE_FACTOR 1")
    content = content.replace("NEAR_PLANE 0.1f", "NEAR_PLANE 0.2f")
    content = content.replace("FAR_PLANE 150.0f", "FAR_PLANE 100.0f")
    
    # 写入修改后的内容
    with open(config_path, 'w') as f:
        f.write(content)
    
    print("已将配置文件恢复为原始参数")
    
def render_test_image(tag="default"):
    """渲染测试图像"""
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
        debug=False
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
    plt.title(f"CUDA光栅化 - {tag}")
    plt.savefig(f"cuda_render_{tag}.png")
    print(f"已保存渲染结果到 cuda_render_{tag}.png")
    
    return rendered_image_np
    
def compile_cuda_module():
    """重新编译CUDA模块"""
    cwd = os.getcwd()
    try:
        os.chdir("/workspace/2dgs/2d-gaussian-splatting/submodules/diff-surfel-rasterization")
        subprocess.run("rm -rf build/*", shell=True)
        subprocess.run("python setup.py install", shell=True)
    finally:
        os.chdir(cwd)
    
def compare_renderings():
    """比较原始参数和优化参数的渲染效果"""
    try:
        # 备份当前配置
        backup_and_modify_config(backup=True, restore=False)
        
        # 使用原始参数编译和渲染
        compile_cuda_module()
        original_render = render_test_image("original")
        
        # 恢复优化后的参数
        backup_and_modify_config(backup=False, restore=True)
        
        # 使用优化参数编译和渲染
        compile_cuda_module()
        optimized_render = render_test_image("optimized")
        
        # 创建对比图像
        fig, axes = plt.subplots(1, 2, figsize=(20, 10))
        
        axes[0].imshow(original_render)
        axes[0].set_title("原始参数")
        axes[0].axis('off')
        
        axes[1].imshow(optimized_render)
        axes[1].set_title("优化参数")
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.savefig("cuda_params_comparison.png")
        print("已保存对比图像到 cuda_params_comparison.png")
        
        # 计算差异图像
        diff = np.abs(optimized_render - original_render)
        plt.figure(figsize=(10, 10))
        plt.imshow(diff * 5.0)  # 放大差异以便观察
        plt.colorbar()
        plt.title("参数优化前后的差异图 (x5)")
        plt.savefig("cuda_params_diff.png")
        print("已保存差异图像到 cuda_params_diff.png")
        
    finally:
        # 确保恢复优化参数
        backup_and_modify_config(backup=False, restore=True)
        compile_cuda_module()

if __name__ == "__main__":
    print("开始测试CUDA光栅化器参数优化效果...")
    compare_renderings()
    print("测试完成!") 