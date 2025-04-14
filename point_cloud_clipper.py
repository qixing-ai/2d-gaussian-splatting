#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
点云裁剪工具 - 使用COLMAP相机参数和图像透明度来剪裁点云

此工具使用COLMAP相机的内参和外参以及图像的背景透明度信息，
对空间中的密集点云进行裁剪，帮助确立3D模型的形状。

支持使用自定义密集点云或生成随机点云进行裁剪。
"""

import os
import sys
import numpy as np
from PIL import Image
import argparse
from tqdm import tqdm
import open3d as o3d
from scipy import ndimage
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
import time

# 导入COLMAP相关函数
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text

def load_colmap_data(basedir):
    """
    加载COLMAP数据（相机参数和点云）
    """
    cameras_file = os.path.join(basedir, "sparse/0/cameras.bin")
    points_file = os.path.join(basedir, "sparse/0/points3D.bin")
    images_file = os.path.join(basedir, "sparse/0/images.bin")

    # 检查二进制文件是否存在，不存在则尝试加载文本文件
    if not os.path.exists(cameras_file):
        cameras_file = os.path.join(basedir, "sparse/0/cameras.txt")
        images_file = os.path.join(basedir, "sparse/0/images.txt")
        points_file = os.path.join(basedir, "sparse/0/points3D.txt")

    # 加载相机内参
    if os.path.exists(cameras_file):
        if cameras_file.endswith('.bin'):
            cam_intrinsics = read_intrinsics_binary(cameras_file)
        else:
            cam_intrinsics = read_intrinsics_text(cameras_file)
    else:
        print(f"错误: 找不到相机参数文件 {cameras_file}")
        return None, None, None, None

    # 加载相机外参
    if os.path.exists(images_file):
        if images_file.endswith('.bin'):
            cam_extrinsics = read_extrinsics_binary(images_file)
        else:
            cam_extrinsics = read_extrinsics_text(images_file)
    else:
        print(f"错误: 找不到图像参数文件 {images_file}")
        return None, None, None, None

    # 加载点云 (只在需要稀疏点云作为边界框时使用)
    sparse_points_xyz = None
    sparse_points_rgb = None
    if os.path.exists(points_file):
        if points_file.endswith('.bin'):
            sparse_points_xyz, sparse_points_rgb, _ = read_points3D_binary(points_file)
        else:
            sparse_points_xyz, sparse_points_rgb, _ = read_points3D_text(points_file)
        print(f"加载了 {sparse_points_xyz.shape[0]} 个稀疏点云点")

    print(f"加载了 {len(cam_intrinsics)} 个相机内参")
    print(f"加载了 {len(cam_extrinsics)} 个相机外参")

    return cam_intrinsics, cam_extrinsics, sparse_points_xyz, sparse_points_rgb

def load_point_cloud(file_path):
    """
    从文件加载点云
    支持的格式: .ply, .pcd, .xyz 等Open3D支持的格式
    """
    print(f"从文件加载密集点云: {file_path}")
    
    try:
        pcd = o3d.io.read_point_cloud(file_path)
        points = np.asarray(pcd.points)
        
        # 如果点云有颜色属性，也加载颜色
        if pcd.has_colors():
            colors = np.asarray(pcd.colors) * 255  # Open3D颜色范围为[0,1]
        else:
            colors = None
            
        print(f"成功加载点云，共 {len(points)} 个点")
        return points, colors
    except Exception as e:
        print(f"加载点云文件时出错: {str(e)}")
        return None, None

def generate_uniform_point_cloud(bounds_min, bounds_max, num_points=1000000):
    """
    在指定边界框内生成均匀分布的点云
    
    参数:
    - bounds_min: 边界框最小值 [x_min, y_min, z_min]
    - bounds_max: 边界框最大值 [x_max, y_max, z_max]
    - num_points: 生成的点云数量
    
    返回:
    - points: 生成的点云坐标 (N, 3)
    """
    print(f"在边界框内生成均匀分布的点云，数量: {num_points}")
    print(f"边界范围: {bounds_min} 到 {bounds_max}")
    
    # 生成三个方向上均匀分布的随机点
    points = np.random.uniform(
        low=bounds_min,
        high=bounds_max,
        size=(num_points, 3)
    )
    
    return points

def alpha_threshold_mask(image_path, threshold=0.5):
    """
    根据透明通道生成二值掩码
    """
    try:
        image = Image.open(image_path).convert('RGBA')
        alpha = np.array(image)[:, :, 3]
        # 二值化透明度通道
        mask = alpha > (threshold * 255)
        
        # 可选：使用形态学操作改善掩码质量
        mask = ndimage.binary_erosion(mask, structure=np.ones((3, 3)), iterations=1)
        
        return mask
    except Exception as e:
        print(f"处理图像 {image_path} 时出错: {e}")
        return None

def project_points_batch(points, R, T, intrinsics, width, height):
    """
    批量将3D点投影到图像平面
    
    参数:
    - points: 点云坐标 (N, 3)
    - R: 旋转矩阵
    - T: 平移向量
    
    返回:
    - pixels: 投影后的像素坐标 (N, 2)
    - valid_mask: 有效投影掩码 (N,)
    """
    # 将点从世界坐标转换到相机坐标 (N, 3)
    points_camera = np.dot(points, R.T) + T
    
    # 检查点是否在相机前方
    z_positive = points_camera[:, 2] > 0
    
    pixels = np.zeros((len(points), 2), dtype=np.int32)
    valid_mask = np.zeros(len(points), dtype=bool)
    
    # 只处理z>0的点
    if not np.any(z_positive):
        return pixels, valid_mask
    
    pts_cam_valid = points_camera[z_positive]
    
    # 根据相机模型进行投影
    if intrinsics.model == "SIMPLE_PINHOLE":
        focal_length = intrinsics.params[0]
        cx = intrinsics.params[1]
        cy = intrinsics.params[2]
        
        # 计算投影
        u = focal_length * pts_cam_valid[:, 0] / pts_cam_valid[:, 2] + cx
        v = focal_length * pts_cam_valid[:, 1] / pts_cam_valid[:, 2] + cy
        
    elif intrinsics.model == "PINHOLE":
        focal_length_x = intrinsics.params[0]
        focal_length_y = intrinsics.params[1]
        cx = intrinsics.params[2]
        cy = intrinsics.params[3]
        
        # 计算投影
        u = focal_length_x * pts_cam_valid[:, 0] / pts_cam_valid[:, 2] + cx
        v = focal_length_y * pts_cam_valid[:, 1] / pts_cam_valid[:, 2] + cy
    else:
        # 对于不支持的相机模型，返回空结果
        print(f"不支持的相机模型: {intrinsics.model}")
        return pixels, valid_mask
    
    # 将u,v合并为投影坐标
    uv = np.column_stack([u, v])
    
    # 检查像素是否在图像范围内
    in_image = (
        (uv[:, 0] >= 0) & (uv[:, 0] < width) & 
        (uv[:, 1] >= 0) & (uv[:, 1] < height)
    )
    
    if not np.any(in_image):
        return pixels, valid_mask
    
    # 更新有效点的像素坐标
    valid_idx = np.where(z_positive)[0][in_image]
    pixels[valid_idx] = uv[in_image].astype(np.int32)
    valid_mask[valid_idx] = True
    
    return pixels, valid_mask

def process_camera(args):
    """
    处理单个相机视角的点云可见性
    用于多进程并行处理
    """
    cam_idx, extr, intr, points, image_path, alpha_threshold = args
    
    # 初始化可见性计数数组
    visibility = np.zeros(len(points), dtype=np.int32)
    
    # 获取相机参数
    R = qvec2rotmat(extr.qvec)
    T = np.array(extr.tvec)
    width, height = intr.width, intr.height
    
    # 获取图像掩码
    mask = alpha_threshold_mask(image_path, alpha_threshold)
    if mask is None:
        return visibility
    
    # 批量投影点云
    pixels, valid_mask = project_points_batch(points, R, T, intr, width, height)
    
    if not np.any(valid_mask):
        return visibility
    
    # 获取有效点的像素坐标
    valid_pixels = pixels[valid_mask]
    
    # 检查点是否在前景中
    valid_indices = np.where(valid_mask)[0]
    for i, (u, v) in enumerate(valid_pixels):
        if mask[v, u]:
            idx = valid_indices[i]
            visibility[idx] = 1
    
    return visibility

def calculate_visibility_mask_parallel(points, cam_extrinsics, cam_intrinsics, images_dir, min_views=3, alpha_threshold=0.5, batch_size=1000, num_workers=None):
    """
    并行计算点云的可见性掩码
    
    参数:
    - points: 点云坐标 (N, 3)
    - cam_extrinsics: 相机外参
    - cam_intrinsics: 相机内参
    - images_dir: 图像目录
    - min_views: 最少需要在多少个视角中可见
    - alpha_threshold: 透明度阈值
    - batch_size: 每批处理的点云数
    - num_workers: 并行工作进程数，默认为CPU核心数减1
    
    返回:
    - visibility_mask: 可见性掩码 (N,)
    - visibility_counts: 每个点被多少相机观察到 (N,)
    """
    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)
    
    start_time = time.time()
    print(f"使用 {num_workers} 个进程并行计算可见性...")
    
    # 准备相机参数列表
    camera_args = []
    for img_id, extr in cam_extrinsics.items():
        intr = cam_intrinsics[extr.camera_id]
        image_path = os.path.join(images_dir, extr.name)
        
        if not os.path.exists(image_path):
            print(f"警告: 找不到图像 {image_path}")
            continue
            
        camera_args.append((img_id, extr, intr, points, image_path, alpha_threshold))
    
    # 创建进程池
    with Pool(processes=num_workers) as pool:
        # 使用tqdm显示进度
        results = list(tqdm(
            pool.imap(process_camera, camera_args),
            total=len(camera_args),
            desc="计算可见性"
        ))
    
    # 合并结果
    visibility_counts = np.sum(results, axis=0)
    visibility_mask = visibility_counts >= min_views
    
    elapsed_time = time.time() - start_time
    print(f"可见性计算完成，耗时 {elapsed_time:.2f} 秒")
    
    return visibility_mask, visibility_counts

def save_point_cloud(points, colors, filename):
    """
    保存点云为PLY文件
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)  # 归一化颜色
    o3d.io.write_point_cloud(filename, pcd)
    print(f"点云已保存至 {filename}")

def colorize_points_by_visibility(points, visibility_counts, max_visibility=None):
    """
    根据可见度为点云着色
    """
    if max_visibility is None:
        max_visibility = np.max(visibility_counts)
    
    # 使用热力图颜色映射
    cmap = plt.get_cmap('viridis')
    normalized_counts = np.clip(visibility_counts / max_visibility, 0, 1)
    colors = cmap(normalized_counts)[:, :3]  # 取RGB通道，丢弃Alpha通道
    
    return colors

def estimate_bounds_from_cameras(cam_extrinsics, scale_factor=1.5):
    """
    根据相机位置估计场景边界
    
    参数:
    - cam_extrinsics: 相机外参
    - scale_factor: 边界缩放因子，用于扩大边界范围
    
    返回:
    - bounds_min: 边界框最小值 [x_min, y_min, z_min]
    - bounds_max: 边界框最大值 [x_max, y_max, z_max]
    """
    # 提取所有相机的位置
    camera_positions = []
    for _, extr in cam_extrinsics.items():
        R = qvec2rotmat(extr.qvec)
        T = np.array(extr.tvec)
        # 相机中心是 -R^T * T
        camera_center = -R.T @ T
        camera_positions.append(camera_center)
    
    camera_positions = np.array(camera_positions)
    
    # 计算边界框
    center = np.mean(camera_positions, axis=0)
    distances = np.linalg.norm(camera_positions - center, axis=1)
    max_dist = np.max(distances)
    
    # 创建一个立方体边界框，边长为相机距离的几倍
    bounds_size = max_dist * scale_factor
    bounds_min = center - bounds_size
    bounds_max = center + bounds_size
    
    return bounds_min, bounds_max

def main():
    parser = argparse.ArgumentParser(description="使用COLMAP相机参数和图像透明度裁剪点云")
    parser.add_argument('--data_dir', type=str, required=True, help='COLMAP数据目录')
    parser.add_argument('--images_dir', type=str, help='图像目录，默认为data_dir/images')
    parser.add_argument('--output_ply', type=str, default='clipped_pointcloud.ply', help='输出点云文件名')
    parser.add_argument('--min_views', type=int, default=3, help='点要在至少多少个视角中可见才保留')
    parser.add_argument('--alpha_threshold', type=float, default=0.5, help='透明度掩码阈值')
    parser.add_argument('--dense_points', type=int, default=1000000, help='生成的均匀密集点云点数')
    parser.add_argument('--input_pointcloud', type=str, help='输入的密集点云文件路径(.ply, .pcd等)')
    parser.add_argument('--vis_output', type=str, default='visibility_pointcloud.ply', help='可视化可见度的点云输出')
    parser.add_argument('--num_workers', type=int, default=None, help='并行工作进程数，默认为CPU核心数减1')
    parser.add_argument('--batch_size', type=int, default=1000, help='批处理大小')
    parser.add_argument('--scene_scale', type=float, default=1.5, help='场景边界框缩放因子')
    parser.add_argument('--bounds_min', type=str, help='手动指定边界框最小值，格式为"x,y,z"')
    parser.add_argument('--bounds_max', type=str, help='手动指定边界框最大值，格式为"x,y,z"')
    
    args = parser.parse_args()
    
    # 设置默认图像目录
    if args.images_dir is None:
        args.images_dir = os.path.join(args.data_dir, 'images')
    
    # 加载COLMAP数据 (只需要相机参数)
    cam_intrinsics, cam_extrinsics, sparse_points_xyz, _ = load_colmap_data(args.data_dir)
    if cam_intrinsics is None or cam_extrinsics is None:
        print("无法加载COLMAP相机参数，退出程序。")
        return
    
    # 确定输入点云
    input_points = None
    input_colors = None
    
    # 1. 如果提供了输入点云文件，则加载它
    if args.input_pointcloud and os.path.exists(args.input_pointcloud):
        input_points, input_colors = load_point_cloud(args.input_pointcloud)
        if input_points is None:
            print("无法加载输入点云文件，将使用其他方法生成点云。")
    
    # 2. 如果没有提供点云文件或加载失败，生成均匀分布的点云
    if input_points is None:
        # 确定点云的边界
        if args.bounds_min and args.bounds_max:
            # 使用用户指定的边界
            try:
                bounds_min = np.array([float(x) for x in args.bounds_min.split(',')])
                bounds_max = np.array([float(x) for x in args.bounds_max.split(',')])
                print(f"使用用户指定的边界: {bounds_min} 到 {bounds_max}")
            except:
                print("无法解析用户指定的边界，将使用估计边界。")
                bounds_min, bounds_max = None, None
        else:
            bounds_min, bounds_max = None, None
        
        # 如果没有指定边界或解析失败，尝试估计边界
        if bounds_min is None or bounds_max is None:
            if sparse_points_xyz is not None and len(sparse_points_xyz) > 0:
                # 使用COLMAP稀疏点云的边界
                bounds_min = np.min(sparse_points_xyz, axis=0) - args.scene_scale
                bounds_max = np.max(sparse_points_xyz, axis=0) + args.scene_scale
                print(f"使用COLMAP稀疏点云边界（扩展{args.scene_scale}倍）: {bounds_min} 到 {bounds_max}")
            else:
                # 使用相机位置估计场景边界
                bounds_min, bounds_max = estimate_bounds_from_cameras(cam_extrinsics, args.scene_scale)
                print(f"使用相机位置估计场景边界: {bounds_min} 到 {bounds_max}")
        
        # 生成均匀分布的密集点云
        print(f"生成均匀分布的密集点云，点数: {args.dense_points}")
        input_points = generate_uniform_point_cloud(bounds_min, bounds_max, args.dense_points)
        input_colors = None
    
    print(f"输入点云大小: {len(input_points)} 个点")
    
    # 计算可见性掩码
    visibility_mask, visibility_counts = calculate_visibility_mask_parallel(
        input_points, cam_extrinsics, cam_intrinsics, args.images_dir, 
        args.min_views, args.alpha_threshold, args.batch_size, args.num_workers
    )
    
    # 筛选符合可见性要求的点
    clipped_points = input_points[visibility_mask]
    kept_percentage = len(clipped_points) / len(input_points) * 100 if len(input_points) > 0 else 0
    print(f"裁剪后的点云包含 {len(clipped_points)} 个点 (原始点云中的 {kept_percentage:.2f}%)")
    
    # 保存裁剪后的点云
    if input_colors is not None:
        clipped_colors = input_colors[visibility_mask]
        save_point_cloud(clipped_points, clipped_colors, args.output_ply)
    else:
        save_point_cloud(clipped_points, None, args.output_ply)
    
    # 保存可视化可见度的点云
    if args.vis_output:
        print("生成可见度可视化点云...")
        colors = colorize_points_by_visibility(input_points, visibility_counts)
        save_point_cloud(input_points, colors * 255, args.vis_output)
        print(f"可见度可视化点云已保存至 {args.vis_output}")
    
    print("点云裁剪完成！")

if __name__ == "__main__":
    main() 