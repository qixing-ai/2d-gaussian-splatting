#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
点云裁剪工具 - 使用COLMAP相机参数和图像透明度来剪裁点云

此工具使用COLMAP相机的内参和外参以及图像的背景透明度信息，
对空间中的均匀生成的密集点云进行裁剪，帮助确立3D模型的形状。
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
    加载COLMAP数据（相机参数）
    """

    cameras_file = os.path.join(basedir, "sparse/0/cameras.txt")
    images_file = os.path.join(basedir, "sparse/0/images.txt")

    # 加载相机内参
    if os.path.exists(cameras_file):
        if cameras_file.endswith('.bin'):
            cam_intrinsics = read_intrinsics_binary(cameras_file)
        else:
            cam_intrinsics = read_intrinsics_text(cameras_file)
    else:
        print(f"错误: 找不到相机参数文件 {cameras_file}")
        return None, None

    # 加载相机外参
    if os.path.exists(images_file):
        if images_file.endswith('.bin'):
            cam_extrinsics = read_extrinsics_binary(images_file)
        else:
            cam_extrinsics = read_extrinsics_text(images_file)
    else:
        print(f"错误: 找不到图像参数文件 {images_file}")
        return None, None

    print(f"加载了 {len(cam_intrinsics)} 个相机内参")
    print(f"加载了 {len(cam_extrinsics)} 个相机外参")

    return cam_intrinsics, cam_extrinsics

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

def generate_surface_point_cloud(bounds_min, bounds_max, cam_extrinsics, cam_intrinsics, images_dir, num_points=1000000):
    """
    在指定边界框内生成面状点云，基于图像中的前景/背景分割边缘
    
    参数:
    - bounds_min: 边界框最小值 [x_min, y_min, z_min]
    - bounds_max: 边界框最大值 [x_max, y_max, z_max]
    - cam_extrinsics: 相机外参
    - cam_intrinsics: 相机内参
    - images_dir: 图像目录
    - num_points: 目标点云数量
    
    返回:
    - points: 生成的面状点云坐标 (N, 3)
    """
    print(f"基于图像边缘生成面状点云，目标数量: {num_points}")
    print(f"边界范围: {bounds_min} 到 {bounds_max}")
    
    # 保存所有相机投射出的表面点
    all_surface_points = []
    
    # 遍历所有相机视角
    for img_id, extr in cam_extrinsics.items():
        intr = cam_intrinsics[extr.camera_id]
        image_path = os.path.join(images_dir, extr.name)
        
        if not os.path.exists(image_path):
            print(f"警告: 找不到图像 {image_path}")
            continue
        
        try:
            # 读取图像的alpha通道
            image = Image.open(image_path).convert('RGBA')
            img_array = np.array(image)
            alpha = img_array[:, :, 3]
            
            # 提取边缘（前景/背景分割线）
            # 使用Sobel边缘检测
            alpha_blur = ndimage.gaussian_filter(alpha, sigma=1.0)
            edge_x = ndimage.sobel(alpha_blur, axis=0)
            edge_y = ndimage.sobel(alpha_blur, axis=1)
            edge_mag = np.sqrt(edge_x**2 + edge_y**2)
            
            # 阈值化为二值边缘图
            threshold = np.percentile(edge_mag, 95)  # 取边缘强度前5%的点
            edge_binary = (edge_mag > threshold).astype(np.uint8)
            
            # 获取边缘点坐标
            edge_pixels = np.where(edge_binary > 0)
            
            if len(edge_pixels[0]) == 0:
                continue
                
            # 如果边缘点太多，随机采样
            max_edge_points = 2000  # 每个视角最多使用的边缘点数
            if len(edge_pixels[0]) > max_edge_points:
                idx = np.random.choice(len(edge_pixels[0]), max_edge_points, replace=False)
                edge_v = edge_pixels[0][idx]
                edge_u = edge_pixels[1][idx]
            else:
                edge_v = edge_pixels[0]
                edge_u = edge_pixels[1]
            
            # 获取相机参数
            R = qvec2rotmat(extr.qvec)
            T = np.array(extr.tvec)
            width, height = intr.width, intr.height
            
            # 从相机向边缘点投射光线
            surface_points = []
            
            # 根据相机模型获取焦距和主点
            if intr.model == "SIMPLE_PINHOLE":
                focal_length = intr.params[0]
                cx = intr.params[1]
                cy = intr.params[2]
                fx, fy = focal_length, focal_length
            elif intr.model == "PINHOLE":
                fx = intr.params[0]
                fy = intr.params[1]
                cx = intr.params[2]
                cy = intr.params[3]
            elif intr.model == "RADIAL":
                focal_length = intr.params[0]
                cx = intr.params[1]
                cy = intr.params[2]
                fx, fy = focal_length, focal_length
            else:
                print(f"不支持的相机模型: {intr.model}")
                continue
            
            # 计算相机中心位置
            camera_center = -R.T @ T
            
            # 对每个边缘点，通过反投影生成不同深度的点
            depths = np.linspace(0.1, 5.0, 20)  # 在不同深度采样，可根据场景调整
            
            for u, v in zip(edge_u, edge_v):
                # 像素坐标归一化到[-1,1]
                x = (u - cx) / fx
                y = (v - cy) / fy
                
                # 构建方向向量
                direction = np.array([x, y, 1.0])
                direction = direction / np.linalg.norm(direction)
                
                # 将方向从相机坐标系转到世界坐标系
                world_direction = R.T @ direction
                
                # 沿射线在多个深度生成点
                for depth in depths:
                    point = camera_center + world_direction * depth
                    
                    # 检查点是否在边界框内
                    if (point >= bounds_min).all() and (point <= bounds_max).all():
                        surface_points.append(point)
            
            all_surface_points.extend(surface_points)
            print(f"从图像 {extr.name} 生成了 {len(surface_points)} 个表面点")
            
        except Exception as e:
            print(f"处理图像 {image_path} 时出错: {e}")
            continue
    
    # 转换为numpy数组
    if not all_surface_points:
        print("警告: 没有生成任何表面点，将回退到均匀分布的点云")
        return generate_uniform_point_cloud(bounds_min, bounds_max, num_points)
    
    surface_points = np.array(all_surface_points)
    
    # 如果点云数量超过目标数量，随机采样
    if len(surface_points) > num_points:
        idx = np.random.choice(len(surface_points), num_points, replace=False)
        surface_points = surface_points[idx]
    
    # 如果点云数量不足，通过添加周围小的随机扰动来增加点数
    if len(surface_points) < num_points:
        print(f"生成的表面点数量 ({len(surface_points)}) 不足目标数量 ({num_points})，添加随机扰动增加点数")
        points_needed = num_points - len(surface_points)
        
        # 计算点云边界尺寸的5%作为扰动范围
        bounds_size = bounds_max - bounds_min
        noise_scale = bounds_size * 0.05
        
        # 随机选择原始点添加噪声
        indices = np.random.choice(len(surface_points), points_needed)
        base_points = surface_points[indices]
        
        # 添加随机扰动
        noise = np.random.normal(0, 1, (points_needed, 3)) * noise_scale
        additional_points = base_points + noise
        
        # 确保仍在边界内
        for i in range(3):
            additional_points[:, i] = np.clip(additional_points[:, i], bounds_min[i], bounds_max[i])
        
        # 合并点云
        surface_points = np.vstack([surface_points, additional_points])
    
    print(f"最终生成了 {len(surface_points)} 个表面点")
    return surface_points

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
    elif intrinsics.model == "RADIAL":
        focal_length = intrinsics.params[0]
        cx = intrinsics.params[1]
        cy = intrinsics.params[2]
        k1 = intrinsics.params[3]
        k2 = intrinsics.params[4]
        
        # 归一化坐标
        x_norm = pts_cam_valid[:, 0] / pts_cam_valid[:, 2]
        y_norm = pts_cam_valid[:, 1] / pts_cam_valid[:, 2]
        
        # 计算径向畸变
        r2 = x_norm**2 + y_norm**2
        radial_factor = 1.0 + k1 * r2 + k2 * r2**2
        
        # 应用畸变
        x_dist = x_norm * radial_factor
        y_dist = y_norm * radial_factor
        
        # 投影到像素平面
        u = focal_length * x_dist + cx
        v = focal_length * y_dist + cy
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
    处理单个相机视角的点云，直接标记前景点和背景点
    用于多进程并行处理
    
    返回:
    - visibility: 点的可见性标记，True表示前景点，False表示背景点
    """
    cam_idx, extr, intr, points, image_path, _ = args
    
    # 初始化可见性数组，默认为None（未被观测到）
    visibility = np.full(len(points), None, dtype=object)
    
    # 获取相机参数
    R = qvec2rotmat(extr.qvec)
    T = np.array(extr.tvec)
    width, height = intr.width, intr.height
    
    try:
        # 读取图像的alpha通道
        image = Image.open(image_path).convert('RGBA')
        alpha = np.array(image)[:, :, 3]
        
        # 批量投影点云
        pixels, valid_mask = project_points_batch(points, R, T, intr, width, height)
        
        if not np.any(valid_mask):
            return visibility
        
        # 获取有效点的像素坐标
        valid_pixels = pixels[valid_mask]
        
        # 更新有效点的可见性标记
        valid_indices = np.where(valid_mask)[0]
        for i, (u, v) in enumerate(valid_pixels):
            # 添加边界检查，确保u和v不超过图像大小
            if 0 <= v < alpha.shape[0] and 0 <= u < alpha.shape[1]:
                idx = valid_indices[i]
                if alpha[v, u] > 0:  # 非透明区域(前景)
                    visibility[idx] = True
                else:  # 透明区域(背景)
                    visibility[idx] = False
        
        return visibility
    except Exception as e:
        print(f"处理图像 {image_path} 时出错: {e}")
        return visibility

def calculate_points_visibility_parallel(points, cam_extrinsics, cam_intrinsics, images_dir, batch_size=1000, num_workers=None):
    """
    并行计算点云的可见性
    
    参数:
    - points: 点云坐标 (N, 3)
    - cam_extrinsics: 相机外参
    - cam_intrinsics: 相机内参
    - images_dir: 图像目录
    - batch_size: 每批处理的点云数
    - num_workers: 并行工作进程数，默认为CPU核心数减1
    
    返回:
    - foreground_mask: 前景点的掩码，True表示是前景点
    """
    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)
    
    start_time = time.time()
    print(f"使用 {num_workers} 个进程并行计算点云可见性...")
    
    # 准备相机参数列表
    camera_args = []
    for img_id, extr in cam_extrinsics.items():
        intr = cam_intrinsics[extr.camera_id]
        image_path = os.path.join(images_dir, extr.name)
        
        if not os.path.exists(image_path):
            print(f"警告: 找不到图像 {image_path}")
            continue
            
        camera_args.append((img_id, extr, intr, points, image_path, None))
    
    # 创建进程池
    with Pool(processes=num_workers) as pool:
        # 使用tqdm显示进度
        results = list(tqdm(
            pool.imap(process_camera, camera_args),
            total=len(camera_args),
            desc="计算点云可见性"
        ))
    
    # 初始化前景点掩码，默认都是False
    foreground_mask = np.zeros(len(points), dtype=bool)
    
    # 合并结果：如果任意一个相机将点标记为前景，且没有任何相机将其标记为背景，则认为是前景点
    for point_idx in range(len(points)):
        is_foreground = False
        is_background = False
        
        for camera_result in results:
            visibility = camera_result[point_idx]
            
            if visibility is False:  # 明确标记为背景
                is_background = True
                break  # 一旦被标记为背景，就不再考虑该点
            elif visibility is True:  # 明确标记为前景
                is_foreground = True
        
        # 只有被至少一个相机标记为前景，且没有任何相机将其标记为背景，才保留
        foreground_mask[point_idx] = is_foreground and not is_background
    
    elapsed_time = time.time() - start_time
    print(f"点云可见性计算完成，耗时 {elapsed_time:.2f} 秒")
    print(f"前景点数量: {np.sum(foreground_mask)}/{len(points)}")
    
    return foreground_mask

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

def main():
    parser = argparse.ArgumentParser(description="使用COLMAP相机参数和图像透明度对点云进行评分和可视化")
    parser.add_argument('--data_dir', type=str, required=True, help='COLMAP数据目录')
    parser.add_argument('--images_dir', type=str, help='图像目录，默认为data_dir/images')
    parser.add_argument('--output_ply', type=str, default='model.ply', help='输出点云文件名')
    parser.add_argument('--dense_points', type=int, default=1000000, help='生成的均匀密集点云点数')
    parser.add_argument('--num_workers', type=int, default=None, help='并行工作进程数，默认为CPU核心数减1')
    parser.add_argument('--batch_size', type=int, default=1000, help='批处理大小')
    parser.add_argument('--volume_x', type=float, default=4, help='体积X轴长度')
    parser.add_argument('--volume_y', type=float, default=2, help='体积Y轴长度')
    parser.add_argument('--volume_z', type=float, default=1.5, help='体积Z轴长度')
    parser.add_argument('--center_offset_x', type=float, default=-0.8, help='中心点X轴偏移量')
    parser.add_argument('--center_offset_y', type=float, default=0.0, help='中心点Y轴偏移量')
    parser.add_argument('--center_offset_z', type=float, default=0.3, help='中心点Z轴偏移量')
    parser.add_argument('--no_clip', action='store_true', help='不进行裁切，只生成点云')
    parser.add_argument('--use_uniform', action='store_true', help='使用均匀分布点云而非表面点云（默认使用表面点云）')
    
    args = parser.parse_args()
    
    # 设置默认图像目录
    if args.images_dir is None:
        args.images_dir = os.path.join(args.data_dir, 'images')
    
    # 加载COLMAP数据 (只需要相机参数)
    cam_intrinsics, cam_extrinsics = load_colmap_data(args.data_dir)
    if cam_intrinsics is None or cam_extrinsics is None:
        print("无法加载COLMAP相机参数，退出程序。")
        return
    
    # 获取相机中心位置(用于定位人像)
    camera_positions = []
    for _, extr in cam_extrinsics.items():
        R = qvec2rotmat(extr.qvec)
        T = np.array(extr.tvec)
        camera_center = -R.T @ T
        camera_positions.append(camera_center)
    
    center = np.mean(np.array(camera_positions), axis=0)
    
    # 应用中心点偏移
    center[0] += args.center_offset_x
    center[1] += args.center_offset_y
    center[2] += args.center_offset_z
    
    # 设置以中心点为基准的体积边界(x/y/z)
    half_x = args.volume_x / 2
    half_y = args.volume_y / 2
    half_z = args.volume_z / 2
    
    # 创建长方形边界
    bounds_min = np.array([center[0] - half_x, center[1] - half_y, center[2] - half_z])
    bounds_max = np.array([center[0] + half_x, center[1] + half_y, center[2] + half_z])
    
    print(f"使用体积边界:")
    print(f"  中心点: [{center[0]:.4f}, {center[1]:.4f}, {center[2]:.4f}] (应用偏移量: [{args.center_offset_x:.4f}, {args.center_offset_y:.4f}, {args.center_offset_z:.4f}])")
    print(f"  尺寸: {args.volume_x:.2f} x {args.volume_y:.2f} x {args.volume_z:.2f}")
    print(f"  X轴: {bounds_min[0]:.4f} 到 {bounds_max[0]:.4f}, 宽度: {bounds_max[0] - bounds_min[0]:.4f}")
    print(f"  Y轴: {bounds_min[1]:.4f} 到 {bounds_max[1]:.4f}, 高度: {bounds_max[1] - bounds_min[1]:.4f}")
    print(f"  Z轴: {bounds_min[2]:.4f} 到 {bounds_max[2]:.4f}, 深度: {bounds_max[2] - bounds_min[2]:.4f}")
    
    # 生成点云(默认表面点云，可选均匀分布)
    if args.use_uniform:
        print(f"生成均匀分布的密集点云，点数: {args.dense_points}")
        input_points = generate_uniform_point_cloud(bounds_min, bounds_max, args.dense_points)
    else:
        print(f"生成基于图像边缘的表面点云，目标点数: {args.dense_points}")
        input_points = generate_surface_point_cloud(
            bounds_min, bounds_max, 
            cam_extrinsics, cam_intrinsics, 
            args.images_dir, args.dense_points
        )
    
    if args.no_clip:
        # 不进行裁切，直接保存全部点云
        print("跳过裁切步骤，保存所有生成的点云")
        # 为所有点设置统一颜色（蓝色）
        colors = np.zeros((len(input_points), 3))
        colors[:, 2] = 1.0  # 设置蓝色通道
        save_point_cloud(input_points, colors * 255, args.output_ply)
        print(f"原始点云已保存至 {args.output_ply}")
    else:
        # 计算点云可见性
        foreground_mask = calculate_points_visibility_parallel(
            input_points, cam_extrinsics, cam_intrinsics, 
            args.images_dir, args.batch_size, args.num_workers
        )
        
        # 过滤点云
        filtered_points = input_points[foreground_mask]
        # 为点云设置统一颜色（蓝色）
        colors = np.zeros((len(filtered_points), 3))
        colors[:, 2] = 1.0  # 设置蓝色通道
        
        # 保存过滤后的点云
        save_point_cloud(filtered_points, colors * 255, args.output_ply)
        print(f"过滤后的点云已保存至 {args.output_ply}")
    
    print("点云处理完成！")

if __name__ == "__main__":
    main()


    