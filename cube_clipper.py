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

def generate_voxel_grid(bounds_min, bounds_max, voxel_size=0.05):
    """
    在指定边界框内生成均匀的体素网格点云
    
    参数:
    - bounds_min: 边界框最小值 [x_min, y_min, z_min]
    - bounds_max: 边界框最大值 [x_max, y_max, z_max]
    - voxel_size: 体素大小（立方体边长）
    
    返回:
    - points: 生成的体素网格中心点坐标 (N, 3)
    """
    print(f"在边界框内生成体素网格，体素大小: {voxel_size}")
    print(f"边界范围: {bounds_min} 到 {bounds_max}")
    
    # 计算每个维度的体素数量
    num_voxels_x = int((bounds_max[0] - bounds_min[0]) / voxel_size) + 1
    num_voxels_y = int((bounds_max[1] - bounds_min[1]) / voxel_size) + 1
    num_voxels_z = int((bounds_max[2] - bounds_min[2]) / voxel_size) + 1
    
    print(f"体素网格尺寸: {num_voxels_x} x {num_voxels_y} x {num_voxels_z} = {num_voxels_x * num_voxels_y * num_voxels_z} 个体素")
    
    # 生成每个维度的坐标
    x_coords = np.linspace(bounds_min[0], bounds_max[0], num_voxels_x)
    y_coords = np.linspace(bounds_min[1], bounds_max[1], num_voxels_y)
    z_coords = np.linspace(bounds_min[2], bounds_max[2], num_voxels_z)
    
    # 使用meshgrid创建3D网格
    X, Y, Z = np.meshgrid(x_coords, y_coords, z_coords, indexing='ij')
    
    # 将网格点转换为坐标点列表
    points = np.column_stack([X.flatten(), Y.flatten(), Z.flatten()])
    
    return points

def generate_surface_voxel_grid(bounds_min, bounds_max, cam_extrinsics, cam_intrinsics, images_dir, voxel_size=0.05):
    """
    在指定边界框内生成适应表面的体素网格点云
    
    参数:
    - bounds_min: 边界框最小值 [x_min, y_min, z_min]
    - bounds_max: 边界框最大值 [x_max, y_max, z_max]
    - cam_extrinsics: 相机外参
    - cam_intrinsics: 相机内参
    - images_dir: 图像目录
    - voxel_size: 体素大小（立方体边长）
    
    返回:
    - points: 生成的表面体素网格点云 (N, 3)
    """
    print(f"生成适应表面的体素网格点云，体素大小: {voxel_size}")
    print(f"边界范围: {bounds_min} 到 {bounds_max}")
    
    # 先生成完整的体素网格点云和索引
    grid_points = generate_voxel_grid(bounds_min, bounds_max, voxel_size)
    print(f"基础体素网格点数: {len(grid_points)}")
    
    # 计算体素网格的尺寸
    voxel_dims = np.ceil((bounds_max - bounds_min) / voxel_size).astype(int)
    nx, ny, nz = voxel_dims
    print(f"体素网格维度: [{nx}, {ny}, {nz}]")
    
    # 创建体素索引映射 - 从3D体素索引到点云中的索引
    # 用于快速查找体素对应的点
    voxel_map = {}
    for i, point in enumerate(grid_points):
        # 计算体素索引
        voxel_idx = tuple(np.floor((point - bounds_min) / voxel_size).astype(int))
        voxel_map[voxel_idx] = i
    
    # 收集所有相机图像中的前景/背景边缘信息
    edge_voxels = set()  # 使用集合存储边缘体素的索引
    
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
                
            # 获取相机参数
            R = qvec2rotmat(extr.qvec)
            T = np.array(extr.tvec)
            width, height = intr.width, intr.height
            
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
            
            # 投射光线，找到与体素网格相交的边缘体素
            edge_v = edge_pixels[0]
            edge_u = edge_pixels[1]
            
            # 随机采样边缘点以提高性能
            num_samples = min(len(edge_u), 5000)
            if len(edge_u) > num_samples:
                indices = np.random.choice(len(edge_u), num_samples, replace=False)
                edge_u = edge_u[indices]
                edge_v = edge_v[indices]
            
            # 高效的DDA算法进行光线投射
            for u, v in zip(edge_u, edge_v):
                # 像素坐标归一化
                x = (u - cx) / fx
                y = (v - cy) / fy
                
                # 构建方向向量
                direction = np.array([x, y, 1.0])
                direction = direction / np.linalg.norm(direction)
                
                # 将方向从相机坐标系转到世界坐标系
                world_direction = R.T @ direction
                
                # 光线的起点
                ray_origin = camera_center
                
                # 计算光线与包围盒的相交点
                # 光线公式: ray_origin + t * world_direction
                # 需要计算t值使得点在包围盒表面
                t_min = np.full(3, -np.inf)
                t_max = np.full(3, np.inf)
                
                for i in range(3):
                    if abs(world_direction[i]) > 1e-6:
                        t1 = (bounds_min[i] - ray_origin[i]) / world_direction[i]
                        t2 = (bounds_max[i] - ray_origin[i]) / world_direction[i]
                        t_min[i] = min(t1, t2)
                        t_max[i] = max(t1, t2)
                    elif ray_origin[i] < bounds_min[i] or ray_origin[i] > bounds_max[i]:
                        # 光线平行于轴且不在边界框内
                        continue
                
                tmin = max(np.max(t_min), 0)  # 确保不会向相机后面走
                tmax = np.min(t_max)
                
                if tmin > tmax:
                    continue  # 光线未击中包围盒
                
                # 开始体素穿越
                # 初始位置
                curr_pos = ray_origin + tmin * world_direction
                
                # 确保起点在体积内
                if not ((curr_pos >= bounds_min).all() and (curr_pos <= bounds_max).all()):
                    curr_pos = np.clip(curr_pos, bounds_min, bounds_max)
                
                # 计算初始体素索引
                voxel_idx = np.floor((curr_pos - bounds_min) / voxel_size).astype(int)
                
                # 计算下一个体素的步长
                step = np.sign(world_direction).astype(int)
                
                # 计算t值的增量
                delta_t = voxel_size / np.abs(world_direction)
                delta_t = np.where(np.isfinite(delta_t), delta_t, np.inf)
                
                # 计算到下一个体素边界的t值
                next_t = np.zeros(3)
                for i in range(3):
                    if step[i] > 0:
                        next_t[i] = ((voxel_idx[i] + 1) * voxel_size + bounds_min[i] - ray_origin[i]) / world_direction[i]
                    elif step[i] < 0:
                        next_t[i] = (voxel_idx[i] * voxel_size + bounds_min[i] - ray_origin[i]) / world_direction[i]
                    else:
                        next_t[i] = np.inf
                
                # 沿光线遍历体素
                max_steps = 100  # 限制步数
                for _ in range(max_steps):
                    # 检查当前体素是否有效
                    if (voxel_idx >= 0).all() and (voxel_idx < voxel_dims).all():
                        voxel_key = tuple(voxel_idx)
                        if voxel_key in voxel_map:
                            edge_voxels.add(voxel_map[voxel_key])
                            break  # 找到一个边缘体素后停止
                    
                    # 找出下一个要穿越的轴
                    axis = np.argmin(next_t)
                    
                    # 更新体素索引
                    voxel_idx[axis] += step[axis]
                    
                    # 检查是否已经离开体积
                    if voxel_idx[axis] < 0 or voxel_idx[axis] >= voxel_dims[axis]:
                        break
                    
                    # 更新下一个t值
                    next_t[axis] += delta_t[axis]
            
            print(f"从图像 {extr.name} 处理后的边缘体素数: {len(edge_voxels)}")
            
        except Exception as e:
            print(f"处理图像 {image_path} 时出错: {e}")
            continue
    
    if not edge_voxels:
        print("警告: 没有找到边缘体素，将使用完整体素网格")
        return grid_points
    
    # 提取边缘体素点
    edge_points = grid_points[list(edge_voxels)]
    print(f"找到的边缘体素点数: {len(edge_points)}")
    
    # 如果边缘体素太少，扩展邻域
    if len(edge_points) < 1000:
        print("边缘体素数量较少，扩展到邻域体素")
        expanded_voxels = set(edge_voxels)
        
        # 定义邻域偏移量 (26个邻居)
        neighbors = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    neighbors.append(np.array([dx, dy, dz]))
        
        # 扩展到邻居体素
        for idx in edge_voxels:
            point = grid_points[idx]
            # 转换为体素索引
            voxel_idx = np.floor((point - bounds_min) / voxel_size).astype(int)
            
            # 检查26个邻居
            for neighbor in neighbors:
                neighbor_idx = voxel_idx + neighbor
                neighbor_key = tuple(neighbor_idx)
                
                # 检查邻居体素是否有效且在体素映射中
                if (neighbor_idx >= 0).all() and (neighbor_idx < voxel_dims).all() and neighbor_key in voxel_map:
                    expanded_voxels.add(voxel_map[neighbor_key])
        
        # 更新边缘体素集合
        edge_voxels = expanded_voxels
        edge_points = grid_points[list(edge_voxels)]
        print(f"扩展后的边缘体素点数: {len(edge_points)}")
    
    return edge_points

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
        img_array = np.array(image)
        alpha = img_array[:, :, 3]
        
        # 检查图像是否完全透明（没有前景）
        if np.max(alpha) == 0:
            print(f"跳过完全透明的图像: {image_path}")
            return visibility
        
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
        
        # 检查图像是否完全透明
        try:
            image = Image.open(image_path).convert('RGBA')
            alpha = np.array(image)[:, :, 3]
            if np.max(alpha) == 0:
                print(f"跳过完全透明的图像: {image_path}")
                continue
        except Exception as e:
            print(f"读取图像 {image_path} 时出错: {e}")
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

def convert_ply_to_txt(ply_filename, txt_filename, cam_extrinsics=None):
    """
    将PLY点云文件转换为COLMAP格式的TXT点云文件
    
    参数:
    - ply_filename: 输入PLY文件路径
    - txt_filename: 输出TXT文件路径
    - cam_extrinsics: 相机外参，用于添加track信息
    """
    print(f"将PLY点云转换为COLMAP格式的TXT文件: {txt_filename}")
    
    # 读取PLY点云
    pcd = o3d.io.read_point_cloud(ply_filename)
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors) * 255.0  # 转换回0-255范围
    
    # 如果没有颜色信息，使用默认颜色
    if len(colors) == 0:
        colors = np.ones((len(points), 3)) * 128  # 默认灰色
    
    # 创建文件并写入头部信息
    with open(txt_filename, 'w') as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        f.write(f"# Number of points: {len(points)}, mean track length: 1.0\n")
        
        # 写入点云数据
        for i, (point, color) in enumerate(zip(points, colors)):
            # COLMAP格式：点ID, X, Y, Z, R, G, B, ERROR, TRACK[]
            point_id = i + 1  # 点ID从1开始
            x, y, z = point
            r, g, b = int(color[0]), int(color[1]), int(color[2])
            error = 1.0  # 默认误差
            
            # 构建基本信息
            line = f"{point_id} {x} {y} {z} {r} {g} {b} {error}"
            
            # 添加简单的track信息(如果没有实际相机信息，则添加虚拟track)
            if cam_extrinsics is None:
                # 添加一个虚拟track
                line += f" 1 1 {i}"
            else:
                # 这里可以根据实际相机信息生成更复杂的track
                # 暂时使用简化版本
                line += f" 1 1 {i}"
            
            f.write(line + "\n")
    
    print(f"成功将点云转换为COLMAP格式并保存至 {txt_filename}")

def main():
    parser = argparse.ArgumentParser(description="使用COLMAP相机参数和图像透明度对点云进行评分和可视化")
    parser.add_argument('--data_dir', type=str, required=True, help='COLMAP数据目录')
    parser.add_argument('--images_dir', type=str, help='图像目录，默认为data_dir/images')
    parser.add_argument('--output_ply', type=str, default='model.ply', help='输出点云文件名')
    parser.add_argument('--output_txt', type=str, default='model.txt', help='输出TXT格式点云文件名')
    parser.add_argument('--dense_points', type=int, default=1000000, help='生成的均匀密集点云点数')
    parser.add_argument('--num_workers', type=int, default=None, help='并行工作进程数，默认为CPU核心数减1')
    parser.add_argument('--batch_size', type=int, default=1000, help='批处理大小')
    parser.add_argument('--volume_x', type=float, default=4, help='体积X轴长度')
    parser.add_argument('--volume_y', type=float, default=2, help='体积Y轴长度')
    parser.add_argument('--volume_z', type=float, default=1.5, help='体积Z轴长度')
    parser.add_argument('--center_offset_x', type=float, default=-0.8, help='体积X轴定位偏移量（不移动中心点）')
    parser.add_argument('--center_offset_y', type=float, default=0.2, help='体积Y轴定位偏移量（不移动中心点）')
    parser.add_argument('--center_offset_z', type=float, default=0.3, help='体积Z轴定位偏移量（不移动中心点）')
    parser.add_argument('--no_clip', action='store_true', help='不进行裁切，只生成点云')
    parser.add_argument('--use_uniform', action='store_true', help='使用均匀分布点云而非表面点云（默认使用表面点云）')
    parser.add_argument('--use_voxel', action='store_true', help='使用体素网格点云而非表面点云或随机点云')
    parser.add_argument('--voxel_size', type=float, default=0.05, help='体素大小（当use_voxel为True时有效）')
    parser.add_argument('--surface_voxel', action='store_true', help='使用适应表面的体素网格点云')
    parser.add_argument('--no_txt', action='store_true', help='不生成TXT格式点云文件（默认会生成）')
    
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
    
    # 创建体积中心点（仅用于体积定位，不修改原始中心点）
    volume_center = np.copy(center)
    volume_center[0] += args.center_offset_x
    volume_center[1] += args.center_offset_y
    volume_center[2] += args.center_offset_z
    
    # 设置以体积中心点为基准的体积边界(x/y/z)
    half_x = args.volume_x / 2
    half_y = args.volume_y / 2
    half_z = args.volume_z / 2
    
    # 创建长方形边界
    bounds_min = np.array([volume_center[0] - half_x, volume_center[1] - half_y, volume_center[2] - half_z])
    bounds_max = np.array([volume_center[0] + half_x, volume_center[1] + half_y, volume_center[2] + half_z])
    
    print(f"使用体积边界:")
    print(f"  原始中心点: [{center[0]:.4f}, {center[1]:.4f}, {center[2]:.4f}]")
    print(f"  体积定位点: [{volume_center[0]:.4f}, {volume_center[1]:.4f}, {volume_center[2]:.4f}] (应用偏移量: [{args.center_offset_x:.4f}, {args.center_offset_y:.4f}, {args.center_offset_z:.4f}])")
    print(f"  尺寸: {args.volume_x:.2f} x {args.volume_y:.2f} x {args.volume_z:.2f}")
    print(f"  X轴: {bounds_min[0]:.4f} 到 {bounds_max[0]:.4f}, 宽度: {bounds_max[0] - bounds_min[0]:.4f}")
    print(f"  Y轴: {bounds_min[1]:.4f} 到 {bounds_max[1]:.4f}, 高度: {bounds_max[1] - bounds_min[1]:.4f}")
    print(f"  Z轴: {bounds_min[2]:.4f} 到 {bounds_max[2]:.4f}, 深度: {bounds_max[2] - bounds_min[2]:.4f}")
    
    # 生成点云(四种模式：表面点云、表面体素网格、均匀体素网格、均匀分布)
    if args.surface_voxel:
        print(f"生成适应表面的体素网格点云，体素大小: {args.voxel_size}")
        input_points = generate_surface_voxel_grid(
            bounds_min, bounds_max,
            cam_extrinsics, cam_intrinsics,
            args.images_dir, args.voxel_size
        )
    elif args.use_voxel:
        print(f"生成均匀体素网格点云，体素大小: {args.voxel_size}")
        input_points = generate_voxel_grid(bounds_min, bounds_max, args.voxel_size)
    elif args.use_uniform:
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
    
    # 将PLY转换为TXT格式（默认开启）
    if not args.no_txt:
        txt_filename = args.output_txt
        convert_ply_to_txt(args.output_ply, txt_filename, cam_extrinsics)
    
    print("点云处理完成！")

if __name__ == "__main__":
    main()


    # python point_cloud_build.py --data_dir /workspace/2dgs/reoutput --images_dir /workspace/2dgs/reoutput/images/ 

# 使用示例:
# 1. 使用原始表面点云方式：
#    python cube_clipper.py --data_dir /workspace/2dgs/reoutput --images_dir /workspace/2dgs/reoutput/images/ 
#
# 2. 使用均匀体素网格（速度更快，但不贴合表面）：
#    python cube_clipper.py --data_dir /workspace/2dgs/reoutput --images_dir /workspace/2dgs/reoutput/images/ --use_voxel --voxel_size 0.05
#
# 3. 使用表面适应的体素网格（推荐，速度较快且贴合表面）：
#    python cube_clipper.py --data_dir /workspace/2dgs/reoutput --images_dir /workspace/2dgs/reoutput/images/ --surface_voxel --voxel_size 0.03
#
# 4. 调整体素大小（更小的值生成更精细的网格）：
#    python cube_clipper.py --data_dir /workspace/2dgs/reoutput --images_dir /workspace/2dgs/reoutput/images/ --surface_voxel --voxel_size 0.02
#
# 5. 不进行裁剪，只生成体素网格：
#    python cube_clipper.py --data_dir /workspace/2dgs/reoutput --images_dir /workspace/2dgs/reoutput/images/ --surface_voxel --voxel_size 0.05 --no_clip 