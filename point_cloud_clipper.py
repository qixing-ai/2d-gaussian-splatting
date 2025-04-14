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

def alpha_threshold_mask(image_path, threshold=0.5):
    """
    根据透明通道生成二值掩码
    
    参数:
    - image_path: 图像路径
    - threshold: 透明度阈值
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
    处理单个相机视角的点云得分
    用于多进程并行处理
    
    返回:
    - scores: 根据可见性的得分数组，前景+1，背景-1
    """
    cam_idx, extr, intr, points, image_path, _ = args
    
    # 初始化得分数组
    scores = np.zeros(len(points), dtype=np.int32)
    
    # 获取相机参数
    R = qvec2rotmat(extr.qvec)
    T = np.array(extr.tvec)
    width, height = intr.width, intr.height
    
    try:
        # 直接读取图像的alpha通道，不做阈值处理
        image = Image.open(image_path).convert('RGBA')
        alpha = np.array(image)[:, :, 3]
        
        # 批量投影点云
        pixels, valid_mask = project_points_batch(points, R, T, intr, width, height)
        
        if not np.any(valid_mask):
            return scores
        
        # 获取有效点的像素坐标
        valid_pixels = pixels[valid_mask]
        
        # 更新有效点的得分
        valid_indices = np.where(valid_mask)[0]
        for i, (u, v) in enumerate(valid_pixels):
            # 添加边界检查，确保u和v不超过图像大小
            if 0 <= v < alpha.shape[0] and 0 <= u < alpha.shape[1]:
                idx = valid_indices[i]
                if alpha[v, u] > 0:  # 非透明区域(前景)
                    scores[idx] = 1
                else:  # 透明区域(背景)
                    scores[idx] = -1
        
        return scores
    except Exception as e:
        print(f"处理图像 {image_path} 时出错: {e}")
        return scores

def calculate_points_scores_parallel(points, cam_extrinsics, cam_intrinsics, images_dir, batch_size=1000, num_workers=None):
    """
    并行计算点云的累积得分
    
    参数:
    - points: 点云坐标 (N, 3)
    - cam_extrinsics: 相机外参
    - cam_intrinsics: 相机内参
    - images_dir: 图像目录
    - batch_size: 每批处理的点云数
    - num_workers: 并行工作进程数，默认为CPU核心数减1
    
    返回:
    - point_scores: 每个点的累积得分 (N,)
    """
    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)
    
    start_time = time.time()
    print(f"使用 {num_workers} 个进程并行计算点云得分...")
    
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
            desc="计算点云得分"
        ))
    
    # 合并结果：累加所有相机的得分
    point_scores = np.sum(results, axis=0)
    
    elapsed_time = time.time() - start_time
    print(f"点云得分计算完成，耗时 {elapsed_time:.2f} 秒")
    print(f"得分范围: 最小 {np.min(point_scores)}, 最大 {np.max(point_scores)}")
    
    return point_scores

def colorize_points_by_score(points, scores):
    """
    根据得分为点云着色，使用单一颜色的深浅
    
    参数:
    - points: 点云坐标 (N, 3)
    - scores: 点云得分 (N,)
    
    返回:
    - colors: 点云颜色 (N, 3)，RGB格式，范围0-255
    """
    # 找出得分范围
    min_score = np.min(scores)
    max_score = np.max(scores)
    print(f"点云得分范围: {min_score} 到 {max_score}")
    
    # 归一化得分到[0, 1]区间
    score_range = max_score - min_score
    if score_range == 0:
        normalized_scores = np.ones_like(scores, dtype=float) * 0.5
    else:
        normalized_scores = (scores - min_score) / score_range
    
    # 使用单一颜色（这里选择蓝色）的深浅来表示得分
    # 得分高的点颜色深，得分低的点颜色浅
    colors = np.zeros((len(scores), 3))
    colors[:, 2] = normalized_scores  # 设置蓝色通道，得分越高，蓝色越深
    
    return colors

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
    parser = argparse.ArgumentParser(description="使用COLMAP相机参数和图像透明度对点云进行评分和可视化")
    parser.add_argument('--data_dir', type=str, required=True, help='COLMAP数据目录')
    parser.add_argument('--images_dir', type=str, help='图像目录，默认为data_dir/images')
    parser.add_argument('--output_ply', type=str, default='scored_pointcloud.ply', help='输出点云文件名')
    parser.add_argument('--dense_points', type=int, default=1000000, help='生成的均匀密集点云点数')
    parser.add_argument('--num_workers', type=int, default=None, help='并行工作进程数，默认为CPU核心数减1')
    parser.add_argument('--batch_size', type=int, default=1000, help='批处理大小')
    parser.add_argument('--scene_scale', type=float, default=1.5, help='场景边界框缩放因子')
    parser.add_argument('--bounds_min', type=str, help='手动指定边界框最小值，格式为"x,y,z"')
    parser.add_argument('--bounds_max', type=str, help='手动指定边界框最大值，格式为"x,y,z"')
    parser.add_argument('--keep_percentage', type=float, default=0.1, help='保留得分最高的点云百分比，默认50%')
    
    args = parser.parse_args()
    
    # 设置默认图像目录
    if args.images_dir is None:
        args.images_dir = os.path.join(args.data_dir, 'images')
    
    # 加载COLMAP数据 (只需要相机参数)
    cam_intrinsics, cam_extrinsics = load_colmap_data(args.data_dir)
    if cam_intrinsics is None or cam_extrinsics is None:
        print("无法加载COLMAP相机参数，退出程序。")
        return
    
    # 确定边界
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
    
    # 如果没有指定边界或解析失败，使用相机位置估计边界
    if bounds_min is None or bounds_max is None:
        # 使用相机位置估计场景边界
        bounds_min, bounds_max = estimate_bounds_from_cameras(cam_extrinsics, args.scene_scale)
        print(f"使用相机位置估计场景边界: {bounds_min} 到 {bounds_max}")
    
    # 生成均匀分布的密集点云
    print(f"生成均匀分布的密集点云，点数: {args.dense_points}")
    input_points = generate_uniform_point_cloud(bounds_min, bounds_max, args.dense_points)
    
    print(f"输入点云大小: {len(input_points)} 个点")
    
    # 计算点云得分(不再计算可见性掩码)
    point_scores = calculate_points_scores_parallel(
        input_points, cam_extrinsics, cam_intrinsics, args.images_dir, 
        args.batch_size, args.num_workers
    )
    
    # 筛选得分前N%的点
    keep_percentage = args.keep_percentage
    if keep_percentage < 100:
        # 计算分数阈值（保留前N%的点）
        score_threshold = np.percentile(point_scores, 100 - keep_percentage)
        # 选择得分高于阈值的点
        high_score_mask = point_scores >= score_threshold
        # 筛选点和对应的分数
        filtered_points = input_points[high_score_mask]
        filtered_scores = point_scores[high_score_mask]
        
        print(f"保留得分最高的 {keep_percentage}% 的点，阈值为 {score_threshold}")
        print(f"筛选后点云包含 {len(filtered_points)} 个点，原始点云为 {len(input_points)} 个点")
    else:
        filtered_points = input_points
        filtered_scores = point_scores
    
    # 根据得分为点云着色
    colors = colorize_points_by_score(filtered_points, filtered_scores)
    
    # 保存带颜色的点云
    save_point_cloud(filtered_points, colors * 255, args.output_ply)
    print(f"带得分颜色的点云已保存至 {args.output_ply}")
    
    print("点云处理完成！")

if __name__ == "__main__":
    main() 