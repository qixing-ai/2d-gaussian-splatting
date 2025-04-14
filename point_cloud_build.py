#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
点云裁剪工具 - 使用COLMAP相机参数和图像透明度边缘来剪裁点云

此工具使用COLMAP相机的内参和外参以及图像的背景透明度边缘信息，
对空间中的均匀生成的密集点云进行裁剪，创建中空的3D模型。
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

def generate_uniform_point_cloud(bounds_min, bounds_max, num_points=1000000, center_point_ratio=0.8, center_area_scale=0.3):
    """
    在指定边界框内生成分布的点云，中心位置密集，边缘位置稀疏
    
    参数:
    - bounds_min: 边界框最小值 [x_min, y_min, z_min]
    - bounds_max: 边界框最大值 [x_max, y_max, z_max]
    - num_points: 生成的点云数量
    - center_point_ratio: 中心区域点的比例（0-1之间）
    - center_area_scale: 中心区域大小比例（相对于总体积）
    
    返回:
    - points: 生成的点云坐标 (N, 3)
    """
    print(f"在边界框内生成中心密集、边缘稀疏的点云，数量: {num_points}")
    print(f"边界范围: {bounds_min} 到 {bounds_max}")
    print(f"中心区域点比例: {center_point_ratio:.2f}, 中心区域大小比例: {center_area_scale:.2f}")
    
    # 计算体积中心
    center = (bounds_min + bounds_max) / 2
    
    # 生成随机点
    points = []
    
    # 分配点的比例：中心区域点和边缘区域点
    center_points_count = int(num_points * center_point_ratio)
    edge_points_count = num_points - center_points_count
    
    # 中心区域范围
    center_min = center - (bounds_max - bounds_min) * center_area_scale / 2
    center_max = center + (bounds_max - bounds_min) * center_area_scale / 2
    
    # 生成中心区域的密集点
    center_points = np.random.uniform(
        low=center_min,
        high=center_max,
        size=(center_points_count, 3)
    )
    
    # 生成边缘区域的稀疏点（采用非均匀分布，距离中心越远概率越低）
    edge_points = []
    
    # 计算对角线长度的一半，用作缩放标准差
    half_diagonal = np.linalg.norm((bounds_max - bounds_min) / 2)
    sigma = half_diagonal * 0.25  # 标准差为对角线长度的25%
    
    # 使用正态分布生成以中心为均值的点
    normal_points = np.random.normal(loc=center, scale=sigma, size=(edge_points_count * 2, 3))
    
    # 过滤掉太接近中心的点和超出边界的点
    for point in normal_points:
        # 过滤掉中心区域的点
        if (point > center_min).all() and (point < center_max).all():
            continue
        
        # 过滤掉超出边界的点
        if (point < bounds_min).any() or (point > bounds_max).any():
            continue
        
        edge_points.append(point)
        if len(edge_points) >= edge_points_count:
            break
    
    # 如果正态分布没有生成足够的点，用均匀分布补充
    remaining_points = edge_points_count - len(edge_points)
    if remaining_points > 0:
        # 创建掩码，排除中心区域
        def generate_edge_point():
            while True:
                point = np.random.uniform(low=bounds_min, high=bounds_max)
                if not ((point > center_min).all() and (point < center_max).all()):
                    return point
        
        for _ in range(remaining_points):
            edge_points.append(generate_edge_point())
    
    # 合并中心点和边缘点
    all_points = np.vstack([center_points, np.array(edge_points[:edge_points_count])])
    
    print(f"生成了 {len(all_points)} 个点 (中心区域: {len(center_points)}, 边缘区域: {len(edge_points[:edge_points_count])})")
    return all_points

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
    处理单个相机视角的点云，使用图像前景背景边界来标记点
    用于多进程并行处理
    
    返回:
    - visibility: 点的可见性标记，True表示边缘点，False表示背景点，None表示内部点或未观测点
    """
    cam_idx, extr, intr, points, image_path, edge_thickness = args
    
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
        
        # 使用边缘检测找到前景背景分割线
        # 首先二值化alpha通道
        binary_alpha = (alpha > 0).astype(np.uint8) * 255
        
        # 使用Sobel边缘检测器检测边缘
        edges = ndimage.sobel(binary_alpha) > 50  # 简单的Sobel边缘检测
        # 按用户指定的厚度膨胀边缘
        edges = ndimage.binary_dilation(edges, iterations=edge_thickness)
        
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
            if 0 <= v < binary_alpha.shape[0] and 0 <= u < binary_alpha.shape[1]:
                idx = valid_indices[i]
                if edges[v, u]:  # 边缘点(前景与背景交界处)
                    visibility[idx] = True
                elif binary_alpha[v, u] == 0:  # 纯背景区域
                    visibility[idx] = False
                # 其他区域（前景内部）保持为None
        
        return visibility
    except Exception as e:
        print(f"处理图像 {image_path} 时出错: {e}")
        return visibility

def calculate_points_visibility_parallel(points, cam_extrinsics, cam_intrinsics, images_dir, edge_thickness=2, batch_size=1000, num_workers=None):
    """
    并行计算点云的可见性，使用图像边缘检测来创建中空模型
    
    参数:
    - points: 点云坐标 (N, 3)
    - cam_extrinsics: 相机外参
    - cam_intrinsics: 相机内参
    - images_dir: 图像目录
    - edge_thickness: 边缘厚度（膨胀迭代次数）
    - batch_size: 每批处理的点云数
    - num_workers: 并行工作进程数，默认为CPU核心数减1
    
    返回:
    - edge_mask: 边缘点的掩码，True表示是边缘点(保留)
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
            
        camera_args.append((img_id, extr, intr, points, image_path, edge_thickness))
    
    # 创建进程池
    with Pool(processes=num_workers) as pool:
        # 使用tqdm显示进度
        results = list(tqdm(
            pool.imap(process_camera, camera_args),
            total=len(camera_args),
            desc="计算点云可见性"
        ))
    
    # 初始化边缘点掩码，默认都是False
    edge_mask = np.zeros(len(points), dtype=bool)
    
    # 合并结果：只保留被标记为边缘点，且没有被任何相机标记为背景的点
    for point_idx in range(len(points)):
        is_edge = False
        is_background = False
        
        for camera_result in results:
            visibility = camera_result[point_idx]
            
            if visibility is False:  # 明确标记为背景
                is_background = True
                break  # 一旦被标记为背景，就不再考虑该点
            elif visibility is True:  # 明确标记为边缘
                is_edge = True
        
        # 只有被至少一个相机标记为边缘，且没有任何相机将其标记为背景，才保留
        edge_mask[point_idx] = is_edge and not is_background
    
    elapsed_time = time.time() - start_time
    print(f"点云可见性计算完成，耗时 {elapsed_time:.2f} 秒")
    print(f"边缘点数量: {np.sum(edge_mask)}/{len(points)}")
    
    return edge_mask

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
    parser = argparse.ArgumentParser(description="使用COLMAP相机参数和图像透明度边缘创建中空3D模型")
    parser.add_argument('--data_dir', type=str, required=True, help='COLMAP数据目录')
    parser.add_argument('--images_dir', type=str, help='图像目录，默认为data_dir/images')
    parser.add_argument('--output_ply', type=str, default='model.ply', help='输出点云文件名')
    parser.add_argument('--dense_points', type=int, default=1000000, help='生成的密集点云点数')
    parser.add_argument('--num_workers', type=int, default=None, help='并行工作进程数，默认为CPU核心数减1')
    parser.add_argument('--batch_size', type=int, default=1000, help='批处理大小')
    parser.add_argument('--center_offset_x', type=float, default=0.0, help='中心点X轴偏移量')
    parser.add_argument('--center_offset_y', type=float, default=0.0, help='中心点Y轴偏移量')
    parser.add_argument('--center_offset_z', type=float, default=0.0, help='中心点Z轴偏移量')
    parser.add_argument('--no_clip', action='store_true', help='不进行裁切，只生成点云')
    parser.add_argument('--edge_thickness', type=int, default=2, help='边缘厚度（膨胀迭代次数）')
    parser.add_argument('--center_point_ratio', type=float, default=0.9, help='中心区域点的比例（0-1之间）')
    parser.add_argument('--center_area_scale', type=float, default=0.1, help='中心区域大小比例（相对于总体积）')
    
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
    
    # 计算相机位置的边界框，用于确定点云生成区域
    camera_positions = np.array(camera_positions)
    min_bound = np.min(camera_positions, axis=0)
    max_bound = np.max(camera_positions, axis=0)
    
    # 扩大边界框，确保包含整个场景
    center = (min_bound + max_bound) / 2
    size = max_bound - min_bound
    scale_factor = 1.5  # 扩大边界框的系数
    
    # 创建扩大后的边界框
    bounds_min = center - size * scale_factor / 2
    bounds_max = center + size * scale_factor / 2
    
    # 应用用户指定的边界偏移(如果需要)
    if args.center_offset_x != 0 or args.center_offset_y != 0 or args.center_offset_z != 0:
        offset = np.array([args.center_offset_x, args.center_offset_y, args.center_offset_z])
        bounds_min += offset
        bounds_max += offset
    
    print(f"使用扩展的相机边界框:")
    print(f"  中心点: [{center[0]:.4f}, {center[1]:.4f}, {center[2]:.4f}]")
    print(f"  尺寸: {(bounds_max[0]-bounds_min[0]):.2f} x {(bounds_max[1]-bounds_min[1]):.2f} x {(bounds_max[2]-bounds_min[2]):.2f}")
    print(f"  X轴: {bounds_min[0]:.4f} 到 {bounds_max[0]:.4f}")
    print(f"  Y轴: {bounds_min[1]:.4f} 到 {bounds_max[1]:.4f}")
    print(f"  Z轴: {bounds_min[2]:.4f} 到 {bounds_max[2]:.4f}")
    
    # 生成均匀分布的密集点云
    print(f"生成中心密集边缘稀疏的点云，点数: {args.dense_points}")
    input_points = generate_uniform_point_cloud(
        bounds_min, 
        bounds_max, 
        args.dense_points,
        args.center_point_ratio,
        args.center_area_scale
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
        edge_mask = calculate_points_visibility_parallel(input_points, cam_extrinsics, cam_intrinsics, 
                                                        args.images_dir, args.edge_thickness, 
                                                        args.batch_size, args.num_workers)
        
        # 过滤点云
        filtered_points = input_points[edge_mask]
        # 为点云设置统一颜色（蓝色）
        colors = np.zeros((len(filtered_points), 3))
        colors[:, 2] = 1.0  # 设置蓝色通道
        
        # 保存过滤后的点云
        save_point_cloud(filtered_points, colors * 255, args.output_ply)
        print(f"过滤后的点云已保存至 {args.output_ply}")
    
    print("点云处理完成！")

if __name__ == "__main__":
    main()


    