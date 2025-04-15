#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
3D重建工具 - 结合Visual Hull和SDF，使用COLMAP相机参数和图像透明度创建三角面片模型

此工具综合使用Visual Hull和SDF (有符号距离场)算法，
通过COLMAP相机参数和图像透明度信息重建3D模型表面。
"""

import os
import sys
import numpy as np
from PIL import Image
import argparse
from tqdm import tqdm
import trimesh
import cv2
from skimage import measure
from scipy.interpolate import NearestNDInterpolator

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

def extract_camera_parameters(intrinsics):
    """
    提取相机内参参数，统一处理不同相机模型
    
    参数:
    - intrinsics: 相机内参对象
    
    返回:
    - fx: x方向焦距
    - fy: y方向焦距
    - cx: 主点x坐标
    - cy: 主点y坐标
    - k1, k2: 径向畸变系数（如果可用）
    """
    k1, k2 = 0.0, 0.0  # 默认无畸变
    
    if intrinsics.model == "SIMPLE_PINHOLE":
        focal_length = intrinsics.params[0]
        cx = intrinsics.params[1]
        cy = intrinsics.params[2]
        fx, fy = focal_length, focal_length
    elif intrinsics.model == "PINHOLE":
        fx = intrinsics.params[0]
        fy = intrinsics.params[1]
        cx = intrinsics.params[2]
        cy = intrinsics.params[3]
    elif intrinsics.model == "RADIAL":
        focal_length = intrinsics.params[0]
        cx = intrinsics.params[1]
        cy = intrinsics.params[2]
        k1 = intrinsics.params[3]
        k2 = intrinsics.params[4]
        fx, fy = focal_length, focal_length
    else:
        print(f"不支持的相机模型: {intrinsics.model}")
        return None, None, None, None, None, None
    
    return fx, fy, cx, cy, k1, k2

def load_alpha_image(image_path):
    """
    加载图像并提取alpha通道
    
    参数:
    - image_path: 图像路径
    
    返回:
    - alpha: alpha通道
    - image: 完整图像（RGBA格式）
    - success: 是否成功加载
    """
    try:
        if not os.path.exists(image_path):
            return None, None, False
            
        image = Image.open(image_path).convert('RGBA')
        img_array = np.array(image)
        alpha = img_array[:, :, 3]
        
        if np.max(alpha) == 0:
            print(f"跳过完全透明的图像: {image_path}")
            return None, None, False
            
        return alpha, image, True
    except Exception as e:
        print(f"处理图像 {image_path} 时出错: {e}")
        return None, None, False

def extract_contour_from_image(image_path):
    """
    从图像中提取有序的轮廓线
    
    参数:
    - image_path: 图像路径
    
    返回:
    - contour_points: 有序的轮廓点坐标数组列表(每个数组对应一个闭合轮廓)
    """
    alpha, _, success = load_alpha_image(image_path)
    if not success:
        return None
    
    try:
        # 二值化alpha通道
        _, binary = cv2.threshold(alpha, 127, 255, cv2.THRESH_BINARY)
        
        # 找到轮廓 - OpenCV的findContours函数返回有序的轮廓点
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        if not contours:
            print(f"未找到有效轮廓: {image_path}")
            return None
        
        # 过滤出足够大的轮廓
        significant_contours = []
        for contour in contours:
            if len(contour) >= 3:  # 至少需要3个点才能形成有效轮廓
                # 转换为(x,y)格式
                contour_xy = contour.reshape(-1, 2)
                significant_contours.append(contour_xy)
        
        if not significant_contours:
            print(f"未找到足够大的轮廓: {image_path}")
            return None
        
        return significant_contours
    
    except Exception as e:
        print(f"处理图像 {image_path} 时出错: {e}")
        return None

def save_mesh_to_ply(vertices, triangles, output_file):
    """
    保存网格为PLY文件
    
    参数:
    - vertices: 顶点坐标 (N, 3)
    - triangles: 三角形索引 (M, 3)
    - output_file: 输出文件路径
    """
    if vertices is None or triangles is None:
        print("无法保存网格：顶点或三角形为空")
        return False
    
    try:
        # 创建网格
        mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)
        
        # 保存为PLY
        mesh.export(output_file)
        print(f"网格已保存至 {output_file}")
        return True
    except Exception as e:
        print(f"保存网格时出错: {e}")
        return False

def compute_scene_bounds(camera_positions, scene_center=None, margin=0.3):
    """
    计算场景边界，基于相机位置和场景中心
    """
    camera_positions = np.array(camera_positions)
    if scene_center is None:
        scene_center = np.mean(camera_positions, axis=0)
    
    # 计算相机到场景中心的最大距离
    dists = np.linalg.norm(camera_positions - scene_center.reshape(1, 3), axis=1)
    max_dist = np.max(dists)
    
    # 设置场景边界，增加margin以确保覆盖整个场景
    bound_size = max_dist * (1 + margin)
    bounds = [
        [scene_center[0] - bound_size, scene_center[0] + bound_size], 
        [scene_center[1] - bound_size, scene_center[1] + bound_size], 
        [scene_center[2] - bound_size, scene_center[2] + bound_size]
    ]
    
    return bounds

def create_voxel_grid(bounds, resolution):
    """
    创建均匀的体素网格
    """
    # 创建xyz坐标网格
    x = np.linspace(bounds[0][0], bounds[0][1], resolution)
    y = np.linspace(bounds[1][0], bounds[1][1], resolution)
    z = np.linspace(bounds[2][0], bounds[2][1], resolution)
    
    # 创建网格点坐标
    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
    points = np.stack([xx.reshape(-1), yy.reshape(-1), zz.reshape(-1)], axis=-1)
    
    return points.reshape(resolution, resolution, resolution, 3), (x, y, z)

def compute_visual_hull(voxel_grid, voxel_coords, cam_extrinsics, cam_intrinsics, images_dir):
    """
    计算Visual Hull - 检查每个体素是否在所有相机的前景轮廓中
    """
    print("计算Visual Hull...")
    resolution = voxel_grid.shape[0]
    voxel_points = voxel_grid.reshape(-1, 3)
    visual_hull = np.ones(len(voxel_points), dtype=bool)
    
    for img_id, extr in tqdm(cam_extrinsics.items(), desc="Visual Hull处理"):
        intr = cam_intrinsics[extr.camera_id]
        image_path = os.path.join(images_dir, extr.name)
        
        alpha, _, success = load_alpha_image(image_path)
        if not success:
            continue
        
        try:
            # 二值化alpha通道
            _, alpha_binary = cv2.threshold(alpha, 127, 255, cv2.THRESH_BINARY)
            
            # 获取相机参数
            R = qvec2rotmat(extr.qvec)
            T = np.array(extr.tvec)
            
            # 提取相机内参
            fx, fy, cx, cy, _, _ = extract_camera_parameters(intr)
            if fx is None:
                continue
            
            # 转换到相机坐标系
            points_cam = (R @ voxel_points.T).T + T
            
            # 筛选在相机前方的点
            z_positive = points_cam[:, 2] > 0
            if not np.any(z_positive):
                continue
                
            # 投影到图像平面
            x_proj = fx * points_cam[z_positive, 0] / points_cam[z_positive, 2] + cx
            y_proj = fy * points_cam[z_positive, 1] / points_cam[z_positive, 2] + cy
            
            # 转为像素坐标
            x_pix = np.round(x_proj).astype(int)
            y_pix = np.round(y_proj).astype(int)
            
            # 检查是否在图像范围内
            valid_indices = (
                (x_pix >= 0) & (x_pix < alpha.shape[1]) &
                (y_pix >= 0) & (y_pix < alpha.shape[0])
            )
            
            if not np.any(valid_indices):
                continue
                
            # 获取有效的像素坐标和对应的体素索引
            valid_x = x_pix[valid_indices]
            valid_y = y_pix[valid_indices]
            valid_voxel_indices = np.where(z_positive)[0][valid_indices]
            
            # 检查像素是否在前景中
            in_foreground = alpha_binary[valid_y, valid_x] > 0
            
            # 更新Visual Hull
            current_hull_update = ~in_foreground
            visual_hull[valid_voxel_indices[current_hull_update]] = False
            
        except Exception as e:
            print(f"处理图像 {image_path} 时出错: {e}")
            continue
    
    # 重塑为体素网格形状
    visual_hull_grid = visual_hull.reshape(resolution, resolution, resolution)
    print(f"Visual Hull完成: {np.sum(visual_hull)}/{len(visual_hull)} 个体素在内部")
    
    return visual_hull_grid

def generate_base_mesh(voxel_grid, visual_hull, resolution):
    """
    从Visual Hull生成基础网格（用于SDF计算）
    """
    print("从Visual Hull生成基础网格...")
    
    try:
        # 使用marching cubes从体素网格提取网格
        voxel_size = 1.0 / resolution
        vertices, faces, _, _ = measure.marching_cubes(visual_hull, 0.5)
        
        # 调整顶点坐标到合适的尺度
        vertices = vertices * voxel_size - 0.5
        
        # 创建网格对象
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        
        # 简化网格以加速后续计算
        target_faces = min(100000, len(faces))
        if len(faces) > target_faces:
            mesh = mesh.simplify_quadratic_decimation(target_faces)
        
        return mesh.vertices, mesh.faces
    
    except Exception as e:
        print(f"生成基础网格时出错: {e}")
        return None, None

def compute_mesh_sdf(voxel_grid, base_vertices, base_triangles, visual_hull=None, num_samples=100000):
    """
    计算网格的SDF (有符号距离场)
    
    如果提供了Visual Hull，则在其外部区域强制SDF为正值
    """
    print("计算SDF场...")
    
    if base_vertices is None or base_triangles is None:
        print("未能生成有效的基础网格，无法计算SDF")
        return None
    
    # 创建网格对象
    mesh = trimesh.Trimesh(vertices=base_vertices, faces=base_triangles)
    
    resolution = voxel_grid.shape[0]
    voxel_points = voxel_grid.reshape(-1, 3)
    
    # 使用采样点计算SDF
    if len(voxel_points) > num_samples:
        # 随机采样点以加速计算
        indices = np.random.choice(len(voxel_points), num_samples, replace=False)
        sampled_points = voxel_points[indices]
        
        # 计算采样点的有符号距离
        distances, _ = trimesh.proximity.signed_distance(mesh, sampled_points)
        
        # 使用最近邻插值填充整个体素网格
        interpolator = NearestNDInterpolator(sampled_points, distances)
        distances_full = interpolator(voxel_points)
    else:
        # 如果体素点不多，直接计算所有点
        distances_full, _ = trimesh.proximity.signed_distance(mesh, voxel_points)
    
    # 重塑为3D网格
    sdf = distances_full.reshape(resolution, resolution, resolution)
    
    # 如果提供了Visual Hull，则约束SDF
    if visual_hull is not None:
        # 在Visual Hull外部强制SDF为正值
        sdf_constrained = sdf.copy()
        hull_mask = ~visual_hull
        sdf_constrained[hull_mask] = np.abs(sdf_constrained[hull_mask])
        sdf = sdf_constrained
    
    return sdf

def extract_mesh_from_sdf(sdf, voxel_coords, smooth=True):
    """
    从SDF提取三角网格
    
    参数:
    - sdf: 3D SDF数组
    - voxel_coords: 体素坐标 (x_coords, y_coords, z_coords)
    - smooth: 是否对网格进行平滑处理
    
    返回:
    - vertices: 顶点坐标
    - triangles: 三角形面片
    """
    print("从SDF提取三角网格...")
    
    try:
        # 使用skimage.measure提取网格
        vertices, faces, normals, _ = measure.marching_cubes(sdf, 0)
        
        # 缩放顶点到原始坐标系
        x_coords, y_coords, z_coords = voxel_coords
        vertices = np.array([
            x_coords[0] + (x_coords[-1] - x_coords[0]) * vertices[:, 0] / (len(x_coords) - 1),
            y_coords[0] + (y_coords[-1] - y_coords[0]) * vertices[:, 1] / (len(y_coords) - 1),
            z_coords[0] + (z_coords[-1] - z_coords[0]) * vertices[:, 2] / (len(z_coords) - 1)
        ]).T
        
        # 创建trimesh对象以进行后处理
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        
        # 网格平滑
        if smooth:
            mesh = mesh.smoothed()
        
        return mesh.vertices, mesh.faces
    
    except Exception as e:
        print(f"从SDF提取网格时出错: {e}")
        return None, None

def main():
    parser = argparse.ArgumentParser(description="使用COLMAP相机参数和图像透明度结合Visual Hull和SDF重建3D模型")
    parser.add_argument('--data_dir', type=str, required=True, help='COLMAP数据目录')
    parser.add_argument('--images_dir', type=str, help='图像目录，默认为data_dir/images')
    parser.add_argument('--output_mesh', type=str, default='model.ply', help='输出模型文件名')
    parser.add_argument('--resolution', type=int, default=128, help='体素网格分辨率')
    parser.add_argument('--use_vh', action='store_true', help='使用Visual Hull算法')
    parser.add_argument('--smooth', action='store_true', help='对最终模型进行平滑')
    parser.add_argument('--num_samples', type=int, default=100000, help='SDF计算中的采样点数量')
    
    args = parser.parse_args()
    
    # 设置默认图像目录
    if args.images_dir is None:
        args.images_dir = os.path.join(args.data_dir, 'images')
    
    # 加载COLMAP数据
    cam_intrinsics, cam_extrinsics = load_colmap_data(args.data_dir)
    if cam_intrinsics is None or cam_extrinsics is None:
        print("无法加载COLMAP相机参数，退出程序。")
        return
    
    # 计算所有相机的中心位置
    camera_positions = []
    for _, extr in cam_extrinsics.items():
        R = qvec2rotmat(extr.qvec)
        T = np.array(extr.tvec)
        camera_center = -R.T @ T
        camera_positions.append(camera_center)
    
    # 直接使用相机位置的平均值作为场景中心
    scene_center = np.mean(np.array(camera_positions), axis=0)
    
    print(f"场景信息:")
    print(f"  场景中心点: [{scene_center[0]:.4f}, {scene_center[1]:.4f}, {scene_center[2]:.4f}] (使用相机位置平均值)")
    print(f"  分辨率: {args.resolution}")
    
    # ============ Visual Hull和SDF部分 ============
    # 计算场景边界
    bounds = compute_scene_bounds(camera_positions, scene_center)
    
    # 创建体素网格
    voxel_grid, voxel_coords = create_voxel_grid(bounds, args.resolution)
    
    # 计算Visual Hull
    visual_hull = compute_visual_hull(voxel_grid, voxel_coords, cam_extrinsics, cam_intrinsics, args.images_dir)
    
    if visual_hull is None:
        print("无法计算Visual Hull，退出程序。")
        return
    
    # 从Visual Hull生成基础网格
    base_vertices, base_triangles = generate_base_mesh(voxel_grid, visual_hull, args.resolution)
    
    if base_vertices is None:
        print("无法生成基础网格，退出程序。")
        return
    
    # 计算SDF场
    sdf = compute_mesh_sdf(voxel_grid, base_vertices, base_triangles, visual_hull, args.num_samples)
    
    if sdf is None:
        print("无法计算SDF场，退出程序。")
        return
    
    # 从SDF提取最终网格
    final_vertices, final_triangles = extract_mesh_from_sdf(sdf, voxel_coords, args.smooth)
    
    if final_vertices is None:
        print("无法从SDF提取网格，退出程序。")
        return
    
    print(f"SDF方法生成了 {len(final_vertices)} 个顶点和 {len(final_triangles)} 个三角形")
    
    # 保存SDF生成的模型
    success = save_mesh_to_ply(final_vertices, final_triangles, args.output_mesh)
    
    if success:
        print(f"SDF生成的三角面片模型已保存至 {args.output_mesh}")

if __name__ == "__main__":
    main() 


    # python mesh_projection_build.py --data_dir /workspace/2dgs/reoutput --images_dir /workspace/2dgs/reoutput/images/ 