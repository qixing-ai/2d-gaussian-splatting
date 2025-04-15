#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
表面重建工具 - 使用点云和对应的法线图进行泊松表面重建
"""

import os
import sys
import numpy as np
from PIL import Image
import argparse
from tqdm import tqdm
import open3d as o3d
from multiprocessing import Pool, cpu_count
import time

# 假设 colmap_loader 在同一目录下或 Python 路径中
# 如果不在，需要调整路径
try:
    from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
        read_extrinsics_binary, read_intrinsics_binary
except ImportError:
    print("错误: 无法导入 COLMAP 加载器。请确保 scene/colmap_loader.py 在 Python 路径中。")
    sys.exit(1)

def load_colmap_data(basedir):
    """
    加载COLMAP数据（相机参数）- 从 point_cloud_build.py 借鉴
    """
    cameras_file_text = os.path.join(basedir, "sparse/0/cameras.txt")
    images_file_text = os.path.join(basedir, "sparse/0/images.txt")
    cameras_file_bin = os.path.join(basedir, "sparse/0/cameras.bin")
    images_file_bin = os.path.join(basedir, "sparse/0/images.bin")

    cam_intrinsics = None
    cam_extrinsics = None

    # 优先加载二进制文件
    if os.path.exists(cameras_file_bin) and os.path.exists(images_file_bin):
        print("加载二进制 COLMAP 文件...")
        try:
            cam_intrinsics = read_intrinsics_binary(cameras_file_bin)
            cam_extrinsics = read_extrinsics_binary(images_file_bin)
        except Exception as e:
            print(f"加载二进制文件失败: {e}, 尝试加载文本文件...")
            cam_intrinsics = None
            cam_extrinsics = None

    # 如果二进制文件加载失败或不存在，尝试加载文本文件
    if cam_intrinsics is None or cam_extrinsics is None:
        if os.path.exists(cameras_file_text) and os.path.exists(images_file_text):
            print("加载文本 COLMAP 文件...")
            try:
                cam_intrinsics = read_intrinsics_text(cameras_file_text)
                cam_extrinsics = read_extrinsics_text(images_file_text)
            except Exception as e:
                print(f"加载文本文件失败: {e}")
                return None, None
        else:
            print(f"错误: 找不到 COLMAP 相机或图像文件（二进制或文本）")
            print(f"尝试查找路径: {cameras_file_bin}, {images_file_bin}, {cameras_file_text}, {images_file_text}")
            return None, None

    if cam_intrinsics is None or cam_extrinsics is None:
        print("错误: 无法加载任何 COLMAP 相机/图像数据。")
        return None, None

    print(f"加载了 {len(cam_intrinsics)} 个相机内参")
    print(f"加载了 {len(cam_extrinsics)} 个相机外参")

    return cam_intrinsics, cam_extrinsics

def project_points_to_pixel(points, R, T, intrinsics, width, height):
    """
    将3D点投影到图像平面，返回像素坐标和有效性掩码
    (简化版 project_points_batch, 只需基本投影)
    """
    points_camera = np.dot(points, R.T) + T

    # 检查点是否在相机前方
    z_positive = points_camera[:, 2] > 1e-6 # 添加一个小阈值防止除零

    pixels = np.full((len(points), 2), -1, dtype=np.float32) # 初始化为无效值
    valid_mask = np.zeros(len(points), dtype=bool)

    if not np.any(z_positive):
        return pixels, valid_mask

    pts_cam_valid = points_camera[z_positive]
    indices_valid = np.where(z_positive)[0]

    # 根据相机模型进行投影
    if intrinsics.model == "SIMPLE_PINHOLE":
        fx = intrinsics.params[0]
        fy = intrinsics.params[0]
        cx = intrinsics.params[1]
        cy = intrinsics.params[2]
    elif intrinsics.model == "PINHOLE":
        fx = intrinsics.params[0]
        fy = intrinsics.params[1]
        cx = intrinsics.params[2]
        cy = intrinsics.params[3]
    # 简单起见，暂不处理畸变模型 (RADIAL, etc.)
    # 如果您的数据有畸变，需要在这里添加畸变校正的反操作或直接使用畸变投影
    elif intrinsics.model == "SIMPLE_RADIAL":
        fx = intrinsics.params[0]
        fy = intrinsics.params[0]
        cx = intrinsics.params[1]
        cy = intrinsics.params[2]
        # k1 = intrinsics.params[3] # 忽略畸变
    elif intrinsics.model == "RADIAL":
         fx = intrinsics.params[0]
         fy = intrinsics.params[0]
         cx = intrinsics.params[1]
         cy = intrinsics.params[2]
         # k1 = intrinsics.params[3] # 忽略畸变
         # k2 = intrinsics.params[4]
    else:
        print(f"警告: 不支持的或未处理的相机模型: {intrinsics.model}. 投影可能不准确.")
        # 尝试使用针孔模型参数（如果存在）
        if len(intrinsics.params) >= 4:
            fx = intrinsics.params[0]
            fy = intrinsics.params[1]
            cx = intrinsics.params[2]
            cy = intrinsics.params[3]
        elif len(intrinsics.params) >= 3:
             fx = intrinsics.params[0]
             fy = intrinsics.params[0]
             cx = intrinsics.params[1]
             cy = intrinsics.params[2]
        else:
             print(f"错误：无法从模型 {intrinsics.model} 获取足够的参数进行投影。")
             return pixels, valid_mask


    # 计算投影 (u, v)
    u = fx * pts_cam_valid[:, 0] / pts_cam_valid[:, 2] + cx
    v = fy * pts_cam_valid[:, 1] / pts_cam_valid[:, 2] + cy

    # 检查像素是否在图像边界内
    in_image = (u >= 0) & (u < width) & (v >= 0) & (v < height)

    if not np.any(in_image):
        return pixels, valid_mask

    # 更新有效点的像素坐标和掩码
    valid_indices_in_image = indices_valid[in_image]
    pixels[valid_indices_in_image] = np.column_stack([u[in_image], v[in_image]])
    valid_mask[valid_indices_in_image] = True

    return pixels, valid_mask


def decode_normal_map(normal_map_img):
    """
    将法线图PNG图像(RGB, 0-255)解码为法线向量(Nx, Ny, Nz, -1 到 1)
    假设使用 (normal + 1) / 2 编码
    """
    rgb = np.array(normal_map_img).astype(np.float32) / 255.0
    normals = (rgb * 2.0) - 1.0
    # 使法线方向一致，通常假设 Z 指向外
    # 如果法线图编码不一致，可能需要调整这里
    # 例如，如果Nz存储在B通道，并且是指向内的，需要 normals[:, :, 2] *= -1
    
    # 归一化确保是单位向量
    norms = np.linalg.norm(normals, axis=2, keepdims=True)
    # 避免除以零
    valid_norms = norms > 1e-6
    normals[valid_norms[:,:,0]] /= norms[valid_norms]

    return normals


def estimate_normals_from_views(points, cam_extrinsics, cam_intrinsics, normals_dir):
    """
    为点云中的每个点估计法线向量
    """
    num_points = len(points)
    summed_normals = np.zeros((num_points, 3), dtype=np.float64)
    normal_counts = np.zeros(num_points, dtype=np.int32)

    print("从不同视角估计点云法线...")
    for img_id, extr in tqdm(cam_extrinsics.items(), desc="处理相机视角"):
        intr = cam_intrinsics[extr.camera_id]
        # normal_map_path = os.path.join(normals_dir, extr.name) # 假设法线图文件名与图像名相同
        
        # 根据用户描述，构建法线图文件名 (例如 N001.png -> N001_normal.png)
        base_name, ext = os.path.splitext(extr.name)
        normal_filename = f"{base_name}_normal{ext}"
        normal_map_path = os.path.join(normals_dir, normal_filename)

        if not os.path.exists(normal_map_path):
            # 打印更详细的查找失败信息，方便调试
            # print(f"警告: 尝试查找法线图失败: {normal_map_path}")
            continue

        try:
            normal_map_img = Image.open(normal_map_path).convert('RGB')
            width, height = normal_map_img.size

            if intr.width != width or intr.height != height:
                 print(f"警告: 内参尺寸 ({intr.width}x{intr.height}) 与法线图尺寸 ({width}x{height}) 不匹配: {extr.name}. 跳过此视角.")
                 continue

            # 解码法线图
            normals_camera = decode_normal_map(normal_map_img) # (H, W, 3)

            # 获取相机参数
            R = qvec2rotmat(extr.qvec) # 世界到相机的旋转
            T = np.array(extr.tvec)    # 世界到相机的平移

            # 投影点到当前视图
            pixels, valid_mask = project_points_to_pixel(points, R, T, intr, width, height)

            if not np.any(valid_mask):
                continue

            # 获取有效投影的索引和像素坐标
            valid_indices = np.where(valid_mask)[0]
            valid_pixels = pixels[valid_mask].astype(np.int32) # 使用整数坐标查询

            # 读取对应像素的法线 (相机坐标系)
            # 注意：像素坐标 (u, v) 通常对应 NumPy 数组索引 (row, col) = (v, u)
            u_coords = valid_pixels[:, 0]
            v_coords = valid_pixels[:, 1]
            
            # 边界检查 (虽然 project_points_to_pixel 内部已检查，但再次确认更安全)
            valid_coords_mask = (u_coords >= 0) & (u_coords < width) & (v_coords >= 0) & (v_coords < height)
            
            u_coords = u_coords[valid_coords_mask]
            v_coords = v_coords[valid_coords_mask]
            valid_indices = valid_indices[valid_coords_mask] # 更新有效索引

            if len(valid_indices) == 0:
                continue

            normals_cam_view = normals_camera[v_coords, u_coords] # (N_valid, 3)

            # 将法线从相机坐标系转换到世界坐标系
            # n_world = R^T @ n_camera
            normals_world = np.dot(normals_cam_view, R) # 注意：R是world->cam，R.T是cam->world

            # 累加法线和计数
            # 使用 np.add.at 进行原子更新，避免多线程冲突（虽然这里是单线程循环）
            np.add.at(summed_normals, valid_indices, normals_world)
            np.add.at(normal_counts, valid_indices, 1)

        except Exception as e:
            print(f"处理视角 {extr.name} 时出错: {e}")
            # import traceback
            # traceback.print_exc() # 打印详细错误信息
            continue

    # 计算平均法线并归一化
    final_normals = np.zeros_like(summed_normals)
    valid_normal_mask = normal_counts > 0
    summed_valid = summed_normals[valid_normal_mask]
    counts_valid = normal_counts[valid_normal_mask][:, np.newaxis] # 保证可以广播

    averaged_normals = summed_valid / counts_valid
    
    # 归一化平均法线
    norms = np.linalg.norm(averaged_normals, axis=1, keepdims=True)
    valid_norms_mask = norms[:, 0] > 1e-6
    
    normalized_normals = np.zeros_like(averaged_normals)
    normalized_normals[valid_norms_mask] = averaged_normals[valid_norms_mask] / norms[valid_norms_mask]

    final_normals[valid_normal_mask] = normalized_normals
    
    num_no_normal = num_points - np.sum(valid_normal_mask)
    if num_no_normal > 0:
        print(f"警告: {num_no_normal}/{num_points} 个点没有从任何视图获取到有效法线。")

    return final_normals, valid_normal_mask


def main():
    parser = argparse.ArgumentParser(description="使用点云和对应的法线图进行泊松表面重建")
    parser.add_argument('--input_ply', type=str, required=True, help='输入的点云文件 (.ply)')
    parser.add_argument('--data_dir', type=str, required=True, help='COLMAP数据目录 (包含 sparse/0/)')
    parser.add_argument('--normals_dir', type=str, required=True, help='包含法线图PNG文件的目录')
    parser.add_argument('--output_mesh', type=str, default='reconstructed_mesh.ply', help='输出的重建网格文件名 (.ply)')
    parser.add_argument('--poisson_depth', type=int, default=9, help='泊松重建的八叉树深度')
    parser.add_argument('--remove_unseen', action='store_true', help='移除没有有效法线的点再进行重建')

    args = parser.parse_args()

    # 1. 加载输入点云
    print(f"加载输入点云: {args.input_ply}")
    if not os.path.exists(args.input_ply):
        print(f"错误: 输入点云文件未找到: {args.input_ply}")
        return
    pcd_input = o3d.io.read_point_cloud(args.input_ply)
    points = np.asarray(pcd_input.points)
    if len(points) == 0:
        print("错误: 输入点云为空。")
        return
    print(f"点云包含 {len(points)} 个点")

    # 2. 加载 COLMAP 相机数据
    cam_intrinsics, cam_extrinsics = load_colmap_data(args.data_dir)
    if cam_intrinsics is None or cam_extrinsics is None:
        print("无法加载COLMAP数据，退出。")
        return

    # 3. 为点云计算法线
    start_time = time.time()
    estimated_normals, valid_normal_mask = estimate_normals_from_views(
        points, cam_extrinsics, cam_intrinsics, args.normals_dir
    )
    print(f"法线估计完成，耗时 {time.time() - start_time:.2f} 秒")

    # 4. 创建带有法线的 Open3D 点云对象
    pcd_oriented = o3d.geometry.PointCloud()

    if args.remove_unseen:
        print("移除没有有效法线的点...")
        pcd_oriented.points = o3d.utility.Vector3dVector(points[valid_normal_mask])
        pcd_oriented.normals = o3d.utility.Vector3dVector(estimated_normals[valid_normal_mask])
        if pcd_input.has_colors():
            colors = np.asarray(pcd_input.colors)
            pcd_oriented.colors = o3d.utility.Vector3dVector(colors[valid_normal_mask])
        print(f"保留 {len(pcd_oriented.points)} 个点进行重建。")
    else:
        pcd_oriented.points = o3d.utility.Vector3dVector(points)
        pcd_oriented.normals = o3d.utility.Vector3dVector(estimated_normals)
         # 如果原始点云有颜色，也传递给重建
        if pcd_input.has_colors():
            pcd_oriented.colors = pcd_input.colors # 直接使用原始颜色
        # 为没有有效法线的点设置一个默认法线（例如 [0,0,0] 或 [0,0,1]）？
        # Poisson 重建通常可以处理法线为零的点，但可能会影响结果
        # 这里保持 estimated_normals 中无效点的法线为 [0,0,0]

    if not pcd_oriented.has_normals():
        print("错误: 未能为任何点计算法线。无法进行泊松重建。")
        return
    if len(pcd_oriented.points) == 0:
        print("错误: 没有点可用于重建（可能所有点都被移除了）。")
        return

    # 检查法线是否有效
    num_zero_normals = np.sum(np.linalg.norm(np.asarray(pcd_oriented.normals), axis=1) < 1e-6)
    if num_zero_normals > 0:
         print(f"警告: 重建前点云中存在 {num_zero_normals} 个零法线。")


    # 5. 执行泊松表面重建
    print(f"执行泊松表面重建 (深度={args.poisson_depth})...")
    start_time = time.time()
    # with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd_oriented, depth=args.poisson_depth, linear_fit=False # linear_fit=False 通常更好
    )
    print(f"泊松重建完成，耗时 {time.time() - start_time:.2f} 秒")

    # 可选：移除低密度顶点/面片来清理网格
    # densities_array = np.asarray(densities)
    # density_threshold = np.percentile(densities_array, 5) # 移除密度最低的5%
    # vertices_to_remove = densities_array < density_threshold
    # mesh.remove_vertices_by_mask(vertices_to_remove)
    # print(f"移除了密度低于 {density_threshold:.2f} 的顶点")


    # 6. 保存重建的网格
    print(f"保存重建的网格到: {args.output_mesh}")
    o3d.io.write_triangle_mesh(args.output_mesh, mesh, write_vertex_normals=True, write_vertex_colors=pcd_oriented.has_colors())

    print("表面重建完成！")

if __name__ == "__main__":
    main() 


