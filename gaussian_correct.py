#!/usr/bin/env python3
"""
高斯参数校正脚本（CUDA加速版本）

功能：
1. 加载训练好的2D高斯模型
2. 对每个视角检测可见高斯点
3. 校正高斯点的法线方向（GPU加速）
4. 保存校正后的高斯参数

使用方法：
python gaussian_correct.py -s /path/to/scene -m /path/to/model --angle_threshold 90.0 --output_dir corrected_model
"""

import numpy as np
import torch
import pickle
import os
from scene import Scene
from argparse import ArgumentParser
from gaussian_renderer import GaussianModel
from arguments import ModelParams
from utils.general_utils import build_rotation
from plyfile import PlyData, PlyElement
from utils.system_utils import mkdir_p

def quat_to_rot_matrix_cuda(q):
    """将四元数转换为旋转矩阵（CUDA版本）"""
    if q.dim() == 1:
        q = q.unsqueeze(0)
    rot_matrix = build_rotation(q)
    return rot_matrix

def get_gaussian_normal_world_cuda(rotations, scales):
    """获取高斯点在世界坐标系下的法线方向（批量CUDA版本）"""
    # rotations: [N, 4], scales: [N, 2]
    device = rotations.device
    N = rotations.shape[0]
    
    # 构建旋转矩阵
    R = build_rotation(rotations)  # [N, 3, 3]
    
    # 法线方向是旋转矩阵的第三列（局部Z轴）
    normals = R[:, :, 2]  # [N, 3]
    return normals

def correct_normal_direction_cuda(normals, view_directions, angle_threshold_deg=90.0):
    """校正法线方向（批量CUDA版本）"""
    device = normals.device
    angle_threshold_rad = torch.tensor(np.radians(angle_threshold_deg), device=device)
    
    # 计算点积
    dot_products = torch.sum(normals * view_directions, dim=1)  # [N]
    
    # 计算角度
    angles = torch.acos(torch.clamp(torch.abs(dot_products), 0, 1))
    
    # 需要翻转的条件：角度大于阈值或点积为负
    flip_mask = (angles > angle_threshold_rad) | (dot_products < 0)
    
    # 翻转法线
    corrected_normals = normals.clone()
    corrected_normals[flip_mask] = -corrected_normals[flip_mask]
    
    return corrected_normals, flip_mask

def update_gaussian_rotation_for_normal_cuda(rotations, scales, target_normals):
    """更新高斯点的旋转参数以匹配目标法线方向（批量CUDA版本）"""
    device = rotations.device
    N = rotations.shape[0]
    
    # 获取当前法线
    current_normals = get_gaussian_normal_world_cuda(rotations, scales)
    
    # 检查哪些需要更新 - 使用逐元素比较
    diff = torch.abs(current_normals - target_normals)
    need_update = torch.any(diff > 1e-6, dim=1)  # [N] 布尔张量
    
    if not torch.any(need_update):
        return rotations
    
    # 只处理需要更新的点
    update_indices = torch.where(need_update)[0]
    if len(update_indices) == 0:
        return rotations
    
    current_normals_update = current_normals[update_indices]
    target_normals_update = target_normals[update_indices]
    rotations_update = rotations[update_indices]
    
    # 计算旋转校正
    new_rotations = rotations.clone()
    
    for i, idx in enumerate(update_indices):
        current_normal = current_normals_update[i].detach().cpu().numpy()
        target_normal = target_normals_update[i].detach().cpu().numpy()
        rotation = rotations_update[i].detach().cpu().numpy()
        
        # 使用CPU版本的旋转更新（复杂的四元数运算）
        new_rotation = update_gaussian_rotation_for_normal_cpu(rotation, None, target_normal, current_normal)
        new_rotations[idx] = torch.tensor(new_rotation, device=device)
    
    return new_rotations

def update_gaussian_rotation_for_normal_cpu(rotation, scale, target_normal, current_normal=None):
    """CPU版本的旋转更新（用于复杂的四元数运算）"""
    if current_normal is None:
        # 如果没有提供当前法线，从旋转中计算
        R = quat_to_rot_matrix(rotation)
        current_normal = R[:, 2]
    
    # 如果法线已经正确，不需要更新
    if np.allclose(current_normal, target_normal, atol=1e-6):
        return rotation
    
    # 计算从当前法线到目标法线的旋转
    v = np.cross(current_normal, target_normal)
    s = np.linalg.norm(v)
    c = np.dot(current_normal, target_normal)
    
    if s < 1e-6:  # 法线平行或反平行
        if c > 0:  # 平行，不需要旋转
            return rotation
        else:  # 反平行，需要180度旋转
            if abs(current_normal[0]) < 0.9:
                axis = np.array([1, 0, 0])
            else:
                axis = np.array([0, 1, 0])
            axis = axis - np.dot(axis, current_normal) * current_normal
            axis = axis / np.linalg.norm(axis)
            correction_quat = np.array([0, axis[0], axis[1], axis[2]])
    else:
        # 一般情况的旋转
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        R_correction = np.eye(3) + vx + np.dot(vx, vx) * ((1 - c) / (s * s))
        correction_quat = rot_matrix_to_quat(R_correction)
    
    # 应用校正旋转到原始旋转
    q_orig = rotation  # [w, x, y, z]
    q_corr = correction_quat  # [w, x, y, z]
    
    # 四元数乘法
    w = q_corr[0]*q_orig[0] - q_corr[1]*q_orig[1] - q_corr[2]*q_orig[2] - q_corr[3]*q_orig[3]
    x = q_corr[0]*q_orig[1] + q_corr[1]*q_orig[0] + q_corr[2]*q_orig[3] - q_corr[3]*q_orig[2]
    y = q_corr[0]*q_orig[2] - q_corr[1]*q_orig[3] + q_corr[2]*q_orig[0] + q_corr[3]*q_orig[1]
    z = q_corr[0]*q_orig[3] + q_corr[1]*q_orig[2] - q_corr[2]*q_orig[1] + q_corr[3]*q_orig[0]
    
    new_rotation = np.array([w, x, y, z])
    new_rotation = new_rotation / np.linalg.norm(new_rotation)
    
    return new_rotation

def rot_matrix_to_quat(R):
    """将旋转矩阵转换为四元数 (w, x, y, z)"""
    # 使用Shepperd's method
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    
    if trace > 0:
        s = np.sqrt(trace + 1.0) * 2  # s = 4 * qw
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2  # s = 4 * qx
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2  # s = 4 * qy
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2  # s = 4 * qz
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s
    
    return np.array([qw, qx, qy, qz])

def scale_to_matrix(scale):
    """将缩放向量转换为缩放矩阵"""
    return np.diag([scale[0], scale[1], 1.0])

def get_gaussian_normal_world(rotation, scale):
    """获取高斯点在世界坐标系下的法线方向"""
    R = quat_to_rot_matrix(rotation)
    S = scale_to_matrix([scale[0], scale[1]])
    L = np.dot(R, S)
    return L[:, 2]  # 局部Z轴

def check_gaussian_visibility(gaussians, camera):
    """检查高斯点在特定相机视角下的可见性"""
    from diff_surfel_rasterization import GaussianRasterizer, GaussianRasterizationSettings
    import math
    
    tanfovx = math.tan(camera.FoVx * 0.5)
    tanfovy = math.tan(camera.FoVy * 0.5)
    
    raster_settings = GaussianRasterizationSettings(
        image_height=int(camera.image_height),
        image_width=int(camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda"),
        scale_modifier=1.0,
        viewmatrix=camera.world_view_transform,
        projmatrix=camera.full_proj_transform,
        sh_degree=gaussians.active_sh_degree,
        campos=camera.camera_center,
        prefiltered=False,
        debug=False
    )
    
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    visible_mask = rasterizer.markVisible(gaussians.get_xyz)
    return visible_mask.cpu().numpy()

def quat_to_rot_matrix(q):
    """将四元数转换为旋转矩阵（CPU版本）"""
    q_tensor = torch.tensor(q, device="cuda").unsqueeze(0)
    rot_matrix = build_rotation(q_tensor).squeeze(0).cpu().numpy()
    return rot_matrix

if __name__ == "__main__":
    parser = ArgumentParser()
    model = ModelParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--angle_threshold", type=float, default=90.0,
                      help="法线校正角度阈值(度)")
    parser.add_argument("--output_dir", default="corrected_model", type=str,
                      help="校正后模型保存目录")
    parser.add_argument("--save_original", action="store_true",
                      help="同时保存原始参数用于对比")
    parser.add_argument("--batch_size", type=int, default=10000,
                      help="批处理大小，用于CUDA加速")
    args = parser.parse_args()

    print("=== 2D高斯参数校正工具（CUDA加速版本）===")
    
    # 检查CUDA可用性
    if not torch.cuda.is_available():
        print("警告：CUDA不可用，将使用CPU模式")
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")
        print(f"使用CUDA设备: {torch.cuda.get_device_name()}")
    
    # 加载场景和参数
    dataset = model.extract(args)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
    
    # 提取原始高斯参数并移到GPU
    centers_gpu = gaussians._xyz  # 保持在GPU上
    scales_gpu = gaussians.get_scaling  # 保持在GPU上
    rotations_gpu = gaussians.get_rotation  # 保持在GPU上
    opacities_gpu = gaussians.get_opacity  # 保持在GPU上
    
    # CPU版本用于保存
    centers = centers_gpu.detach().cpu().numpy()
    scales = scales_gpu.detach().cpu().numpy()
    rotations = rotations_gpu.detach().cpu().numpy()
    opacities = opacities_gpu.detach().cpu().numpy()
    
    print(f"加载了 {len(centers)} 个高斯点")
    
    # 获取所有训练相机
    train_cameras = scene.train_cameras[1.0]
    print(f"加载了 {len(train_cameras)} 个训练视角")
    
    # 初始化校正结果（在GPU上）
    corrected_rotations_gpu = rotations_gpu.clone()
    correction_stats = {
        'total_gaussians': len(centers),
        'corrected_count': 0,
        'correction_per_view': [],
        'angle_threshold': args.angle_threshold
    }
    
    print("开始多视角法线校正（CUDA加速）...")
    
    # 对每个相机视角进行法线校正
    for cam_idx, camera in enumerate(train_cameras):
        print(f"处理视角 {cam_idx+1}/{len(train_cameras)}: {camera.image_name}")
        
        try:
            # 检测可见高斯点
            visible_mask = check_gaussian_visibility(gaussians, camera)
            visible_indices = torch.where(torch.tensor(visible_mask, device=device))[0]
            
            if len(visible_indices) == 0:
                print("  没有可见的高斯点")
                continue
                
            print(f"  可见高斯点数量: {len(visible_indices)}")
            
            # 获取相机位置（GPU）
            camera_position = camera.camera_center  # 已经在GPU上
            
            # 批量处理可见点
            batch_size = args.batch_size
            total_corrections = 0
            
            for batch_start in range(0, len(visible_indices), batch_size):
                batch_end = min(batch_start + batch_size, len(visible_indices))
                batch_indices = visible_indices[batch_start:batch_end]
                
                # 获取批次数据
                batch_centers = centers_gpu[batch_indices]  # [B, 3]
                batch_scales = scales_gpu[batch_indices]    # [B, 2]
                batch_rotations = corrected_rotations_gpu[batch_indices]  # [B, 4]
                
                # 计算视线方向（批量）
                view_directions = camera_position.unsqueeze(0) - batch_centers  # [B, 3]
                view_directions = view_directions / torch.norm(view_directions, dim=1, keepdim=True)
                
                # 获取当前法线方向（批量）
                current_normals = get_gaussian_normal_world_cuda(batch_rotations, batch_scales)
                
                # 校正法线方向（批量）
                corrected_normals, flip_mask = correct_normal_direction_cuda(
                    current_normals, view_directions, args.angle_threshold
                )
                
                # 统计需要校正的点
                num_corrections = torch.sum(flip_mask).item()
                total_corrections += num_corrections
                
                if num_corrections > 0:
                    # 更新需要校正的旋转参数
                    corrected_batch_rotations = update_gaussian_rotation_for_normal_cuda(
                        batch_rotations, batch_scales, corrected_normals
                    )
                    
                    # 更新全局旋转参数
                    corrected_rotations_gpu[batch_indices] = corrected_batch_rotations
            
            correction_stats['correction_per_view'].append({
                'view_name': camera.image_name,
                'visible_count': len(visible_indices),
                'corrected_count': total_corrections
            })
            
            print(f"  校正了 {total_corrections} 个高斯点")
            
        except Exception as e:
            print(f"  处理视角 {cam_idx+1} 时出错: {e}")
            continue
    
    # 将结果移回CPU
    corrected_rotations = corrected_rotations_gpu.detach().cpu().numpy()
    
    # 统计总校正数量
    total_corrected = np.sum([not np.allclose(rotations[i], corrected_rotations[i]) 
                             for i in range(len(rotations))])
    correction_stats['corrected_count'] = total_corrected
    
    print(f"\n校正完成！总共校正了 {total_corrected} 个高斯点")
    
    # 创建输出目录
    output_dir = args.output_dir
    mkdir_p(output_dir)
    
    # 创建point_cloud目录结构，与训练保存格式一致
    iteration_str = f"iteration_{scene.loaded_iter if scene.loaded_iter else 'corrected'}"
    point_cloud_dir = os.path.join(output_dir, "point_cloud", iteration_str)
    mkdir_p(point_cloud_dir)
    
    # 保存校正后的高斯参数为PLY格式（与训练代码格式一致）
    def save_corrected_ply(path, centers, scales, corrected_rotations, opacities, features_dc, features_rest, max_sh_degree):
        """保存校正后的高斯参数为PLY格式，与GaussianModel.save_ply完全一致"""
        mkdir_p(os.path.dirname(path))
        
        # 构建属性列表（与GaussianModel.construct_list_of_attributes一致）
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # DC分量
        for i in range(features_dc.shape[1] * features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        # Rest分量
        for i in range(features_rest.shape[1] * features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(scales.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(corrected_rotations.shape[1]):
            l.append('rot_{}'.format(i))
        
        # 准备数据（与训练代码完全一致）
        xyz = centers
        normals = np.zeros_like(xyz)
        
        # 处理球谐系数：transpose(1,2)然后flatten
        f_dc = np.transpose(features_dc, (0, 2, 1)).reshape(features_dc.shape[0], -1)
        f_rest = np.transpose(features_rest, (0, 2, 1)).reshape(features_rest.shape[0], -1)
        
        # 创建数据类型
        dtype_full = [(attribute, 'f4') for attribute in l]
        
        # 组合所有属性
        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scales, corrected_rotations), axis=1)
        elements[:] = list(map(tuple, attributes))
        
        # 保存PLY文件
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)
    
    # 从加载的高斯模型中提取球谐系数
    features_dc = gaussians._features_dc.detach().cpu().numpy()  # [N, 1, 3]
    features_rest = gaussians._features_rest.detach().cpu().numpy()  # [N, SH_rest, 3]
    
    # 保存校正后的模型
    corrected_ply_path = os.path.join(point_cloud_dir, "point_cloud.ply")
    save_corrected_ply(corrected_ply_path, centers, scales, corrected_rotations, opacities, 
                      features_dc, features_rest, gaussians.max_sh_degree)
    print(f"校正后的高斯模型已保存到: {corrected_ply_path}")
    
    # 如果需要，保存原始参数用于对比
    if args.save_original:
        original_ply_path = os.path.join(point_cloud_dir, "point_cloud_original.ply")
        save_corrected_ply(original_ply_path, centers, scales, rotations, opacities, 
                          features_dc, features_rest, gaussians.max_sh_degree)
        print(f"原始高斯模型已保存到: {original_ply_path}")
    
    # 复制相机配置文件
    import shutil
    if os.path.exists(os.path.join(scene.model_path, "cameras.json")):
        shutil.copy2(os.path.join(scene.model_path, "cameras.json"), 
                    os.path.join(output_dir, "cameras.json"))
        print("相机配置文件已复制")
    
    if os.path.exists(os.path.join(scene.model_path, "input.ply")):
        shutil.copy2(os.path.join(scene.model_path, "input.ply"), 
                    os.path.join(output_dir, "input.ply"))
        print("输入点云文件已复制")
    
    # 保存详细统计信息
    stats_file = os.path.join(output_dir, 'correction_stats.pkl')
    with open(stats_file, 'wb') as f:
        pickle.dump(correction_stats, f)
    print(f"校正统计信息已保存到: {stats_file}")
    
    # 保存配置信息
    config_file = os.path.join(output_dir, 'correction_config.txt')
    with open(config_file, 'w') as f:
        f.write(f"原始模型路径: {scene.model_path}\n")
        f.write(f"加载迭代: {scene.loaded_iter}\n")
        f.write(f"角度阈值: {args.angle_threshold}°\n")
        f.write(f"批处理大小: {args.batch_size}\n")
        f.write(f"使用设备: {device}\n")
        f.write(f"校正时间: {__import__('datetime').datetime.now()}\n")
    
    # 打印校正摘要
    print("\n=== 校正摘要 ===")
    print(f"总高斯点数: {correction_stats['total_gaussians']}")
    print(f"校正点数: {correction_stats['corrected_count']}")
    print(f"校正比例: {correction_stats['corrected_count']/correction_stats['total_gaussians']*100:.2f}%")
    print(f"角度阈值: {correction_stats['angle_threshold']}°")
    print(f"批处理大小: {args.batch_size}")
    print(f"使用设备: {device}")
    print(f"\n校正后的模型可以直接用于:")
    print(f"- 渲染: python render.py -m {output_dir}")
    print(f"- 网格提取: python render.py -m {output_dir} --skip_train --skip_test")
    print(f"- 其他后续处理流程") 