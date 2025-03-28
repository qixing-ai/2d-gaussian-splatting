#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import math
import numpy as np
from typing import NamedTuple

class BasicPointCloud(NamedTuple):
    points : np.array
    colors : np.array
    normals : np.array

def geom_transform_points(points, transf_matrix):
    P, _ = points.shape
    ones = torch.ones(P, 1, dtype=points.dtype, device=points.device)
    points_hom = torch.cat([points, ones], dim=1)
    points_out = torch.matmul(points_hom, transf_matrix.unsqueeze(0))

    denom = points_out[..., 3:] + 0.0000001
    return (points_out[..., :3] / denom).squeeze(dim=0)

def getWorld2View(R, t):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    return np.float32(Rt)

def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)

def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

def getProjectionMatrixShift(znear, zfar, fovX, fovY, width, height, principal_point_ndc):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    # the origin at center of image plane
    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    # shift the frame window due to the non-zero principle point offsets
    cx = width * principal_point_ndc[0]
    cy = height * principal_point_ndc[1]
    focal_x = fov2focal(fovX, width)
    focal_y = fov2focal(fovY, height)
    offset_x = cx - (width / 2)
    offset_x = (offset_x / focal_x) * znear
    offset_y = cy - (height / 2)
    offset_y = (offset_y / focal_y) * znear

    top = top + offset_y
    left = left + offset_x
    right = right + offset_x
    bottom = bottom + offset_y

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))

def apply_radial_distortion(x, y, k):
    """应用径向畸变
    
    Args:
        x, y: 归一化的相机坐标
        k: 径向畸变参数，可以是一个值 (SIMPLE_RADIAL) 或两个值 (RADIAL)
        
    Returns:
        dx, dy: 畸变位移量
    """
    r2 = x*x + y*y
    
    if len(k) == 1:
        # SIMPLE_RADIAL: 只有一个径向畸变系数 k
        radial = k[0] * r2
    elif len(k) == 2:
        # RADIAL: 有两个径向畸变系数 k1, k2
        radial = k[0] * r2 + k[1] * r2 * r2
    else:
        raise ValueError(f"Unsupported number of distortion parameters: {len(k)}")
    
    dx = x * radial
    dy = y * radial
    
    return dx, dy

def undistort_points(x, y, k, max_iterations=100, tolerance=1e-8):
    """使用迭代方法对点进行去畸变
    
    Args:
        x, y: 畸变后的归一化相机坐标
        k: 径向畸变参数
        max_iterations: 最大迭代次数
        tolerance: 收敛容差
        
    Returns:
        x_undist, y_undist: 去畸变后的归一化相机坐标
    """
    # 初始猜测值为输入坐标
    x_undist, y_undist = x, y
    
    for i in range(max_iterations):
        # 计算当前猜测值产生的畸变
        dx, dy = apply_radial_distortion(x_undist, y_undist, k)
        
        # 计算畸变后的坐标
        x_dist = x_undist + dx
        y_dist = y_undist + dy
        
        # 计算畸变坐标与目标坐标之间的误差
        error_x = x_dist - x
        error_y = y_dist - y
        
        # 更新猜测值
        x_undist = x_undist - error_x
        y_undist = y_undist - error_y
        
        # 如果误差足够小，则退出迭代
        if abs(error_x) < tolerance and abs(error_y) < tolerance:
            break
    
    return x_undist, y_undist

def apply_radial_distortion_vectorized(x, y, k):
    """应用径向畸变的向量化版本
    
    Args:
        x, y: 归一化的相机坐标数组
        k: 径向畸变参数，可以是一个值 (SIMPLE_RADIAL) 或两个值 (RADIAL)
        
    Returns:
        dx, dy: 畸变位移量数组
    """
    r2 = x*x + y*y
    
    if len(k) == 1:
        # SIMPLE_RADIAL: 只有一个径向畸变系数 k
        radial = k[0] * r2
    elif len(k) == 2:
        # RADIAL: 有两个径向畸变系数 k1, k2
        radial = k[0] * r2 + k[1] * r2 * r2
    else:
        raise ValueError(f"Unsupported number of distortion parameters: {len(k)}")
    
    dx = x * radial
    dy = y * radial
    
    return dx, dy

def undistort_points_vectorized(x, y, k, max_iterations=20, tolerance=1e-8):
    """使用迭代方法对点数组进行去畸变
    
    Args:
        x, y: 畸变后的归一化相机坐标数组
        k: 径向畸变参数
        max_iterations: 最大迭代次数
        tolerance: 收敛容差
        
    Returns:
        x_undist, y_undist: 去畸变后的归一化相机坐标数组
    """
    # 初始猜测值为输入坐标
    x_undist, y_undist = x.copy(), y.copy()
    
    for i in range(max_iterations):
        # 计算当前猜测值产生的畸变
        dx, dy = apply_radial_distortion_vectorized(x_undist, y_undist, k)
        
        # 计算畸变后的坐标
        x_dist = x_undist + dx
        y_dist = y_undist + dy
        
        # 计算畸变坐标与目标坐标之间的误差
        error_x = x_dist - x
        error_y = y_dist - y
        
        # 更新猜测值
        x_undist = x_undist - error_x
        y_undist = y_undist - error_y
        
        # 计算最大误差
        max_error = max(np.max(np.abs(error_x)), np.max(np.abs(error_y)))
        
        # 如果误差足够小，则退出迭代
        if max_error < tolerance:
            break
    
    return x_undist, y_undist