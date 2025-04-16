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
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrixShift, fov2focal

class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image, gt_alpha_mask,
                 gt_normal,
                 image_name, uid, principal_point_ndc,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda",
                 distortion_params=None, camera_model="PINHOLE"
                 ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name
        self.camera_model = camera_model
        self.distortion_params = distortion_params
        self.gt_normal = gt_normal

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        self.original_image = image.clamp(0.0, 1.0) # move to device at dataloader to reduce VRAM requirement
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]

        if gt_alpha_mask is not None:
            # self.original_image *= gt_alpha_mask.to(self.data_device)
            self.gt_alpha_mask = gt_alpha_mask.to(self.data_device)
        else:
            # self.original_image *= torch.ones((1, self.image_height, self.image_width), device=self.data_device) # do we need this?
            self.gt_alpha_mask = None

        if self.gt_normal is not None:
            self.gt_normal = self.gt_normal.to(self.data_device)

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrixShift(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy, width=self.image_width, height=self.image_height, principal_point_ndc=principal_point_ndc).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

    def has_distortion(self):
        """检查相机是否具有畸变参数"""
        return (self.camera_model in ["SIMPLE_RADIAL", "RADIAL"] and 
                self.distortion_params is not None)
    
    def get_image_coords(self, world_coords):
        """获取投影到图像平面的坐标
        
        如果相机有畸变参数，会应用畸变模型
        """
        # 首先使用标准投影矩阵获取未畸变的归一化设备坐标
        cam_coords = torch.matmul(world_coords, self.world_view_transform.transpose(0, 1))
        cam_coords = torch.matmul(cam_coords, self.projection_matrix.transpose(0, 1))
        
        # 归一化
        cam_coords = cam_coords / cam_coords[:, 3].unsqueeze(1)
        
        # 如果有畸变参数，应用畸变
        if self.has_distortion():
            # 从 NDC 转换回相机坐标
            x = cam_coords[:, 0].cpu().numpy()
            y = cam_coords[:, 1].cpu().numpy()
            
            # 应用径向畸变（注意：这是一个简化模型，实际中可能更复杂）
            from utils.graphics_utils import apply_radial_distortion
            dx, dy = apply_radial_distortion(x, y, self.distortion_params)
            
            # 更新归一化设备坐标
            cam_coords[:, 0] = torch.from_numpy(x + dx).to(cam_coords.device)
            cam_coords[:, 1] = torch.from_numpy(y + dy).to(cam_coords.device)
        
        return cam_coords
    
    def undistort_image(self):
        """对相机图像进行去畸变处理
        
        返回去畸变后的图像
        """
        if not self.has_distortion():
            # 如果没有畸变参数，直接返回原始图像
            return self.original_image
        
        # 实现图像去畸变
        import torch.nn.functional as F
        import numpy as np
        
        # 创建目标坐标网格 (归一化设备坐标系 NDC)
        h, w = self.image_height, self.image_width
        y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
        
        # 将像素坐标转换为归一化坐标 [-1, 1]
        x_norm = (x.float() / (w - 1) * 2 - 1).to(self.data_device)
        y_norm = (y.float() / (h - 1) * 2 - 1).to(self.data_device)
        
        # 计算像素坐标对应的相机坐标 (通过反转投影)
        fx = fov2focal(self.FoVx, w)
        fy = fov2focal(self.FoVy, h)
        cx = w / 2
        cy = h / 2
        
        # 将归一化设备坐标转换为相机坐标
        x_cam = (x.float() - cx) / fx
        y_cam = (y.float() - cy) / fy
        
        # 转换为 numpy 进行处理
        x_cam_np = x_cam.cpu().numpy().flatten()
        y_cam_np = y_cam.cpu().numpy().flatten()
        
        # 使用向量化操作对所有坐标进行去畸变
        from utils.graphics_utils import undistort_points_vectorized
        x_undist, y_undist = undistort_points_vectorized(x_cam_np, y_cam_np, self.distortion_params)
        
        # 转回像素坐标
        x_undist = x_undist * fx + cx
        y_undist = y_undist * fy + cy
        
        # 归一化到 [-1, 1] 范围
        x_undist = x_undist / (w - 1) * 2 - 1
        y_undist = y_undist / (h - 1) * 2 - 1
        
        # 重塑为网格形状
        x_undist = x_undist.reshape(h, w)
        y_undist = y_undist.reshape(h, w)
        
        # 创建采样网格
        grid = torch.zeros((h, w, 2), device=self.data_device)
        grid[..., 0] = torch.tensor(x_undist, device=self.data_device)
        grid[..., 1] = torch.tensor(y_undist, device=self.data_device)
        
        # 应用采样
        grid = grid.unsqueeze(0)  # 添加批次维度
        undistorted = F.grid_sample(self.original_image.unsqueeze(0), grid, align_corners=True)
        
        return undistorted.squeeze(0)

class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]

