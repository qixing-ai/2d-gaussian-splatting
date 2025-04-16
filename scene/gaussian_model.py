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
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation

class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(center, scaling, scaling_modifier, rotation):
            RS = build_scaling_rotation(torch.cat([scaling * scaling_modifier, torch.ones_like(scaling)], dim=-1), rotation).permute(0,2,1)
            trans = torch.zeros((center.shape[0], 4, 4), dtype=torch.float, device="cuda")
            trans[:,:3,:3] = RS
            trans[:, 3,:3] = center
            trans[:, 3, 3] = 1
            return trans
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, sh_degree : int):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling) #.clamp(max=1)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_xyz, self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 2)
        rots = torch.rand((fused_point_cloud.shape[0], 4), device="cuda")

        opacities = self.inverse_opacity_activation(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self):
        opacities_new = self.inverse_opacity_activation(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        # 记录裁剪前的点数，用于调试
        points_before = self.get_xyz.shape[0]
        
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        # 确保梯度累积器和分母大小与点云大小一致
        if self.xyz_gradient_accum.shape[0] != self.get_xyz.shape[0]:
            # 创建新的梯度累积器
            new_grad_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
            # 只复制有效范围内的梯度累积数据
            valid_range = min(self.xyz_gradient_accum.shape[0], self.get_xyz.shape[0])
            if valid_range > 0:
                # 处理valid_points_mask可能超出范围的情况
                valid_mask_range = min(valid_points_mask.sum().item(), valid_range)
                if valid_mask_range > 0:
                    new_grad_accum[:valid_mask_range] = self.xyz_gradient_accum[valid_points_mask][:valid_mask_range]
            self.xyz_gradient_accum = new_grad_accum
        else:
            self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        # 确保分母大小与点云大小一致
        if self.denom.shape[0] != self.get_xyz.shape[0]:
            # 创建新的分母
            new_denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
            # 只复制有效范围内的分母数据
            valid_range = min(self.denom.shape[0], self.get_xyz.shape[0])
            if valid_range > 0:
                # 处理valid_points_mask可能超出范围的情况
                valid_mask_range = min(valid_points_mask.sum().item(), valid_range)
                if valid_mask_range > 0:
                    new_denom[:valid_mask_range] = self.denom[valid_points_mask][:valid_mask_range]
            self.denom = new_denom
        else:
            self.denom = self.denom[valid_points_mask]

        # 更新其他数据大小
        if self.max_radii2D.shape[0] != self.get_xyz.shape[0]:
            # 创建新的最大半径
            new_max_radii = torch.zeros((self.get_xyz.shape[0]), device="cuda")
            # 只复制有效范围内的最大半径数据
            valid_range = min(self.max_radii2D.shape[0], self.get_xyz.shape[0])
            if valid_range > 0:
                # 处理valid_points_mask可能超出范围的情况
                valid_mask_range = min(valid_points_mask.sum().item(), valid_range)
                if valid_mask_range > 0:
                    new_max_radii[:valid_mask_range] = self.max_radii2D[valid_points_mask][:valid_mask_range]
            self.max_radii2D = new_max_radii
        else:
            self.max_radii2D = self.max_radii2D[valid_points_mask]
        
        # 打印裁剪信息
        points_after = self.get_xyz.shape[0]
        if points_before != points_after:
            points_removed = points_before - points_after
            print(f"裁剪: {points_before} -> {points_after} (移除了 {points_removed} 点)")

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        
        # 确保梯度张量大小与点云大小匹配
        if grads.shape[0] != n_init_points:
            # 创建一个合适大小的填充梯度张量
            padded_grad = torch.zeros((n_init_points), device="cuda")
            # 只复制有效范围内的梯度数据
            valid_range = min(grads.shape[0], n_init_points)
            if valid_range > 0:
                # 对梯度张量取范数并保留有效部分
                norm_grads = torch.norm(grads[:valid_range], dim=-1)
                padded_grad[:valid_range] = norm_grads
        else:
            # 梯度大小匹配，正常计算
            padded_grad = torch.norm(grads, dim=-1)
        
        # 根据梯度和大小条件选择需要分裂的点
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        # 如果没有选中任何点，直接返回
        if not selected_pts_mask.any():
            return

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        stds = torch.cat([stds, 0 * torch.ones_like(stds[:,:1])], dim=-1)
        means = torch.zeros_like(stds)
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # 确保梯度张量的大小与点云大小匹配
        if grads.shape[0] != self.get_xyz.shape[0]:
            print(f"克隆操作 - 调整梯度张量大小: {grads.shape[0]} -> {self.get_xyz.shape[0]}")
            # 创建一个全零梯度掩码
            grad_mask = torch.zeros(self.get_xyz.shape[0], dtype=torch.bool, device="cuda")
            # 创建一个全零梯度范数张量
            norm_grads = torch.zeros(self.get_xyz.shape[0], dtype=torch.float32, device="cuda")
            # 只对有梯度的点设置掩码（确保不超出索引范围）
            valid_pts = min(grads.shape[0], self.get_xyz.shape[0])
            if valid_pts > 0:
                # 对梯度张量取范数并保留有效部分
                valid_norms = torch.norm(grads[:valid_pts], dim=-1)
                norm_grads[:valid_pts] = valid_norms
                grad_mask[:valid_pts] = valid_norms >= grad_threshold
        else:
            # 梯度大小匹配，正常计算
            norm_grads = torch.norm(grads, dim=-1)
            grad_mask = norm_grads >= grad_threshold

        # 应用梯度条件和缩放条件
        selected_pts_mask = torch.logical_and(grad_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        # 如果没有选中任何点，直接返回
        if not selected_pts_mask.any():
            return
            
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        # 计算梯度
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        # 记录裁剪前的点数，用于调试
        points_before = self.get_xyz.shape[0]
        
        # 先执行裁剪操作，移除不透明度低的点，但设置一个更低的阈值
        # 更保守的裁剪，降低阈值至原来的10%（之前是30%）
        adjusted_min_opacity = min_opacity * 0.1
        prune_mask = (self.get_opacity < adjusted_min_opacity).squeeze()
        
        if max_screen_size:
            # 进一步增加大尺寸点的容忍度，降低裁剪数量
            big_points_vs = self.max_radii2D > (max_screen_size * 3.0)  # 增加到3.0倍容忍度（之前是2.0）
            big_points_ws = self.get_scaling.max(dim=1).values > 0.25 * extent  # 从0.2提高到0.25
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        
        # 去掉5%限制，直接使用阈值裁剪
        total_points = self.get_xyz.shape[0]
        points_to_prune = prune_mask.sum().item()
        print(f"使用阈值裁剪: 将移除 {points_to_prune} 点 ({(points_to_prune/total_points)*100:.2f}%)")
        
        # 执行裁剪
        self.prune_points(prune_mask)
        
        # 记录裁剪后的点数，用于调试
        points_after = self.get_xyz.shape[0]
        points_removed = points_before - points_after
        print(f"稠密化裁剪: {points_before} -> {points_after} (移除了 {points_removed} 点, {(points_removed/points_before)*100:.2f}%)")
        
        # 重要：确保梯度张量与当前点云大小匹配
        if grads.shape[0] != points_after:
            print(f"调整梯度张量大小 - 裁剪前: {grads.shape[0]}, 裁剪后: {points_after}")
            # 创建新的梯度张量，大小与当前点云匹配
            new_grads = torch.zeros((points_after, grads.shape[1]), device="cuda", dtype=grads.dtype)
            # 复制有效范围内的梯度
            valid_range = min(grads.shape[0], points_after)
            if valid_range > 0:
                new_grads[:valid_range] = grads[:valid_range]
            grads = new_grads
        
        # 然后执行稠密化和分裂操作
        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        # 打印最终点数
        final_points = self.get_xyz.shape[0]
        if final_points != points_after:
            print(f"稠密化后: {points_after} -> {final_points} (添加了 {final_points-points_after} 点)")
        
        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        # 原始2DGS方法 - 使用梯度范数
        # self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter], dim=-1, keepdim=True)
        
        # AbsGS改进方法 - 计算绝对值和而非范数
        grad = viewspace_point_tensor.grad[update_filter]
        grad_abs = torch.abs(grad)
        self.xyz_gradient_accum[update_filter] += torch.sum(grad_abs, dim=-1, keepdim=True)
        
        self.denom[update_filter] += 1
        
    def set_background_opacity_to_zero(self, foreground_mask, visibility_filter=None, bg_opacity_factor=0.0):
        """
        将背景区域的高斯点不透明度直接设置为0
        
        Args:
            foreground_mask: 前景掩码 [B,1,H,W]，值为1表示前景，0表示背景
            visibility_filter: 可见性过滤器，指定哪些高斯点需要处理
            bg_opacity_factor: 忽略此参数，保持为0以向后兼容
        """
        # 当前实现无法直接通过前景掩码标识哪些高斯点位于背景区域
        # 所以我们只能处理可见高斯点，即visibility_filter标识的点
        
        # 如果没有提供可见性过滤器，则不处理任何点
        if visibility_filter is None or not visibility_filter.any():
            return False, 0
            
        # 获取当前不透明度
        opacity = self.get_opacity
        
        # 创建新的不透明度张量，所有可见高斯点不透明度设为0
        new_opacity = opacity.clone()
        new_opacity[visibility_filter] = 0.0
        
        # 更新优化器中的不透明度值
        opacities_new = self.inverse_opacity_activation(new_opacity)
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]
        
        # 返回处理状态和处理的点数量
        num_processed = visibility_filter.sum().item()
        return True, num_processed