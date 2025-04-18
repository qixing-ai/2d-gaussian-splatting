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

from scene.cameras import Camera
import numpy as np
from utils.general_utils import PILtoTorch
from utils.graphics_utils import fov2focal
import os
from PIL import Image

WARNED = False

def loadCam(args, id, cam_info, resolution_scale):
    orig_w, orig_h = cam_info.image.size

    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    if len(cam_info.image.split()) > 3:
        import torch
        resized_image_rgb = torch.cat([PILtoTorch(im, resolution) for im in cam_info.image.split()[:3]], dim=0)
        loaded_mask = PILtoTorch(cam_info.image.split()[3], resolution)
        gt_image = resized_image_rgb
    else:
        resized_image_rgb = PILtoTorch(cam_info.image, resolution)
        loaded_mask = None
        gt_image = resized_image_rgb

    # --- Load Ground Truth Normal ---
    loaded_normal = None
    if hasattr(args, 'gt_normal_path') and args.gt_normal_path is not None and hasattr(args, 'gt_normal_suffix'):
        try:
            base_name_with_ext = os.path.basename(cam_info.image_name)
            base_name = os.path.splitext(base_name_with_ext)[0]
            normal_filename = base_name + args.gt_normal_suffix
            normal_path = os.path.join(args.gt_normal_path, normal_filename)

            if os.path.exists(normal_path):
                normal_image = Image.open(normal_path)
                if normal_image.mode != 'RGB':
                   try:
                       normal_image = normal_image.convert('RGB')
                   except ValueError:
                       print(f"Warning: Could not convert normal map {normal_path} to RGB. Skipping normal for this view.")
                       normal_image = None

                if normal_image:
                    loaded_normal = PILtoTorch(normal_image, resolution)
                    # Optional: Normalize normal values if needed (e.g., to [-1, 1])
                    # loaded_normal = (loaded_normal * 2.0) - 1.0
            else:
                 pass # 添加 pass 避免空块

        except Exception as e:
            pass # 添加 pass 避免空块
    # --- End Load Ground Truth Normal ---

    # 获取相机模型类型和畸变参数
    camera_model = getattr(cam_info, 'camera_model', "PINHOLE")
    distortion_params = getattr(cam_info, 'distortion_params', None)

    return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T,
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY,
                  image=gt_image, gt_alpha_mask=loaded_mask,
                  gt_normal=loaded_normal,
                  image_name=cam_info.image_name, uid=id,
                  principal_point_ndc=cam_info.principal_point_ndc,
                  data_device=args.data_device,
                  camera_model=camera_model,
                  distortion_params=distortion_params)

def cameraList_from_camInfos(cam_infos, resolution_scale, args):
    camera_list = []

    for id, c in enumerate(cam_infos):
        camera_list.append(loadCam(args, id, c, resolution_scale))

    return camera_list

def camera_to_JSON(id, camera : Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(camera.FovY, camera.height),
        'fx' : fov2focal(camera.FovX, camera.width)
    }
    return camera_entry