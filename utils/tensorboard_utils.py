import os
import torch
import numpy as np
from plyfile import PlyData
import matplotlib.pyplot as plt
import io
import trimesh
from typing import Callable, Any, Dict, List, Tuple
from utils.image_utils import psnr, colormap
from scene import Scene
from gaussian_renderer import GaussianModel

def add_ply_to_tensorboard(writer, tag, ply_path, global_step=None):
    """将PLY文件添加到TensorBoard中可视化
    
    Args:
        writer: TensorBoard SummaryWriter实例
        tag: 在TensorBoard中显示的标签
        ply_path: PLY文件路径
        global_step: 全局步数
    """
    try:
        # 读取点云并添加到TensorBoard
        mesh = trimesh.load(ply_path)
        
        # 提取点和颜色
        vertices = mesh.vertices
        colors = mesh.colors / 255.0 if mesh.colors is not None else np.ones_like(vertices)
        
        # 添加到TensorBoard
        writer.add_mesh(
            tag,
            vertices=[vertices],
            colors=[colors],
            global_step=global_step
        )
        
    except Exception as e:
        print(f"无法添加PLY到TensorBoard: {e}")
        # 记录文件路径
        writer.add_text(f'{tag}/ply_path', f"PLY文件: {ply_path}", global_step)

def log_tb_image(tb_writer, tag_prefix, image_name, image_tensor, global_step, cmap=None, normalize=True):
    """辅助函数，用于将图像记录到 TensorBoard"""
    if tb_writer is None:
        return
    
    processed_image = image_tensor.detach().cpu()
    
    # 如果需要，进行归一化
    if normalize and processed_image.numel() > 0: # 仅在张量非空时进行归一化
        norm = processed_image.max()
        if norm > 1e-6: # 避免除以零或非常小的值
            processed_image = processed_image / norm
        else:
            processed_image.zero_() # 如果最大值接近零，则将图像设为全黑

    # 如果指定了 colormap，则应用它
    if cmap is not None:
        # 确保输入是单通道图像
        if processed_image.shape[0] != 1:
             print(f"警告：Colormap 只能应用于单通道图像，但收到了 {processed_image.shape[0]} 通道。跳过 Colormap 应用。")
        elif processed_image.numel() > 0: # 仅在张量非空时应用 colormap
             # 将张量数据转换为 NumPy 数组，并传递设备信息
             image_np = processed_image.cpu().numpy()[0] # 移到 CPU 并转 NumPy
             processed_image = colormap(image_np, cmap=cmap, device=processed_image.device) # 传递原始设备
        else:
             processed_image = torch.zeros((3, processed_image.shape[-2], processed_image.shape[-1]), dtype=torch.uint8) # 创建一个空的彩色图像占位符
    else:
        # 确保图像是 3 通道 (RGB)
        if processed_image.shape[0] == 1:
            processed_image = processed_image.repeat(3, 1, 1) # 灰度图转 RGB

    # 添加批次维度 (N)
    if processed_image.dim() == 3:
        processed_image = processed_image[None] # [C, H, W] -> [N, C, H, W]

    # 确保数据类型是适合 add_images 的类型 (通常是 float 或 uint8)
    if processed_image.dtype != torch.uint8 and processed_image.dtype != torch.float32:
         processed_image = processed_image.float() # 转换为 float

    # 检查形状是否有效
    if processed_image.dim() != 4 or processed_image.shape[1] not in [1, 3, 4]:
        print(f"警告：跳过 TensorBoard 记录，图像形状无效: {processed_image.shape} for tag {tag_prefix}_view_{image_name}")
        return
        
    try:
        tb_writer.add_images(f"{tag_prefix}_view_{image_name}", processed_image, global_step=global_step)
    except Exception as e:
        print(f"记录 TensorBoard 图像时出错 ({tag_prefix}_view_{image_name}): {e}")
        print(f"图像形状: {processed_image.shape}, 数据类型: {processed_image.dtype}, Min: {processed_image.min()}, Max: {processed_image.max()}")

@torch.no_grad()
def training_report(
    tb_writer,
    iteration: int,
    Ll1: torch.Tensor, # 注意：这里是报告用的 L1，不是计算总 loss 用的 masked L1
    loss: torch.Tensor,
    l1_loss_func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    elapsed: float,
    testing_iterations: List[int],
    scene: Scene,
    renderFunc: Callable[..., Dict[str, Any]],
    renderArgs: Tuple[Any, ...]
):
    """在 TensorBoard 上记录训练和测试指标及图像"""
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/reg_loss', Ll1.item(), iteration) # Ll1 is unmasked L1 for reporting
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)
        tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()},
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    render_pkg = renderFunc(viewpoint, scene.gaussians, *renderArgs)
                    image = torch.clamp(render_pkg["render"], 0.0, 1.0).to("cuda")
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)

                    # Log images to TensorBoard (first 5 views)
                    if tb_writer and (idx < 5):
                        view_name = viewpoint.image_name
                        log_tb_image(tb_writer, f"{config['name']}/Render", view_name, image, iteration, normalize=False)

                        # Log depth map
                        if "surf_depth" in render_pkg:
                            depth = render_pkg["surf_depth"]
                            # 检查深度图是否有效 (非空且有正值)
                            if depth.numel() > 0 and depth.max() > 1e-6:
                                log_tb_image(tb_writer, f"{config['name']}/Depth", view_name, depth, iteration, cmap='turbo', normalize=True)
                            else:
                                print(f"警告：迭代 {iteration}, 视角 {view_name} 的深度图无效或全黑，跳过记录。")


                        # Log normal maps (rendered and surface)
                        if "rend_normal" in render_pkg:
                             log_tb_image(tb_writer, f"{config['name']}/Normal_Rendered", view_name, render_pkg["rend_normal"] * 0.5 + 0.5, iteration, normalize=False)
                        if "surf_normal" in render_pkg:
                             log_tb_image(tb_writer, f"{config['name']}/Normal_Surface", view_name, render_pkg["surf_normal"] * 0.5 + 0.5, iteration, normalize=False)

                        # Log alpha map
                        if "rend_alpha" in render_pkg:
                            log_tb_image(tb_writer, f"{config['name']}/Alpha", view_name, render_pkg["rend_alpha"], iteration, normalize=False)

                        # Log distortion map
                        if "rend_dist" in render_pkg:
                            dist = render_pkg["rend_dist"]
                             # 检查失真图是否有效
                            if dist.numel() > 0:
                                log_tb_image(tb_writer, f"{config['name']}/Distortion", view_name, dist, iteration, cmap='viridis', normalize=True) # 使用 viridis colormap
                            else:
                                print(f"警告：迭代 {iteration}, 视角 {view_name} 的失真图无效，跳过记录。")


                        # Log ground truth only on the first testing iteration
                        if iteration == testing_iterations[0]:
                            log_tb_image(tb_writer, f"{config['name']}/Ground_Truth", view_name, gt_image, iteration, normalize=False)

                    # Calculate metrics
                    # 使用传入的 l1_loss 函数
                    l1_test += l1_loss_func(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()

                # Average metrics over views
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print(f"[ITER {iteration}] Evaluating {config['name']}: L1 {l1_test:.{5}f} PSNR {psnr_test:.{5}f}")
                if tb_writer:
                    tb_writer.add_scalar(f"{config['name']}/loss_viewpoint - l1_loss", l1_test, iteration)
                    tb_writer.add_scalar(f"{config['name']}/loss_viewpoint - psnr", psnr_test, iteration)

        torch.cuda.empty_cache() 