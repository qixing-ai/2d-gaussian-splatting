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

import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim, compute_color_loss, edge_aware_normal_loss, depth_convergence_loss, background_loss
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr, render_net_image, colormap
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import matplotlib.pyplot as plt
import numpy as np
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    ema_dist_for_log = 0.0
    ema_normal_for_log = 0.0
    ema_conv_for_log = 0.0  # 用于记录深度收敛损失
    ema_bg_for_log = 0.0  # 用于记录背景损失
    
    # 添加视角loss记录
    viewpoint_losses = {}  # 记录每个视角的loss
    all_losses = []  # 记录所有loss
    dynamic_threshold = 2.0  # 初始阈值
    threshold_update_interval = 100  # 每100次迭代更新一次阈值
    threshold_multiplier = 1.5  # 阈值倍数，高于平均loss的1.5倍视为高loss

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):        

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        
        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        
        gt_image = viewpoint_cam.original_image.cuda()
        
        # 颜色损失计算
        Ll1 = l1_loss(image, gt_image)
        
        # 检查是否启用了多尺度SSIM
        if hasattr(opt, 'use_ms_ssim') and opt.use_ms_ssim:
            loss = compute_color_loss(image, gt_image, lambda_dssim=opt.lambda_dssim, use_ms_ssim=True)
        else:
            # 原始颜色损失计算
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        
        # regularization
        lambda_normal = opt.lambda_normal if iteration > 7000 else 0.0
        lambda_dist = opt.lambda_dist if iteration > 3000 else 0.0
        # 深度收敛损失权重设置
        lambda_conv = opt.lambda_depth_convergence if hasattr(opt, 'use_depth_convergence') and opt.use_depth_convergence and iteration > opt.conv_start_iter else 0.0
        # 背景损失权重设置
        lambda_bg = opt.lambda_background if hasattr(opt, 'use_background_loss') and opt.use_background_loss and iteration > opt.bg_start_iter else 0.0

        rend_dist = render_pkg["rend_dist"]
        rend_normal = render_pkg['rend_normal']
        surf_normal = render_pkg['surf_normal']
        
        # 法线损失计算
        if lambda_normal > 0:
            # 检查是否启用了边缘感知法向损失
            if hasattr(opt, 'use_edge_aware_normal') and opt.use_edge_aware_normal:
                # 使用边缘感知法向损失
                edge_weight_exp = opt.edge_weight_exponent if hasattr(opt, 'edge_weight_exponent') else 4.0
                lambda_cons = opt.lambda_consistency if hasattr(opt, 'lambda_consistency') else 0.5
                
                normal_loss = lambda_normal * edge_aware_normal_loss(
                    rendered_normal=rend_normal,
                    gt_rgb=gt_image,
                    surf_normal=surf_normal,
                    q=edge_weight_exp,
                    lambda_consistency=lambda_cons
                )
            else:
                # 使用原始法线一致性损失
                normal_error = (1 - (rend_normal * surf_normal).sum(dim=0))[None]
                normal_loss = lambda_normal * (normal_error).mean()
        else:
            normal_loss = torch.tensor(0.0, device='cuda')
            
        dist_loss = lambda_dist * (rend_dist).mean()
        
        # 计算深度收敛损失
        if lambda_conv > 0:
            # 获取渲染深度和渲染不透明度
            surf_depth = render_pkg['surf_depth']
            render_alpha = render_pkg['rend_alpha']
            # 计算深度收敛损失
            convergence_loss = lambda_conv * depth_convergence_loss(surf_depth, render_alpha)
        else:
            convergence_loss = torch.tensor(0.0, device='cuda')

        # 计算背景损失
        if lambda_bg > 0:
            # 获取渲染的alpha通道
            render_alpha = render_pkg['rend_alpha']
            
            # 每100次迭代打印一次背景损失调试信息
            if iteration % 100 == 0:
                # 计算背景损失并获取调试信息
                bg_loss_val, debug_info = background_loss(render_alpha, gt_image, threshold=0.1, debug=True)
                bg_loss = lambda_bg * bg_loss_val
                
                # 打印调试信息
                print(f"\n[背景损失调试] 迭代: {iteration}")
                print(f"  背景区域平均Alpha: {debug_info['bg_alpha_avg'].item():.4f}")
                print(f"  前景区域平均Alpha: {debug_info['fg_alpha_avg'].item():.4f}")
                print(f"  背景损失值: {bg_loss.item():.6f}")
                print(f"  背景像素比例: {debug_info['bg_pixel_ratio'].item():.2f}")
                
                # 如果debug_info中有自适应阈值信息，则打印
                if 'adaptive_threshold' in debug_info:
                    print(f"  使用自适应阈值: {debug_info['adaptive_threshold'].item():.4f}")
                else:
                    print(f"  使用固定阈值: 0.1")
                
                # 如果是训练初期，每1000次迭代记录前景/背景掩码到tensorboard
                if tb_writer is not None and iteration <= 5000 and iteration % 1000 == 0:
                    # 修复维度问题，确保掩码是2D格式 (HW)
                    background_mask = debug_info['background_mask'].squeeze()  # 从 [1,H,W] 变为 [H,W]
                    foreground_mask = debug_info['foreground_mask'].squeeze()
                    
                    # 添加到tensorboard
                    tb_writer.add_image('masks/background', background_mask, iteration, dataformats='HW')
                    tb_writer.add_image('masks/foreground', foreground_mask, iteration, dataformats='HW')
                    
                    # 保存背景掩码可视化图像
                    bg_mask_path = save_mask_visualization(
                        background_mask,  # 已经squeeze过的掩码
                        gt_image, 
                        iteration, 
                        scene.model_path, 
                        "background_mask"
                    )
                    print(f"  背景掩码可视化已保存到: {bg_mask_path}")
                    
                    # 记录到TensorBoard
                    if tb_writer is not None:
                        bg_image = plt.imread(bg_mask_path)
                        bg_image = np.transpose(bg_image[:, :, :3], (2, 0, 1))  # HWC -> CHW
                        tb_writer.add_image('visualizations/background_mask', bg_image, iteration)
            else:
                # 计算普通背景损失
                bg_loss = lambda_bg * background_loss(render_alpha, gt_image)
        else:
            bg_loss = torch.tensor(0.0, device='cuda')

        # 总损失
        total_loss = loss + dist_loss + normal_loss + convergence_loss + bg_loss
        
        # 记录当前视角的loss
        viewpoint_name = viewpoint_cam.image_name
        current_loss = total_loss.item()
        
        # 更新视角平均loss
        if viewpoint_name not in viewpoint_losses:
            viewpoint_losses[viewpoint_name] = current_loss
        else:
            # 使用指数移动平均更新loss
            viewpoint_losses[viewpoint_name] = 0.7 * viewpoint_losses[viewpoint_name] + 0.3 * current_loss
        
        # 更新动态阈值
        all_losses.append(current_loss)
        if iteration % threshold_update_interval == 0 and len(all_losses) > 0:
            # 计算最近1000个loss的平均值
            recent_losses = all_losses[-1000:] if len(all_losses) > 1000 else all_losses
            avg_loss = sum(recent_losses) / len(recent_losses)
            dynamic_threshold = avg_loss * threshold_multiplier
        
        total_loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_dist_for_log = 0.4 * dist_loss.item() + 0.6 * ema_dist_for_log
            ema_normal_for_log = 0.4 * normal_loss.item() + 0.6 * ema_normal_for_log
            ema_conv_for_log = 0.4 * convergence_loss.item() + 0.6 * ema_conv_for_log
            ema_bg_for_log = 0.4 * bg_loss.item() + 0.6 * ema_bg_for_log

            if iteration % 10 == 0:
                loss_dict = {
                    "Loss": f"{ema_loss_for_log:.{5}f}",
                    "distort": f"{ema_dist_for_log:.{5}f}",
                    "normal": f"{ema_normal_for_log:.{5}f}",
                    "conv": f"{ema_conv_for_log:.{5}f}",
                    "bg": f"{ema_bg_for_log:.{5}f}",
                    "Points": f"{len(gaussians.get_xyz)}"
                }
                progress_bar.set_postfix(loss_dict)

                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            if tb_writer is not None:
                tb_writer.add_scalar('train_loss_patches/dist_loss', ema_dist_for_log, iteration)
                tb_writer.add_scalar('train_loss_patches/normal_loss', ema_normal_for_log, iteration)
                tb_writer.add_scalar('train_loss_patches/conv_loss', ema_conv_for_log, iteration)
                tb_writer.add_scalar('train_loss_patches/bg_loss', ema_bg_for_log, iteration)
                tb_writer.add_scalar('train_loss_patches/dynamic_threshold', dynamic_threshold, iteration)
                
                # 记录当前视角的loss
                tb_writer.add_scalar(f'viewpoint_losses/{viewpoint_name}', current_loss, iteration)
                
                # 每1000次迭代记录一次所有视角的平均loss
                if iteration % 1000 == 0:
                    # 找出loss最高的5个视角
                    sorted_viewpoints = sorted(viewpoint_losses.items(), key=lambda x: x[1], reverse=True)
                    top_5_viewpoints = sorted_viewpoints[:5]
                    
                    # 记录到TensorBoard
                    for i, (name, loss_val) in enumerate(top_5_viewpoints):
                        tb_writer.add_scalar(f'top_loss_viewpoints/rank_{i+1}', loss_val, iteration)
                        tb_writer.add_text(f'top_loss_viewpoints/name_{i+1}', name, iteration)

            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, opt.opacity_cull, scene.cameras_extent, size_threshold)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

        with torch.no_grad():        
            if network_gui.conn == None:
                network_gui.try_connect(dataset.render_items)
            while network_gui.conn != None:
                try:
                    net_image_bytes = None
                    custom_cam, do_training, keep_alive, scaling_modifer, render_mode = network_gui.receive()
                    if custom_cam != None:
                        render_pkg = render(custom_cam, gaussians, pipe, background, scaling_modifer)   
                        net_image = render_net_image(render_pkg, dataset.render_items, render_mode, custom_cam)
                        net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                    metrics_dict = {
                        "#": gaussians.get_opacity.shape[0],
                        "loss": ema_loss_for_log
                        # Add more metrics as needed
                    }
                    # Send the data
                    network_gui.send(net_image_bytes, dataset.source_path, metrics_dict)
                    if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                        break
                except Exception as e:
                    # raise e
                    network_gui.conn = None

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

@torch.no_grad()
def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/reg_loss', Ll1.item(), iteration)
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
                    if tb_writer and (idx < 5):
                        from utils.general_utils import colormap
                        depth = render_pkg["surf_depth"]
                        norm = depth.max()
                        depth = depth / norm
                        depth = colormap(depth.cpu().numpy()[0], cmap='turbo')
                        tb_writer.add_images(config['name'] + "_view_{}/depth".format(viewpoint.image_name), depth[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)

                        try:
                            rend_alpha = render_pkg['rend_alpha']
                            rend_normal = render_pkg["rend_normal"] * 0.5 + 0.5
                            surf_normal = render_pkg["surf_normal"] * 0.5 + 0.5
                            tb_writer.add_images(config['name'] + "_view_{}/rend_normal".format(viewpoint.image_name), rend_normal[None], global_step=iteration)
                            tb_writer.add_images(config['name'] + "_view_{}/surf_normal".format(viewpoint.image_name), surf_normal[None], global_step=iteration)
                            tb_writer.add_images(config['name'] + "_view_{}/rend_alpha".format(viewpoint.image_name), rend_alpha[None], global_step=iteration)

                            rend_dist = render_pkg["rend_dist"]
                            rend_dist = colormap(rend_dist.cpu().numpy()[0])
                            tb_writer.add_images(config['name'] + "_view_{}/rend_dist".format(viewpoint.image_name), rend_dist[None], global_step=iteration)
                        except:
                            pass

                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)

                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()

                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        torch.cuda.empty_cache()

def save_mask_visualization(mask, gt_image, iteration, output_dir, name="background_mask"):
    """
    保存掩码可视化图像，用于调试背景/前景分割
    
    Args:
        mask: 掩码张量 [H, W]
        gt_image: 原始图像 [3, H, W]
        iteration: 当前迭代次数
        output_dir: 输出目录
        name: 图像名称前缀
    """
    # 创建输出目录
    viz_dir = os.path.join(output_dir, "mask_viz")
    os.makedirs(viz_dir, exist_ok=True)
    
    # 准备数据
    mask_np = mask.detach().cpu().numpy()
    image_np = gt_image.permute(1, 2, 0).detach().cpu().numpy()
    
    # 创建图像
    plt.figure(figsize=(12, 5))
    
    # 原始图像
    plt.subplot(1, 3, 1)
    plt.imshow(image_np)
    plt.title("原始图像")
    plt.axis("off")
    
    # 掩码图像
    plt.subplot(1, 3, 2)
    plt.imshow(mask_np, cmap='gray')
    plt.title(f"{name.replace('_', ' ').title()}")
    plt.axis("off")
    
    # 叠加显示
    plt.subplot(1, 3, 3)
    overlay = np.copy(image_np)
    mask_rgb = np.stack([mask_np] * 3, axis=2)  # 转为3通道
    # 在掩码区域添加蓝色半透明覆盖
    overlay = overlay * (1 - mask_rgb * 0.7) + mask_rgb * np.array([0, 0, 0.8]) * 0.7
    plt.imshow(overlay)
    plt.title("叠加显示")
    plt.axis("off")
    
    # 保存图像
    plt.tight_layout()
    output_path = os.path.join(viz_dir, f"{name}_{iteration:06d}.png")
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    return output_path

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)
    
    
    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint)

    # All done
    print("\nTraining complete.")
