import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim, compute_color_loss, edge_aware_normal_loss, depth_convergence_loss
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr, render_net_image, colormap
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import numpy as np
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

# 导入ply可视化工具
from utils.tensorboard_utils import add_ply_to_tensorboard

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

        # 检查图像是否完全透明
        is_fully_transparent = False
        # 优先检查gt_alpha_mask是否存在且是否全为0
        if hasattr(viewpoint_cam, 'gt_alpha_mask') and viewpoint_cam.gt_alpha_mask is not None:
            # 假设 gt_alpha_mask 是一个张量，检查其最大值
            # 使用一个小的阈值以防浮点数精度问题
            if torch.max(viewpoint_cam.gt_alpha_mask) < 1e-6:
                 is_fully_transparent = True
        # 如果没有 alpha 掩码，检查原始图像是否有 alpha 通道 (RGBA)
        elif viewpoint_cam.original_image.shape[0] == 4:
             # 提取 alpha 通道 (索引为 3)
             alpha_channel = viewpoint_cam.original_image[3, :, :]
             # 检查 alpha 通道的最大值
             if torch.max(alpha_channel) < 1e-6: # 假设 alpha 范围是 0-1
                 is_fully_transparent = True

        # 如果图像完全透明，则打印信息并跳过此迭代
        if is_fully_transparent:
            print(f"Skipping fully transparent image: {viewpoint_cam.image_name} (Iteration {iteration})")
            # 如果弹出此相机后堆栈为空，则重新填充
            # （注意：现有逻辑已在每次迭代开始时检查并填充空堆栈）
            continue # 跳到下一次迭代

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
        if iteration <= 3000:
            lambda_normal = 0.0
        elif iteration <= 5000:
            lambda_normal = opt.lambda_normal * (iteration - 3000) / 2000  # 线性增加
        else:
            lambda_normal = opt.lambda_normal
        lambda_dist = opt.lambda_dist if iteration > 3000 else 0.0
        # 深度收敛损失权重设置
        lambda_conv = opt.lambda_depth_convergence if hasattr(opt, 'use_depth_convergence') and opt.use_depth_convergence and iteration > opt.conv_start_iter else 0.0

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

        # 总损失
        total_loss = loss + dist_loss + normal_loss + convergence_loss
        
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

            if iteration % 10 == 0:
                loss_dict = {
                    "Loss": f"{ema_loss_for_log:.{5}f}",
                    "distort": f"{ema_dist_for_log:.{5}f}",
                    "normal": f"{ema_normal_for_log:.{5}f}",
                    "conv": f"{ema_conv_for_log:.{5}f}",
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
                
                # 在TensorBoard中可视化保存的PLY文件
                if tb_writer is not None:
                    # 构建PLY文件路径
                    ply_path = os.path.join(dataset.model_path, "point_cloud", f"iteration_{iteration}", "point_cloud.ply")
                    # 将PLY文件添加到TensorBoard
                    add_ply_to_tensorboard(tb_writer, f"点云/迭代_{iteration}", ply_path, global_step=iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
                
                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    # 计算梯度
                    grads = gaussians.xyz_gradient_accum / gaussians.denom
                    grads[grads.isnan()] = 0.0
                    
                    # 执行稠密化操作
                    gaussians.densify_and_clone(grads, opt.densify_grad_threshold, scene.cameras_extent)
                    gaussians.densify_and_split(grads, opt.densify_grad_threshold, scene.cameras_extent)

                    # 裁剪低不透明度的点
                    opacity_mask = (gaussians.get_opacity < opt.opacity_cull).squeeze()
                    gaussians.prune_points(opacity_mask)
                
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
                except Exception:
                    # 移除 'as e' 和注释掉的 'raise e'
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
    # 每1000次迭代保存一次点云并在TensorBoard中可视化
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[i for i in range(1000, 30001, 1000)])
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
