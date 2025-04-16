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
from utils.image_utils import render_net_image
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import numpy as np
# 从 utils.tensorboard_utils 导入函数
from utils.tensorboard_utils import training_report

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

# 移除 ply 可视化工具的导入（已被注释）

# 移除 log_tb_image 函数定义
# def log_tb_image(tb_writer, tag_prefix, image_name, image_tensor, global_step, cmap=None, normalize=True):
#     ...

# 移除 training_report 函数定义
# @torch.no_grad()
# def training_report(tb_writer, iteration, Ll1, loss, l1_loss_func, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
#     ...

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
    ema_bg_opacity_for_log = 0.0 # 用于记录背景透明度损失
    ema_gt_normal_for_log = 0.0 # 用于记录真值法线损失
    
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
        alpha_data = None
        # 优先检查 gt_alpha_mask
        if hasattr(viewpoint_cam, 'gt_alpha_mask') and viewpoint_cam.gt_alpha_mask is not None:
            alpha_data = viewpoint_cam.gt_alpha_mask
        # 否则，如果原始图像有 alpha 通道，则检查它
        elif viewpoint_cam.original_image.shape[0] == 4:
            alpha_data = viewpoint_cam.original_image[3, :, :] # Alpha 通道

        # 检查 alpha 数据是否全为零 (或接近零)
        is_fully_transparent = (alpha_data is not None) and (torch.max(alpha_data) < 1e-6)

        # 如果图像完全透明，则跳过此迭代
        if is_fully_transparent:
            # print(f"Skipping fully transparent image: {viewpoint_cam.image_name}") # 可以取消注释以进行调试
            continue # 跳到下一次迭代

        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        
        gt_image = viewpoint_cam.original_image.cuda()
        
        # Extract mask
        mask = None
        if hasattr(viewpoint_cam, 'gt_alpha_mask') and viewpoint_cam.gt_alpha_mask is not None:
            mask = viewpoint_cam.gt_alpha_mask.cuda().unsqueeze(0) # [1, H, W]
            # Ensure gt_image is 3 channels if mask exists separately
            if gt_image.shape[0] == 4:
                 gt_image = gt_image[:3, :, :]
        elif gt_image.shape[0] == 4:
            mask = gt_image[3:4, :, :].cuda() # Extract alpha channel as mask [1, H, W]
            gt_image = gt_image[:3, :, :] # Use only RGB channels for ground truth color
            mask = (mask > 0.5).float() # Threshold alpha mask to be binary

        # Calculate unmasked L1 for reporting purposes
        Ll1_report = l1_loss(image, gt_image)

        # 颜色损失计算
        if mask is not None:
            # Apply mask to L1 calculation
            pixel_count = mask.sum()
            if pixel_count > 0:
                masked_l1 = (torch.abs(image - gt_image) * mask).sum() / (pixel_count * image.shape[0] + 1e-6) # Mean L1 over foreground pixels
            else:
                masked_l1 = torch.tensor(0.0, device=image.device) # Avoid division by zero if mask is empty

            # SSIM calculation (on unmasked images for now)
            if hasattr(opt, 'use_ms_ssim') and opt.use_ms_ssim:
                 # Need to ensure ms_ssim is available and imported if used.
                 # Assuming ssim function from loss_utils handles both cases based on internal logic or we'd need separate calls.
                 # For simplicity, using the standard ssim function here. Modify if ms_ssim is intended.
                 ssim_val = ssim(image, gt_image) # Calculate SSIM on full image
                 loss = (1.0 - opt.lambda_dssim) * masked_l1 + opt.lambda_dssim * (1.0 - ssim_val)
            else:
                 ssim_val = ssim(image, gt_image) # Calculate SSIM on full image
                 loss = (1.0 - opt.lambda_dssim) * masked_l1 + opt.lambda_dssim * (1.0 - ssim_val)
        else:
            # Original loss calculation if no mask is available
            Ll1 = Ll1_report # Use the already calculated unmasked L1
            if hasattr(opt, 'use_ms_ssim') and opt.use_ms_ssim:
                 loss = compute_color_loss(image, gt_image, lambda_dssim=opt.lambda_dssim, use_ms_ssim=True)
            else:
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
        # 背景透明度损失权重
        lambda_bg_opacity = opt.lambda_bg_opacity
        # 真值法线损失权重 (从 opt 获取，假设已添加)
        lambda_gt_normal = opt.lambda_gt_normal

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

        # 添加背景透明度损失 (仅当 mask 存在时)
        background_opacity_loss = torch.tensor(0.0, device='cuda')
        # 直接从 opt 获取权重，因为它应该总是存在 (有默认值)
        lambda_bg_opacity = opt.lambda_bg_opacity

        if mask is not None and lambda_bg_opacity > 0:
            render_alpha = render_pkg['rend_alpha'] # 获取渲染的 alpha
            background_mask = (1 - mask)
            num_background_pixels = background_mask.sum()
            if num_background_pixels > 0:
                # 计算背景区域 alpha 的 L1 损失均值
                background_opacity_loss = (torch.abs(render_alpha * background_mask)).sum() / num_background_pixels
          

        # 计算真值法线损失
        gt_normal_loss = torch.tensor(0.0, device='cuda')
        if lambda_gt_normal > 0: # Check if weight is positive first
            # <<< 删除下面的调试打印块 >>>
            # print(f"[Debug Iter {iteration}] Checking gt_normal for {viewpoint_cam.image_name}:")
            # print(f"  - hasattr(viewpoint_cam, 'gt_normal'): {hasattr(viewpoint_cam, 'gt_normal')}")
            # if hasattr(viewpoint_cam, 'gt_normal'):
            #     print(f"  - viewpoint_cam.gt_normal is None: {viewpoint_cam.gt_normal is None}")
            #     if viewpoint_cam.gt_normal is not None:
            #         # 打印数据类型和形状以获取更多信息
            #         print(f"  - viewpoint_cam.gt_normal dtype: {viewpoint_cam.gt_normal.dtype}")
            #         print(f"  - viewpoint_cam.gt_normal shape: {viewpoint_cam.gt_normal.shape}")
            # <<< 调试打印结束 >>>

            if hasattr(viewpoint_cam, 'gt_normal') and viewpoint_cam.gt_normal is not None:
                gt_normal_map = viewpoint_cam.gt_normal.cuda() # [3, H, W], 假设范围是 [-1, 1]
                if mask is not None:
                    # 应用 mask
                    pixel_count = mask.sum()
                    if pixel_count > 0:
                        # 计算前景区域的余弦相似度损失 (1 - cos(theta))
                        cos_sim = torch.sum(surf_normal * gt_normal_map, dim=0, keepdim=True) # [1, H, W]
                        # 应用 mask 并计算平均损失
                        gt_normal_loss = lambda_gt_normal * (mask * (1.0 - cos_sim)).sum() / pixel_count

                else:
                     # 如果没有 mask，计算整个图像的损失
                     gt_normal_loss = lambda_gt_normal * (1.0 - torch.sum(surf_normal * gt_normal_map, dim=0)).mean()

        # 总损失
        total_loss = loss + dist_loss + normal_loss + convergence_loss + lambda_bg_opacity * background_opacity_loss + gt_normal_loss
        
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
            # 添加背景损失 EMA
            ema_bg_opacity_for_log = 0.4 * background_opacity_loss.item() + 0.6 * getattr(locals(), 'ema_bg_opacity_for_log', 0.0)
            # 添加真值法线损失 EMA
            ema_gt_normal_for_log = 0.4 * gt_normal_loss.item() + 0.6 * ema_gt_normal_for_log

            if iteration % 10 == 0:
                loss_dict = {
                    "Loss": f"{ema_loss_for_log:.{5}f}"
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
                # 记录背景透明度损失到 TensorBoard
                if lambda_bg_opacity > 0:
                     tb_writer.add_scalar('train_loss_patches/bg_opacity_loss', ema_bg_opacity_for_log, iteration)
                # 记录真值法线损失到 TensorBoard
                if lambda_gt_normal > 0:
                     tb_writer.add_scalar('train_loss_patches/gt_normal_loss', ema_gt_normal_for_log, iteration)
                # 记录真值法线损失权重到 TensorBoard
                tb_writer.add_scalar('hyperparameters/lambda_gt_normal', lambda_gt_normal, iteration)

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

            # 将用于报告的 L1 损失值传递给 training_report
            # 注意：第三个参数 Ll1 现在是 Ll1_report
            training_report(tb_writer, iteration, Ll1_report, total_loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
                
            # Densification
            if iteration < opt.densify_until_iter:
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
                
                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    # 计算梯度
                    grads = gaussians.xyz_gradient_accum / gaussians.denom
                    grads[grads.isnan()] = 0.0
                    
                    # 执行稠密化操作
                    gaussians.densify_and_clone(grads, opt.densify_grad_threshold, scene.cameras_extent)
                    gaussians.densify_and_split(grads, opt.densify_grad_threshold, scene.cameras_extent)

                    # 裁剪低不透明度的点
                    if iteration > opt.cull_from_iter:
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

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 12_000])
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
