import os
import torch
from random import randint
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
from tqdm import tqdm
from utils.image_utils import render_net_image
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams, TrainingParams
import numpy as np

# 引入工具函数
from utils.tensorboard_utils import training_report
from utils.training_utils import prepare_output_and_logger, update_learning_schedules, perform_densification
from utils.training_utils import compute_viewpoint_loss, log_training_progress
from utils.loss_utils import l1_loss
from utils.loss_functions import (
    calculate_color_loss, calculate_normal_loss, calculate_convergence_loss,
    calculate_background_opacity_loss, get_lambda_normal, get_lambda_dist, get_lambda_convergence
)

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


def training(params):
    """
    主训练函数
    
    Args:
        params: 包含所有参数的对象
    """
    # 从params对象中提取需要的参数
    opt = params.optimization
    pipe = params.pipeline
    dataset = params.model
    testing_iterations = params.test_iterations
    saving_iterations = params.save_iterations
    checkpoint_iterations = params.checkpoint_iterations
    checkpoint = params.start_checkpoint
    
    first_iter = 0
    
    tb_writer = prepare_output_and_logger(params)
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

        # 更新学习率和SH级别
        update_learning_schedules(gaussians, iteration, opt)

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

        # 计算颜色损失
        color_loss, Ll1_report = calculate_color_loss(image, gt_image, mask, opt)

        # 获取损失权重
        lambda_normal = get_lambda_normal(iteration, opt)
        lambda_dist = get_lambda_dist(iteration, opt)
        lambda_conv = get_lambda_convergence(iteration, opt)
        lambda_bg_opacity = opt.lambda_bg_opacity

        # 获取渲染结果
        rend_dist = render_pkg["rend_dist"]
        rend_normal = render_pkg['rend_normal']
        surf_normal = render_pkg['surf_normal']
        
        # 计算正则化损失
        normal_loss = calculate_normal_loss(rend_normal, surf_normal, gt_image, lambda_normal, opt)
        dist_loss = lambda_dist * rend_dist.mean()
        
        # 计算深度收敛损失
        if lambda_conv > 0:
            surf_depth = render_pkg['surf_depth']
            render_alpha = render_pkg['rend_alpha']
            convergence_loss = calculate_convergence_loss(surf_depth, render_alpha, lambda_conv)
        else:
            convergence_loss = torch.tensor(0.0, device='cuda')

        # 计算背景透明度损失
        if 'rend_alpha' in render_pkg:
            render_alpha = render_pkg['rend_alpha']
            background_opacity_loss = calculate_background_opacity_loss(render_alpha, mask, lambda_bg_opacity)
        else:
            background_opacity_loss = torch.tensor(0.0, device='cuda')
          
        # 总损失
        total_loss = color_loss + dist_loss + normal_loss + convergence_loss + lambda_bg_opacity * background_opacity_loss
        
        # 记录当前视角的loss
        viewpoint_name = viewpoint_cam.image_name
        current_loss = total_loss.item()
        
        # 更新视角损失统计
        viewpoint_losses, all_losses, dynamic_threshold = compute_viewpoint_loss(
            viewpoint_losses, current_loss, viewpoint_name, iteration,
            all_losses, dynamic_threshold, threshold_update_interval, threshold_multiplier
        )
        
        total_loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * color_loss.item() + 0.6 * ema_loss_for_log
            ema_dist_for_log = 0.4 * dist_loss.item() + 0.6 * ema_dist_for_log
            ema_normal_for_log = 0.4 * normal_loss.item() + 0.6 * ema_normal_for_log
            ema_conv_for_log = 0.4 * convergence_loss.item() + 0.6 * ema_conv_for_log
            # 添加背景损失 EMA
            ema_bg_opacity_for_log = 0.4 * background_opacity_loss.item() + 0.6 * getattr(locals(), 'ema_bg_opacity_for_log', 0.0)

            if iteration % 10 == 0:
                loss_dict = {
                    "Loss": f"{ema_loss_for_log:.{5}f}"
                }
               
                progress_bar.set_postfix(loss_dict)
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # 记录训练进度
            log_training_progress(
                tb_writer, iteration, ema_loss_for_log, ema_dist_for_log,
                ema_normal_for_log, ema_conv_for_log, ema_bg_opacity_for_log,
                dynamic_threshold, lambda_bg_opacity, 
                viewpoint_name, current_loss, viewpoint_losses
            )
            
            # 使用tensorboard_utils中的函数记录训练报告
            training_report(tb_writer, iteration, Ll1_report, total_loss, l1_loss, 
                          iter_start.elapsed_time(iter_end), testing_iterations, 
                          scene, render, (pipe, background))
                          
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
                
            # 执行稠密化操作
            perform_densification(
                gaussians, iteration, opt, visibility_filter, radii,
                viewspace_point_tensor, scene.cameras_extent,
                white_background=dataset.white_background
            )

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

def prepare_output_and_logger(params):    
    """准备输出目录和日志记录器"""
    # 获取模型参数
    model_params = params.model
    
    if not model_params.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        model_params.model_path = os.path.join("./output/", unique_str[0:10])
        
    # 设置输出文件夹
    print("输出文件夹: {}".format(model_params.model_path))
    os.makedirs(model_params.model_path, exist_ok = True)
    
    # 保存配置参数
    with open(os.path.join(model_params.model_path, "cfg_args"), 'w') as cfg_log_f:
        # 将整个params对象序列化为Namespace并保存
        cfg_log_f.write(str(Namespace(**vars(params))))

    # 创建Tensorboard写入器
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(model_params.model_path)
    else:
        print("Tensorboard不可用: 不记录进度")
    return tb_writer

if __name__ == "__main__":
    # 设置命令行参数解析器
    parser = ArgumentParser(description="Training script parameters")
    
    # 使用统一的参数管理类
    params_manager = TrainingParams(parser)
    
    # 解析参数
    args = parser.parse_args(sys.argv[1:])
    params = params_manager.extract(args)
    
    print("正在优化 " + params.model.model_path)
    
    # 初始化系统状态(RNG)
    safe_state(params.quiet)

    # 启动GUI服务器，配置并运行训练
    network_gui.init(params.ip, params.port)
    torch.autograd.set_detect_anomaly(params.detect_anomaly)
    training(params)

    # 完成
    print("\n训练完成。")
