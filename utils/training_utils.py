import os
import torch
import uuid
from torch.utils.tensorboard import SummaryWriter
from argparse import Namespace

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
    try:
        tb_writer = SummaryWriter(model_params.model_path)
    except ImportError:
        print("Tensorboard不可用: 不记录进度")
    return tb_writer

def update_learning_schedules(gaussians, iteration, opt):
    """更新学习率和SH度"""
    # 更新学习率
    gaussians.update_learning_rate(iteration)
    
    # Every 1000 iterations we increase the levels of SH up to a maximum degree
    if iteration % 1000 == 0:
        gaussians.oneupSHdegree()

def perform_densification(gaussians, iteration, opt, visibility_filter, radii, 
                          viewspace_point_tensor, scene_cameras_extent, white_background=False):
    """执行点云的稠密化和裁剪操作"""
    # 只在指定迭代次数内进行稠密化
    if iteration < opt.densify_until_iter:
        # 更新最大2D半径
        gaussians.max_radii2D[visibility_filter] = torch.max(
            gaussians.max_radii2D[visibility_filter], 
            radii[visibility_filter]
        )
        
        # 添加稠密化统计数据
        gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
        
        # 在指定迭代次数后执行稠密化
        if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
            # 计算梯度
            grads = gaussians.xyz_gradient_accum / gaussians.denom
            grads[grads.isnan()] = 0.0
            
            # 执行稠密化操作
            gaussians.densify_and_clone(grads, opt.densify_grad_threshold, scene_cameras_extent)
            gaussians.densify_and_split(grads, opt.densify_grad_threshold, scene_cameras_extent)

            # 裁剪低不透明度的点
            if iteration > opt.cull_from_iter:
                opacity_mask = (gaussians.get_opacity < opt.opacity_cull).squeeze()
                gaussians.prune_points(opacity_mask)
        
        # 重置不透明度
        if iteration % opt.opacity_reset_interval == 0 or (
            white_background and iteration == opt.densify_from_iter):
            gaussians.reset_opacity()

def compute_viewpoint_loss(viewpoint_losses, current_loss, viewpoint_name, iteration,
                         all_losses, dynamic_threshold, threshold_update_interval, threshold_multiplier):
    """计算并更新视角损失统计"""
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
    
    return viewpoint_losses, all_losses, dynamic_threshold

def log_training_progress(tb_writer, iteration, ema_loss_for_log, ema_dist_for_log, 
                         ema_normal_for_log, ema_conv_for_log, ema_bg_opacity_for_log, 
                         dynamic_threshold, lambda_bg_opacity, 
                         viewpoint_name, current_loss, viewpoint_losses):
    """记录训练进度到TensorBoard"""
    if tb_writer is not None:
        tb_writer.add_scalar('train_loss_patches/dist_loss', ema_dist_for_log, iteration)
        tb_writer.add_scalar('train_loss_patches/normal_loss', ema_normal_for_log, iteration)
        tb_writer.add_scalar('train_loss_patches/conv_loss', ema_conv_for_log, iteration)
        tb_writer.add_scalar('train_loss_patches/dynamic_threshold', dynamic_threshold, iteration)
        
        # 记录背景透明度损失到 TensorBoard
        if lambda_bg_opacity > 0:
             tb_writer.add_scalar('train_loss_patches/bg_opacity_loss', ema_bg_opacity_for_log, iteration)

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