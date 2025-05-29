import torch
from tqdm import tqdm

class TrainingStateManager:
    """训练状态管理器，负责管理EMA损失、进度条、日志等"""
    
    def __init__(self, first_iter, total_iterations):
        self.ema_loss_for_log = 0.0
        self.ema_normal_for_log = 0.0
        self.ema_alpha_for_log = 0.0
        self.progress_bar = tqdm(range(first_iter, total_iterations), desc="Training")
        
    def update_ema_losses(self, loss_dict):
        """更新指数移动平均损失"""
        self.ema_loss_for_log = 0.4 * loss_dict['reconstruction_loss'].item() + 0.6 * self.ema_loss_for_log
        self.ema_normal_for_log = 0.4 * loss_dict['normal_loss'].item() + 0.6 * self.ema_normal_for_log
        self.ema_alpha_for_log = 0.4 * loss_dict['alpha_loss'].item() + 0.6 * self.ema_alpha_for_log
    
    def update_progress_bar(self, iteration, gaussians, update_interval=10):
        """更新进度条"""
        if iteration % update_interval == 0:
            loss_dict = {
                "Loss": f"{self.ema_loss_for_log:.{5}f}",
                "Points": f"{len(gaussians.get_xyz)}"
            }
            self.progress_bar.set_postfix(loss_dict)
            self.progress_bar.update(update_interval)
    
    def close_progress_bar(self):
        """关闭进度条"""
        self.progress_bar.close()
    
    def get_ema_losses(self):
        """获取当前EMA损失值"""
        return {
            'ema_loss': self.ema_loss_for_log,
            'ema_normal': self.ema_normal_for_log,
            'ema_alpha': self.ema_alpha_for_log
        }

class DynamicPruningManager:
    """动态修剪管理器"""
    
    def __init__(self, initial_prune_ratio):
        self.current_prune_ratio = initial_prune_ratio
        self.last_point_count = 0
        self.point_count_history = []
        
        # 硬编码的合理默认值
        self.target_ratio_min = 0.8
        self.target_ratio_max = 1.2
        self.prune_ratio_min = 0.01
        self.prune_ratio_max = 0.15
        self.adjust_factor = 0.02
    
    def update_pruning_ratio(self, current_point_count, iteration, prune_interval):
        """动态调整修剪比例"""
        self.point_count_history.append(current_point_count)
        
        if self.last_point_count > 0:
            point_ratio = current_point_count / self.last_point_count
            
            if point_ratio > self.target_ratio_max:
                self.current_prune_ratio = min(self.current_prune_ratio + self.adjust_factor, self.prune_ratio_max)
            elif point_ratio < self.target_ratio_min:
                self.current_prune_ratio = max(self.current_prune_ratio - self.adjust_factor, self.prune_ratio_min)
            
            # 记录调整信息
            if iteration % (prune_interval * 10) == 0:
                print(f"\n[ITER {iteration}] 点数: {current_point_count}, 变化比例: {point_ratio:.3f}, 修剪比例: {self.current_prune_ratio:.4f}")
        
        self.last_point_count = current_point_count
        return self.current_prune_ratio

def handle_densification_and_pruning(gaussians, opt, iteration, viewspace_point_tensor, visibility_filter, radii, scene, pipe, background, pruning_manager, white_background=False):
    """处理密度化和修剪操作"""
    if iteration >= opt.densify_until_iter:
        return
    
    # 更新密度化统计
    gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
    gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
    
    # 标准密度化和修剪
    if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
        size_threshold = 20 if iteration > opt.opacity_reset_interval else None
        gaussians.densify_and_prune(opt.densify_grad_threshold, opt.opacity_cull, scene.cameras_extent, size_threshold)
    
    # 基于贡献的修剪
    if iteration % opt.contribution_prune_interval == 0 and iteration < opt.prune_strategy_switch_iter:
        contribution = gaussians.compute_multi_view_contribution(
            scene.getTrainCameras(), 
            pipe, 
            background,
            gamma=opt.contribution_gamma
        )
        
        current_prune_ratio = pruning_manager.current_prune_ratio
        if iteration >= opt.prune_strategy_switch_iter:
            current_point_count = len(gaussians.get_xyz)
            current_prune_ratio = pruning_manager.update_pruning_ratio(
                current_point_count, iteration, opt.contribution_prune_interval
            )
        
        gaussians.prune_low_contribution(contribution, prune_ratio=current_prune_ratio)
    
    # 重置不透明度
    if (opt.opacity_reset_interval > 0 and iteration % opt.opacity_reset_interval == 0) or \
       (white_background and iteration == opt.densify_from_iter):
        gaussians.reset_opacity()

def get_random_viewpoint(viewpoint_stack, scene):
    """获取随机视角"""
    from random import randint
    if not viewpoint_stack:
        viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
    return viewpoint_cam, viewpoint_stack 

def log_training_metrics(tb_writer, iteration, loss_dict, ema_losses, elapsed, total_points, current_prune_ratio=None):
    """记录训练指标到TensorBoard"""
    if not tb_writer:
        return
    
    # 主要损失组件
    tb_writer.add_scalar('losses/total_loss', loss_dict['total_loss'].item(), iteration)
    tb_writer.add_scalar('losses/reconstruction_loss', loss_dict['reconstruction_loss'].item(), iteration)
    tb_writer.add_scalar('losses/normal_loss', loss_dict['normal_loss'].item(), iteration)
    tb_writer.add_scalar('losses/alpha_loss', loss_dict['alpha_loss'].item(), iteration)
    
    # 基础损失组件
    tb_writer.add_scalar('loss_components/l1_loss', loss_dict['l1_loss'].item(), iteration)
    tb_writer.add_scalar('loss_components/ms_ssim_loss', loss_dict['ms_ssim_loss'].item(), iteration)
    
    # Lambda参数（权重）
    tb_writer.add_scalar('loss_weights/lambda_normal', loss_dict['lambda_normal'], iteration)
    tb_writer.add_scalar('loss_weights/lambda_alpha', loss_dict['lambda_alpha'], iteration)
    
    # EMA平滑损失（用于趋势观察）
    tb_writer.add_scalar('ema_losses/ema_reconstruction', ema_losses['ema_loss'], iteration)
    tb_writer.add_scalar('ema_losses/ema_normal', ema_losses['ema_normal'], iteration)
    tb_writer.add_scalar('ema_losses/ema_alpha', ema_losses['ema_alpha'], iteration)
    
    # 训练统计信息
    tb_writer.add_scalar('training_stats/iter_time', elapsed, iteration)
    tb_writer.add_scalar('training_stats/total_points', total_points, iteration)
    
    # 动态修剪信息
    if current_prune_ratio is not None:
        tb_writer.add_scalar('training_stats/prune_ratio', current_prune_ratio, iteration)
    
    # 保持向后兼容的记录（如果有其他代码依赖这些名称）
    tb_writer.add_scalar('train_loss_patches/total_loss', loss_dict['total_loss'].item(), iteration)
    tb_writer.add_scalar('train_loss_patches/l1_loss', loss_dict['l1_loss'].item(), iteration)
    tb_writer.add_scalar('train_loss_patches/normal_loss', loss_dict['normal_loss'].item(), iteration)
    tb_writer.add_scalar('train_loss_patches/alpha_loss', loss_dict['alpha_loss'].item(), iteration)

def evaluate_and_log_validation(tb_writer, iteration, testing_iterations, scene, renderFunc, renderArgs):
    """评估验证集并记录结果"""
    if iteration not in testing_iterations:
        return
        
    from utils.image_utils import psnr
    from utils.loss_utils import l1_loss
    import torch
    
    validation_configs = [
        {'name': 'test', 'cameras': scene.getTestCameras()}, 
        {'name': 'train', 'cameras': [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]}
    ]

    for config in validation_configs:
        if not config['cameras'] or len(config['cameras']) == 0:
            continue
            
        l1_test = 0.0
        psnr_test = 0.0
        
        for idx, viewpoint in enumerate(config['cameras']):
            render_pkg = renderFunc(viewpoint, scene.gaussians, *renderArgs)
            image = torch.clamp(render_pkg["render"], 0.0, 1.0).to("cuda")
            gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
            
            # 记录可视化结果（前5个视角）
            if tb_writer and idx < 5:
                log_visualization_results(tb_writer, render_pkg, image, gt_image, viewpoint, config['name'], iteration, testing_iterations)
            
            l1_test += l1_loss(image, gt_image).mean().double()
            psnr_test += psnr(image, gt_image).mean().double()

        # 计算平均指标
        psnr_test /= len(config['cameras'])
        l1_test /= len(config['cameras'])
        
        print(f"\n[ITER {iteration}] 评估 {config['name']}: L1 {l1_test} PSNR {psnr_test}")
        
        if tb_writer:
            tb_writer.add_scalar(f"{config['name']}/loss_viewpoint - l1_loss", l1_test, iteration)
            tb_writer.add_scalar(f"{config['name']}/loss_viewpoint - psnr", psnr_test, iteration)

def log_visualization_results(tb_writer, render_pkg, image, gt_image, viewpoint, config_name, iteration, testing_iterations):
    """记录可视化结果到TensorBoard"""
    from utils.general_utils import colormap
    
    # 深度图可视化
    depth = render_pkg["surf_depth"]
    norm = depth.max()
    depth = depth / norm
    depth = colormap(depth.cpu().numpy()[0], cmap='turbo')
    tb_writer.add_images(f"{config_name}_view_{viewpoint.image_name}/depth", depth[None], global_step=iteration)
    tb_writer.add_images(f"{config_name}_view_{viewpoint.image_name}/render", image[None], global_step=iteration)

    try:
        # 法线和alpha可视化
        rend_alpha = render_pkg['rend_alpha']
        rend_normal = render_pkg["rend_normal"] * 0.5 + 0.5
        surf_normal = render_pkg["surf_normal"] * 0.5 + 0.5
        tb_writer.add_images(f"{config_name}_view_{viewpoint.image_name}/rend_normal", rend_normal[None], global_step=iteration)
        tb_writer.add_images(f"{config_name}_view_{viewpoint.image_name}/surf_normal", surf_normal[None], global_step=iteration)
        tb_writer.add_images(f"{config_name}_view_{viewpoint.image_name}/rend_alpha", rend_alpha[None], global_step=iteration)
    except:
        pass

    # 第一次测试时记录真实图像
    if iteration == testing_iterations[0]:
        tb_writer.add_images(f"{config_name}_view_{viewpoint.image_name}/ground_truth", gt_image[None], global_step=iteration) 