import torch
from tqdm import tqdm

class TrainingStateManager:
    """训练状态管理器，负责管理进度条、日志等"""
    
    def __init__(self, first_iter, total_iterations):
        self.progress_bar = tqdm(range(first_iter, total_iterations), desc="Training")
        
    def update_progress_bar(self, iteration, gaussians, loss_dict, update_interval=10):
        """更新进度条"""
        if iteration % update_interval == 0:
            progress_dict = {
                "Loss": f"{loss_dict['total_loss'].item():.{5}f}",
                "Points": f"{len(gaussians.get_xyz)}"
            }
            self.progress_bar.set_postfix(progress_dict)
            self.progress_bar.update(update_interval)
    
    def close_progress_bar(self):
        """关闭进度条"""
        self.progress_bar.close()

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
        gaussians.densify_and_prune(opt.densify_grad_threshold, opt.opacity_cull, scene.cameras_extent, None)
    
    # 基于贡献的修剪
    if iteration % opt.contribution_prune_interval == 0 and iteration < opt.prune_strategy_switch_iter:
        contribution = gaussians.compute_multi_view_contribution(
            scene.getTrainCameras(), 
            pipe, 
            background,
            gamma=opt.contribution_gamma
        )
        
        # 使用固定的修剪比例，不再动态调整
        gaussians.prune_low_contribution(contribution, prune_ratio=pruning_manager.current_prune_ratio)

def get_random_viewpoint(viewpoint_stack, scene):
    """获取随机视角"""
    from random import randint
    if not viewpoint_stack:
        viewpoint_stack = scene.getTrainCameras().copy()
        # 确保每个视角都有正确的ID
        for i, cam in enumerate(viewpoint_stack):
            cam.id = i
    viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
    return viewpoint_cam, viewpoint_stack 

def log_training_metrics(tb_writer, iteration, loss_dict, elapsed, total_points, current_prune_ratio=None):
    """记录训练指标到TensorBoard"""
    if not tb_writer:
        return
    
    # 训练指标组 - 所有重要的训练信息
    tb_writer.add_scalar('训练指标/总损失', loss_dict['total_loss'].item(), iteration)
    tb_writer.add_scalar('训练指标/重建损失', loss_dict['reconstruction_loss'].item(), iteration)
    tb_writer.add_scalar('训练指标/L1损失', loss_dict['l1_loss'].item(), iteration)
    tb_writer.add_scalar('训练指标/MS-SSIM损失', loss_dict['ms_ssim_loss'].item(), iteration)
    tb_writer.add_scalar('训练指标/法线损失', loss_dict['normal_loss'].item(), iteration)
    tb_writer.add_scalar('训练指标/Alpha损失', loss_dict['alpha_loss'].item(), iteration)
    

    # 深度校正损失（如果存在）
    if 'depth_convergence_loss' in loss_dict:
        tb_writer.add_scalar('训练指标/深度收敛损失', loss_dict['depth_convergence_loss'].item(), iteration)

    
    # 训练统计
    tb_writer.add_scalar('训练指标/点数量', total_points, iteration)
    tb_writer.add_scalar('训练指标/迭代时间(ms)', elapsed, iteration)
    
    # 损失比例分析 - 帮助理解各损失组件的贡献
    total_loss_val = loss_dict['total_loss'].item()
    if total_loss_val > 0:
        # 计算各损失占比（百分比形式）
        reconstruction_ratio = (loss_dict['reconstruction_loss'].item() / total_loss_val) * 100
        l1_ratio = (loss_dict['l1_loss'].item() / total_loss_val) * 100
        ms_ssim_ratio = (loss_dict['ms_ssim_loss'].item() / total_loss_val) * 100
        normal_ratio = (loss_dict['normal_loss'].item() / total_loss_val) * 100
        alpha_ratio = (loss_dict['alpha_loss'].item() / total_loss_val) * 100
        
        # 构建损失占比字典
        loss_ratios = {
            '重建损失占比(%)': reconstruction_ratio,
            'L1损失占比(%)': l1_ratio,
            'MS-SSIM损失占比(%)': ms_ssim_ratio,
            '自适应法线损失占比(%)': normal_ratio,
            'Alpha损失占比(%)': alpha_ratio
        }
        
        # 如果存在深度收敛损失，也添加进去
        if 'depth_convergence_loss' in loss_dict:
            depth_ratio = (loss_dict['depth_convergence_loss'].item() / total_loss_val) * 100
            loss_ratios['深度损失占比(%)'] = depth_ratio
        
        # 将所有损失占比记录到同一个图中
        tb_writer.add_scalars('损失占比分析', loss_ratios, iteration)

def evaluate_and_log_validation(tb_writer, iteration, testing_iterations, scene, renderFunc, renderArgs):
    """评估验证集并记录结果"""
    if iteration not in testing_iterations:
        return
        
    from utils.image_utils import psnr
    from utils.loss_utils import l1_loss
    import torch
    
    validation_configs = [
        {'name': 'test', 'cameras': scene.getTestCameras()}, 
        {'name': 'train', 'cameras': [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(0, 60, 3)]}
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
            
            # 记录视角展示（前20个视角）
            if tb_writer and idx < 20:
                log_visualization_results(tb_writer, render_pkg, image, gt_image, viewpoint, config['name'], iteration, idx)
            
            l1_test += l1_loss(image, gt_image).mean().double()
            psnr_test += psnr(image, gt_image).mean().double()

        # 计算平均指标
        psnr_test /= len(config['cameras'])
        l1_test /= len(config['cameras'])
        
        print(f"\n[ITER {iteration}] 评估 {config['name']}: L1 {l1_test} PSNR {psnr_test}")
        


def log_visualization_results(tb_writer, render_pkg, image, gt_image, viewpoint, config_name, iteration, view_idx):
    """记录可视化结果到TensorBoard"""
    from utils.general_utils import colormap
    
    # 视角展示组 - 只显示法线图
    view_name = f"{config_name}_视角{view_idx+1}"
    
    # 只记录法线可视化
    try:
        if 'rend_normal' in render_pkg:
            rend_normal = render_pkg["rend_normal"] * 0.5 + 0.5
            tb_writer.add_images(f'视角展示/{view_name}/渲染法线', rend_normal[None], global_step=iteration)
        
        if 'surf_normal' in render_pkg:
            surf_normal = render_pkg["surf_normal"] * 0.5 + 0.5
            tb_writer.add_images(f'视角展示/{view_name}/表面法线', surf_normal[None], global_step=iteration)
    except:
        pass 