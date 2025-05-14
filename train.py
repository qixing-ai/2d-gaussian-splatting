#
# 版权声明 (C) 2023, Inria
# GRAPHDECO 研究小组, https://team.inria.fr/graphdeco
# 保留所有权利
#
# 本软件免费用于非商业、研究和评估用途
# 遵循 LICENSE.md 文件中的条款
#
# 咨询请联系 george.drettakis@inria.fr
#

import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim, ms_ssim_loss, edge_aware_curvature_loss
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr, render_net_image
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)  # 准备输出目录和日志记录器
    gaussians = GaussianModel(dataset.sh_degree)  # 创建高斯模型
    scene = Scene(dataset, gaussians)  # 创建场景
    gaussians.training_setup(opt)  # 设置训练参数
    if checkpoint:  # 如果有检查点则恢复
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]  # 背景颜色
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")  # 背景张量

    iter_start = torch.cuda.Event(enable_timing = True)  # 记录迭代开始时间
    iter_end = torch.cuda.Event(enable_timing = True)  # 记录迭代结束时间

    viewpoint_stack = None  # 视角栈
    ema_loss_for_log = 0.0  # 指数移动平均损失
    ema_dist_for_log = 0.0  # 指数移动平均距离损失
    ema_normal_for_log = 0.0  # 指数移动平均法线损失
    ema_alpha_for_log = 0.0  # 指数移动平均alpha损失
    ema_edge_aware_for_log = 0.0  # 指数移动平均边缘感知损失

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="ing")  # 进度条
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):        
        iter_start.record()  # 记录迭代开始

        gaussians.update_learning_rate(iteration)  # 更新学习率

        if iteration % 1000 == 0:  # 每1000次迭代增加SH级别
            gaussians.oneupSHdegree()

        if not viewpoint_stack:  # 随机选择一个相机视角
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        
        render_pkg = render(viewpoint_cam, gaussians, pipe, background)  # 渲染
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        
        gt_image = viewpoint_cam.original_image.cuda()  # 获取真实图像
        Ll1 = l1_loss(image, gt_image)  # 计算L1损失
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * ms_ssim_loss(image, gt_image)  # 总损失
        
        lambda_normal = opt.lambda_normal if iteration > 7000 else 0.0  # 法线正则化权重
        lambda_dist = opt.lambda_dist if iteration > 3000 else 0.0  # 距离正则化权重
        lambda_alpha = opt.lambda_alpha if iteration > 100 else 0.0  # alpha正则化权重
        lambda_edge_aware = opt.lambda_edge_aware if iteration > 1000 else 0.0  # 边缘感知正则化权重

        rend_dist = render_pkg["rend_dist"]  # 渲染距离
        rend_normal  = render_pkg['rend_normal']  # 渲染法线
        surf_normal = render_pkg['surf_normal']  # 表面法线
        # 计算法线角度差异(弧度)
        cos_theta = (rend_normal * surf_normal).sum(dim=0)
        cos_theta = torch.clamp(cos_theta, -0.9999, 0.9999)  # 防止数值不稳定
        angle_error = torch.acos(cos_theta)[None]
        
        # 计算法线长度差异
        rend_norm = rend_normal.norm(dim=0)
        surf_norm = surf_normal.norm(dim=0)
        norm_error = torch.abs(rend_norm - surf_norm)[None]
        
        # 组合损失(角度差异 + 0.1*长度差异)
        normal_error = angle_error + 0.1 * norm_error
        normal_loss = lambda_normal * normal_error.mean()  # 法线损失
        dist_loss = lambda_dist * (rend_dist).mean()  # 距离损失
        
        edge_aware_loss = torch.tensor(0.0, device="cuda")  # 边缘感知损失
        if lambda_edge_aware > 0:
            edge_aware_loss = lambda_edge_aware * edge_aware_curvature_loss(
                image, 
                render_pkg["surf_depth"],
                mask=viewpoint_cam.gt_alpha_mask if hasattr(viewpoint_cam, 'gt_alpha_mask') else None
            )
        
        alpha_loss = torch.tensor(0.0, device="cuda")  # alpha掩码损失
        if hasattr(viewpoint_cam, 'gt_alpha_mask') and lambda_alpha > 0 and viewpoint_cam.gt_alpha_mask is not None:
            gt_alpha = viewpoint_cam.gt_alpha_mask
            rend_alpha = render_pkg['rend_alpha']
            bg_region = (1.0 - gt_alpha)  # 背景区域
            alpha_loss = lambda_alpha * (rend_alpha * bg_region).mean()  # 计算alpha损失

        total_loss = loss + dist_loss + normal_loss + alpha_loss + edge_aware_loss  # 总损失
        total_loss.backward()  # 反向传播

        iter_end.record()  # 记录迭代结束

        with torch.no_grad():
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log  # 更新指数移动平均损失
            ema_dist_for_log = 0.4 * dist_loss.item() + 0.6 * ema_dist_for_log  # 更新指数移动平均距离损失
            ema_normal_for_log = 0.4 * normal_loss.item() + 0.6 * ema_normal_for_log  # 更新指数移动平均法线损失
            ema_alpha_for_log = 0.4 * alpha_loss.item() + 0.6 * ema_alpha_for_log  # 更新指数移动平均alpha损失
            ema_edge_aware_for_log = 0.4 * edge_aware_loss.item() + 0.6 * ema_edge_aware_for_log  # 更新指数移动平均边缘感知损失

            if iteration % 10 == 0:  # 每10次迭代更新进度条
                loss_dict = {
                    "Loss": f"{ema_loss_for_log:.{5}f}",
                    "Points": f"{len(gaussians.get_xyz)}"
                }
                progress_bar.set_postfix(loss_dict)
                progress_bar.update(10)
            if iteration == opt.iterations:  # 训练结束时关闭进度条
                progress_bar.close()

            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background),
                          ema_dist_for_log, ema_normal_for_log, ema_alpha_for_log, ema_edge_aware_for_log, dataset)  # 训练报告
            if (iteration in saving_iterations):  # 保存高斯模型
                scene.save(iteration)

        if iteration < opt.densify_until_iter:  # 密度化和修剪
            gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
            gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

            if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:  # 密度化间隔
                size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                gaussians.densify_and_prune(opt.densify_grad_threshold, opt.opacity_cull, scene.cameras_extent, size_threshold)
            
            if iteration % opt.contribution_prune_interval == 0 and iteration > 3000:  # 基于贡献的修剪
                contribution = gaussians.compute_multi_view_contribution(
                    scene.getTrainCameras(), 
                    pipe, 
                    background,
                    gamma=opt.contribution_gamma
                )
                gaussians.prune_low_contribution(contribution, prune_ratio=opt.prune_ratio)
            
            if (opt.opacity_reset_interval > 0 and iteration % opt.opacity_reset_interval == 0) or (dataset.white_background and iteration == opt.densify_from_iter):
                gaussians.reset_opacity()  # 重置不透明度

            if iteration < opt.iterations:  # 优化器步骤
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):  # 保存检查点
                print("\n[ITER {}] 保存检查点".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

        with torch.no_grad():        
            if network_gui.conn == None:  # 网络GUI连接
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
                    }
                    network_gui.send(net_image_bytes, dataset.source_path, metrics_dict)
                    if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                        break
                except Exception as e:
                    network_gui.conn = None

def prepare_output_and_logger(args):    
    if not args.model_path:  # 设置输出路径
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    print("输出目录: {}".format(args.model_path))  # 创建输出目录
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))  # 保存配置参数

    tb_writer = None  # TensorBoard写入器
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard不可用: 不记录进度")
    return tb_writer

@torch.no_grad()
def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs,
                   ema_dist, ema_normal, ema_alpha, ema_edge_aware, config):
    if tb_writer:  # 记录训练报告到TensorBoard
        # 记录学习率
        tb_writer.add_scalar('learning_rate/position', scene.gaussians.optimizer.param_groups[0]['lr'], iteration)
        tb_writer.add_scalar('learning_rate/opacity', scene.gaussians.optimizer.param_groups[1]['lr'], iteration)
        tb_writer.add_scalar('learning_rate/scaling', scene.gaussians.optimizer.param_groups[2]['lr'], iteration)
        tb_writer.add_scalar('learning_rate/rotation', scene.gaussians.optimizer.param_groups[3]['lr'], iteration)
        tb_writer.add_scalar('learning_rate/features_dc', scene.gaussians.optimizer.param_groups[4]['lr'], iteration)
        tb_writer.add_scalar('learning_rate/features_rest', scene.gaussians.optimizer.param_groups[5]['lr'], iteration)
        
        tb_writer.add_scalar('train_loss_patches/reg_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/dist_loss', ema_dist, iteration)
        tb_writer.add_scalar('train_loss_patches/normal_loss', ema_normal, iteration)
        tb_writer.add_scalar('train_loss_patches/alpha_loss', ema_alpha, iteration)
        tb_writer.add_scalar('train_loss_patches/edge_aware_loss', ema_edge_aware, iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)
        tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)

        if iteration in testing_iterations:  # 在指定测试迭代时执行以下操作:
        # 2. 准备测试集和训练集样本配置
        # 3. 计算并记录L1和PSNR指标
        # 4. 可视化深度图、法线图等渲染结果
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
                    print("\n[ITER {}] 评估 {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                    if tb_writer:
                        tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                        tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)


    

if __name__ == "__main__":
    parser = ArgumentParser(description="训练脚本参数")  # 设置命令行参数解析器
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
    
    print("优化 " + args.model_path)

    safe_state(args.quiet)

    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint)

    print("\n训练完成。")
