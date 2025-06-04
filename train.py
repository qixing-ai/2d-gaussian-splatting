import os
import torch
from utils.loss_utils import compute_training_losses
from utils.training_utils import TrainingStateManager, DynamicPruningManager, handle_densification_and_pruning, get_random_viewpoint, log_training_metrics, evaluate_and_log_validation
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, estimate_scene_radius
import uuid
from utils.image_utils import render_net_image
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
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

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    training_state = TrainingStateManager(first_iter, opt.iterations)
    pruning_manager = DynamicPruningManager(opt.prune_ratio)
    pruning_manager.last_point_count = len(gaussians.get_xyz)
    
    scene_radius = estimate_scene_radius(scene.getTrainCameras())
    
    viewpoint_stack = None
    first_iter += 1
    
    for iteration in range(first_iter, opt.iterations + 1):        
        iter_start.record()
        gaussians.update_learning_rate(iteration)

        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        viewpoint_cam, viewpoint_stack = get_random_viewpoint(viewpoint_stack, scene)
        
        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        
        gt_image = viewpoint_cam.original_image.cuda()
        loss_dict = compute_training_losses(
            render_pkg, gt_image, viewpoint_cam, opt, iteration, scene_radius
        )
        
        loss_dict['total_loss'].backward()
        iter_end.record()

        with torch.no_grad():
            # 确保CUDA操作完成
            torch.cuda.synchronize()
            
            training_state.update_progress_bar(iteration, gaussians, loss_dict)
            
            if iteration == opt.iterations:
                training_state.close_progress_bar()

            log_training_metrics(tb_writer, iteration, loss_dict, 
                               iter_start.elapsed_time(iter_end), len(gaussians.get_xyz), pruning_manager.current_prune_ratio)
            evaluate_and_log_validation(tb_writer, iteration, testing_iterations, scene, render, (pipe, background))
            
            if iteration in saving_iterations:
                scene.save(iteration)

        handle_densification_and_pruning(gaussians, opt, iteration, viewspace_point_tensor, visibility_filter, 
                                        radii, scene, pipe, background, pruning_manager, dataset.white_background)

        if iteration < opt.iterations:
            gaussians.optimizer.step()
            gaussians.optimizer.zero_grad(set_to_none=True)

        if iteration in checkpoint_iterations:
            print(f"\n[ITER {iteration}] 保存检查点")
            torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

        handle_network_gui(gaussians, dataset, pipe, background, loss_dict['total_loss'].item(), iteration, opt)

def handle_network_gui(gaussians, dataset, pipe, background, current_loss, iteration, opt):
    """处理网络GUI"""
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
                    "loss": current_loss
                }
                network_gui.send(net_image_bytes, dataset.source_path, metrics_dict)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    print("输出目录: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard不可用: 不记录进度")
    return tb_writer

if __name__ == "__main__":
    parser = ArgumentParser(description="训练脚本参数")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7000, 15000 ,25000, 30000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7000, 15000 ,25000, 30000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    

    safe_state(args.quiet)

    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint)

    print("\n训练完成。")
