import torch  # 导入PyTorch库
from scene import Scene  # 从scene模块导入Scene类
import os  # 导入操作系统接口模块
from gaussian_renderer import render  # 从gaussian_renderer模块导入render函数
from argparse import ArgumentParser  # 导入命令行参数解析器
from arguments import ModelParams, PipelineParams, get_combined_args  # 导入参数相关类和方法
from gaussian_renderer import GaussianModel  # 从gaussian_renderer导入高斯模型类
from utils.mesh_utils import GaussianExtractor, to_cam_open3d, post_process_mesh  # 导入网格处理工具
from utils.render_utils import generate_path, create_videos  # 导入渲染路径和视频创建工具

import open3d as o3d  # 导入Open3D点云处理库

if __name__ == "__main__":
    parser = ArgumentParser(description="Testing script parameters")  # 创建参数解析器
    model = ModelParams(parser, sentinel=True)  # 初始化模型参数
    pipeline = PipelineParams(parser)  # 初始化管线参数
    parser.add_argument("--iteration", default=-1, type=int)  # 添加迭代次数参数
    parser.add_argument("--skip_train", action="store_true")  # 添加跳过训练参数
    parser.add_argument("--skip_test", action="store_true")  # 添加跳过测试参数
    parser.add_argument("--skip_mesh", action="store_true")  # 添加跳过网格生成参数
    parser.add_argument("--quiet", action="store_true")  # 添加静默模式参数
    parser.add_argument("--render_path", action="store_true")  # 添加渲染路径参数
    parser.add_argument("--voxel_size", default=-1.0, type=float)  # 添加体素大小参数
    parser.add_argument("--depth_trunc", default=-1.0, type=float)  # 添加深度截断参数
    parser.add_argument("--sdf_trunc", default=-1.0, type=float)  # 添加SDF截断参数
    parser.add_argument("--num_cluster", default=50, type=int)  # 添加聚类数量参数
    parser.add_argument("--unbounded", action="store_true")  # 添加无边界模式参数
    parser.add_argument("--mesh_res", default=1024, type=int)  # 添加网格分辨率参数
    args = get_combined_args(parser)  # 获取合并后的参数
    print("Rendering " + args.model_path)  # 打印渲染模型路径


    dataset, iteration, pipe = model.extract(args), args.iteration, pipeline.extract(args)  # 提取数据集、迭代次数和管线参数
    gaussians = GaussianModel(dataset.sh_degree)  # 创建高斯模型实例
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)  # 创建场景实例
    bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]  # 设置背景颜色
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")  # 创建背景张量
    
    train_dir = os.path.join(args.model_path, 'train', "ours_{}".format(scene.loaded_iter))  # 设置训练输出目录
    test_dir = os.path.join(args.model_path, 'test', "ours_{}".format(scene.loaded_iter))  # 设置测试输出目录
    gaussExtractor = GaussianExtractor(gaussians, render, pipe, bg_color=bg_color)  # 创建高斯提取器实例
    
    if not args.skip_train:  # 如果不跳过训练
        print("export training images ...")  # 打印导出训练图像信息
        os.makedirs(train_dir, exist_ok=True)  # 创建训练目录
        gaussExtractor.reconstruction(scene.getTrainCameras())  # 重建训练相机视图
        gaussExtractor.export_image(train_dir)  # 导出训练图像
        
    
    if (not args.skip_test) and (len(scene.getTestCameras()) > 0):  # 如果不跳过测试且有测试相机
        print("export rendered testing images ...")  # 打印导出测试图像信息
        os.makedirs(test_dir, exist_ok=True)  # 创建测试目录
        gaussExtractor.reconstruction(scene.getTestCameras())  # 重建测试相机视图
        gaussExtractor.export_image(test_dir)  # 导出测试图像
    
    
    if args.render_path:  # 如果需要渲染路径
        print("render videos ...")  # 打印渲染视频信息
        traj_dir = os.path.join(args.model_path, 'traj', "ours_{}".format(scene.loaded_iter))  # 设置轨迹目录
        os.makedirs(traj_dir, exist_ok=True)  # 创建轨迹目录
        n_fames = 240  # 设置帧数
        cam_traj = generate_path(scene.getTrainCameras(), n_frames=n_fames)  # 生成相机轨迹
        gaussExtractor.reconstruction(cam_traj)  # 重建轨迹视图
        gaussExtractor.export_image(traj_dir)  # 导出轨迹图像
        create_videos(base_dir=traj_dir,  # 创建视频
                    input_dir=traj_dir, 
                    out_name='render_traj', 
                    num_frames=n_fames)

    if not args.skip_mesh:  # 如果不跳过网格生成
        print("export mesh ...")  # 打印导出网格信息
        os.makedirs(train_dir, exist_ok=True)  # 创建训练目录
        gaussExtractor.gaussians.active_sh_degree = 0  # 设置SH度数为0
        gaussExtractor.reconstruction(scene.getTrainCameras())  # 重建训练相机视图
        if args.unbounded:  # 如果是无边界模式
            name = 'fuse_unbounded.ply'  # 设置无边界网格文件名
            mesh = gaussExtractor.extract_mesh_unbounded(resolution=args.mesh_res)  # 提取无边界网格
        else:  # 否则
            name = 'fuse.ply'  # 设置普通网格文件名
            depth_trunc = (gaussExtractor.radius * 2.0) if args.depth_trunc < 0  else args.depth_trunc  # 计算深度截断值
            voxel_size = (depth_trunc / args.mesh_res) if args.voxel_size < 0 else args.voxel_size  # 计算体素大小
            sdf_trunc = 5.0 * voxel_size if args.sdf_trunc < 0 else args.sdf_trunc  # 计算SDF截断值
            mesh = gaussExtractor.extract_mesh_bounded(voxel_size=voxel_size, sdf_trunc=sdf_trunc, depth_trunc=depth_trunc)  # 提取有边界网格
        
        o3d.io.write_triangle_mesh(os.path.join(train_dir, name), mesh)  # 保存网格文件
        print("mesh saved at {}".format(os.path.join(train_dir, name)))  # 打印网格保存路径
        mesh_post = post_process_mesh(mesh, cluster_to_keep=args.num_cluster)  # 后处理网格
        o3d.io.write_triangle_mesh(os.path.join(train_dir, name.replace('.ply', '_post.ply')), mesh_post)  # 保存后处理网格
        print("mesh post processed saved at {}".format(os.path.join(train_dir, name.replace('.ply', '_post.ply'))))  # 打印后处理网格保存路径
