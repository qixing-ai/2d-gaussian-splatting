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
import sys
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, OptimizationParams
from train import training, prepare_output_and_logger

def config_parser():
    parser = ArgumentParser(description="2D高斯散射训练增强版 (2D Gaussian Splatting Enhanced)")
    
    # 基本路径设置
    parser.add_argument('--source_path', '-s', required=True, type=str, 
                      help="数据集路径 (COLMAP或NeRF合成数据集)")
    parser.add_argument('--model_path', type=str, default="",
                      help="输出模型路径")
    parser.add_argument('--checkpoint', type=str, default="",
                      help="恢复训练的检查点路径")
    
    # 增强损失函数设置
    parser.add_argument('--use_edge_aware_normal', action="store_true", 
                      help="启用边缘感知法向损失")
    parser.add_argument('--no_edge_aware_normal', action="store_false", dest="use_edge_aware_normal",
                      help="禁用边缘感知法向损失")
    parser.set_defaults(use_edge_aware_normal=True)  # 默认启用
    parser.add_argument('--edge_weight_exponent', type=float, default=4.0,
                      help="边缘权重指数q (默认: 4)")
    parser.add_argument('--lambda_consistency', type=float, default=0.5,
                      help="法线一致性权重 (默认: 0.5)")
    parser.add_argument('--use_ms_ssim', action="store_true",
                      help="启用多尺度SSIM损失")
    parser.add_argument('--no_ms_ssim', action="store_false", dest="use_ms_ssim",
                      help="禁用多尺度SSIM，使用普通SSIM")
    parser.set_defaults(use_ms_ssim=True)  # 默认启用
    parser.add_argument('--lambda_dssim', type=float, default=0.2,
                      help="SSIM损失权重 (默认: 0.2)")
    
    # 正则化设置
    parser.add_argument('--lambda_normal', type=float, default=0.05,
                      help="法线正则化强度 (默认: 0.05)")
    parser.add_argument('--lambda_dist', type=float, default=0.0,
                      help="深度失真正则化强度 (默认: 0)")
    parser.add_argument('--depth_ratio', type=float, default=0.0,
                      help="深度比率: 0表示平均深度，1表示中值深度 (0效果较好)")
    
    # 自定义迭代次数
    parser.add_argument('--iterations', type=int, default=30000,
                      help="总训练迭代次数 (默认: 30000)")
    
    return parser

if __name__ == "__main__":
    parser = config_parser()
    args = parser.parse_args()
    
    print("===> 2D高斯散射增强版训练")
    print("===> 启用边缘感知法向损失:", args.use_edge_aware_normal)
    print("===> 启用多尺度SSIM:", args.use_ms_ssim)
    
    # 设置参数
    model_params = ModelParams(parser)
    pipeline_params = PipelineParams(parser)
    opt_params = OptimizationParams(parser)
    
    # 应用命令行参数
    model_params.source_path = args.source_path
    model_params.model_path = args.model_path
    
    # 更新损失函数参数
    opt_params.use_edge_aware_normal = args.use_edge_aware_normal
    opt_params.edge_weight_exponent = args.edge_weight_exponent
    opt_params.lambda_consistency = args.lambda_consistency
    opt_params.use_ms_ssim = args.use_ms_ssim
    opt_params.lambda_dssim = args.lambda_dssim
    opt_params.lambda_normal = args.lambda_normal
    opt_params.lambda_dist = args.lambda_dist
    opt_params.iterations = args.iterations
    
    # 更新管道参数
    pipeline_params.depth_ratio = args.depth_ratio
    
    print("\n===> 训练参数:")
    print(f"数据集路径: {model_params.source_path}")
    print(f"输出路径: {model_params.model_path}")
    print(f"边缘权重指数: {opt_params.edge_weight_exponent}")
    print(f"法线正则化强度: {opt_params.lambda_normal}")
    print(f"深度失真正则化强度: {opt_params.lambda_dist}")
    print(f"SSIM损失权重: {opt_params.lambda_dssim}")
    print(f"训练迭代次数: {opt_params.iterations}")
    
    # 设置测试/保存/检查点迭代
    test_iterations = [i for i in range(0, args.iterations+1, 5000)]
    test_iterations += [1000, 2000, 3000, 4000]
    test_iterations.sort()
    
    save_iterations = [i for i in range(0, args.iterations+1, 5000)]
    save_iterations += [args.iterations]
    save_iterations.sort()
    
    checkpoint_iterations = [i for i in range(0, args.iterations+1, 10000)]
    checkpoint_iterations += [args.iterations]
    checkpoint_iterations.sort()
    
    # 设置系统路径和导入数据
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from scene.dataset_readers import createDatasetReader
    
    # 创建数据集对象
    gaussians_dataset = createDatasetReader(model_params)
    
    # 使用相同的训练函数
    training(gaussians_dataset, opt_params, pipeline_params,
            test_iterations, save_iterations, checkpoint_iterations, 
            args.checkpoint)

    print("===> 训练完成!") 