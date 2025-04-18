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

from argparse import ArgumentParser, Namespace
import sys
import os

class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False, descriptions=None):
        group = parser.add_argument_group(name)
        descriptions = descriptions or {}
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            actual_type = t if t != type(None) else str 
            value = value if not fill_none else None
            help_text = descriptions.get(key, None)  # 获取描述
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true", help=help_text)
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=actual_type, help=help_text)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true", help=help_text)
                else:
                    group.add_argument("--" + key, default=value, type=actual_type, help=help_text)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

class ModelParams(ParamGroup): 
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._resolution = -1
        self._white_background = False
        self.data_device = "cuda"
        self.eval = False
        self.render_items = ['RGB', 'Alpha', 'Normal', 'Depth', 'Edge', 'Curvature']
        descriptions = {
            "sh_degree": "球谐系数的度数",
            "source_path": "数据集路径",
            "model_path": "输出模型路径",
            "images": "图像文件夹名称",
            "resolution": "图像分辨率，-1表示使用原始分辨率",
            "white_background": "使用白色背景",
        }
        super().__init__(parser, "Loading Parameters", sentinel, descriptions=descriptions)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g

class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.depth_ratio = 0.0
        self.debug = False
        descriptions = {
            "depth_ratio": "深度比率: 0表示平均深度，1表示中值深度 (默认: 0)",
            "debug": "启用调试模式",
        }
        super().__init__(parser, "Pipeline Parameters", descriptions=descriptions)

class OptimizationParams(ParamGroup):
    def __init__(self, parser, lr=0.00016, iterations=14_000):
        # 基本学习率参数
        self.iterations = iterations
        self.position_lr_init = lr
        self.position_lr_final = 0.0000008
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 14_000
        self.feature_lr = 0.0025
        self.opacity_lr = 0.05
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        
        # 稠密化和裁剪参数
        self.densification_interval = 100
        self.opacity_reset_interval = 1000
        self.densify_from_iter = 500
        self.densify_until_iter = 12_000
        self.densify_grad_threshold = 0.0001
        self.cull_from_iter = -1  # 默认值为-1，extract中会检查
        self.percent_dense = 0.01
        self.opacity_cull = 0.1
        
        # 损失权重参数
        self.lambda_dssim = 0.2
        self.lambda_dist = 0.0005
        self.lambda_normal = 0.1
        self.lambda_consistency = 0.5
        self.lambda_depth_convergence = 0.1
        self.lambda_bg_opacity = 0.1
        self.lambda_depth_smoothness = 1.0
        
        # 特性开关参数
        self.use_depth_convergence = True
        self.use_edge_aware_normal = True
        self.use_ms_ssim = True
        self.use_fused_ssim = True
        self.set_background_opacity_to_zero = True
        self.use_data_augmentation = False
        
        # 其他配置参数
        self.conv_start_iter = 1000
        self.bg_start_iter = 500
        self.edge_weight_exponent = 4.0
        
        # 参数描述字典
        descriptions = {
            # 基本学习率参数
            "iterations": "训练迭代次数",
            "position_lr_init": "初始位置学习率",
            "position_lr_final": "最终位置学习率",
            "position_lr_delay_mult": "位置学习率延迟乘数",
            "position_lr_max_steps": "位置学习率变化最大步数",
            "feature_lr": "特征学习率",
            "opacity_lr": "不透明度学习率",
            "scaling_lr": "缩放学习率",
            "rotation_lr": "旋转学习率",
            
            # 稠密化和裁剪参数
            "densification_interval": "稠密化间隔",
            "opacity_reset_interval": "不透明度重置间隔",
            "densify_from_iter": "从哪个迭代开始稠密化",
            "densify_until_iter": "到哪个迭代停止稠密化",
            "densify_grad_threshold": "稠密化梯度阈值",
            "cull_from_iter": "从哪个迭代开始进行不透明度裁剪 (默认与 densify_from_iter 相同)",
            "percent_dense": "初始点云密度的百分比",
            "opacity_cull": "不透明度剪枝阈值 (默认: 0.005)，小于此值的点会被裁剪",
            
            # 损失权重参数
            "lambda_dssim": "SSIM在颜色损失中的权重 (默认: 0.2)",
            "lambda_dist": "深度失真正则化强度 (默认: 0.0)",
            "lambda_normal": "法线正则化强度 (默认: 0.05)",
            "lambda_consistency": "法线一致性权重 (默认: 0.5)，控制原始法线一致性损失的权重",
            "lambda_depth_convergence": "深度收敛损失权重 (默认: 0.01)",
            "lambda_bg_opacity": "背景透明度损失权重",
            "lambda_depth_smoothness": "背景深度平滑损失权重",
            
            # 特性开关参数
            "use_depth_convergence": "启用深度收敛损失，强制相邻高斯基元深度接近，使表面更平滑",
            "use_edge_aware_normal": "启用边缘感知法向损失，在边缘区域放松约束，平坦区域增强约束",
            "use_ms_ssim": "启用多尺度SSIM损失，提升多视角一致性",
            "use_fused_ssim": "启用fused_ssim损失 (比ms_ssim更快，但可能效果略差，优先级高于ms_ssim)",
            "set_background_opacity_to_zero": "启用背景高斯点不透明度清零 (默认开启)",
            "use_data_augmentation": "是否使用数据增强",
            
            # 其他配置参数
            "conv_start_iter": "从哪个迭代开始应用深度收敛损失 (默认: 3000)",
            "bg_start_iter": "从哪个迭代开始应用背景点处理 (默认: 500)",
            "edge_weight_exponent": "边缘权重指数q (默认: 4.0)，控制边缘敏感度",
        }
        
        # 调用基类构造函数处理参数注册
        super().__init__(parser, "Optimization Parameters", descriptions=descriptions)

    def extract(self, args):
        g = super().extract(args)
        # 如果命令行未指定cull_from_iter，则使用densify_from_iter的值
        if g.cull_from_iter == -1:
            g.cull_from_iter = g.densify_from_iter
        return g

def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)

# 添加TrainingParams类
class TrainingParams:
    """统一管理所有训练参数的类"""
    def __init__(self, parser: ArgumentParser):
        # 实例化各个参数组并保存引用
        self.model_params = ModelParams(parser, sentinel=False)
        self.optimization_params = OptimizationParams(parser)
        self.pipeline_params = PipelineParams(parser)
        
        # 添加网络和调试参数
        network_group = parser.add_argument_group("网络和调试参数")
        network_group.add_argument('--ip', type=str, default="127.0.0.1", help="GUI服务器IP地址")
        network_group.add_argument('--port', type=int, default=6009, help="GUI服务器端口")
        network_group.add_argument('--detect_anomaly', action='store_true', default=False, help="启用PyTorch异常检测")
        network_group.add_argument("--quiet", action="store_true", help="静默模式，减少输出信息")
        
        # 添加保存和检查点参数
        checkpoint_group = parser.add_argument_group("保存和检查点参数")
        checkpoint_group.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 12_000], 
                                     help="在哪些迭代进行测试")
        checkpoint_group.add_argument("--save_iterations", nargs="+", type=int, 
                                     default=[i for i in range(1000, 30001, 1000)],
                                     help="在哪些迭代保存模型")
        checkpoint_group.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[],
                                     help="在哪些迭代保存检查点")
        checkpoint_group.add_argument("--start_checkpoint", type=str, default=None,
                                     help="从哪个检查点恢复训练")

    def extract(self, args):
        """从命令行参数中提取所有需要的参数"""
        params = GroupParams()
        
        # 提取模型、优化和渲染管线参数
        model = self.model_params.extract(args)
        optimization = self.optimization_params.extract(args)
        pipeline = self.pipeline_params.extract(args)
        
        # 设置到params对象
        params.model = model
        params.optimization = optimization
        params.pipeline = pipeline
        
        # 提取网络和调试参数
        params.ip = args.ip
        params.port = args.port
        params.detect_anomaly = args.detect_anomaly
        params.quiet = args.quiet
        
        # 提取保存和检查点参数
        params.test_iterations = args.test_iterations
        params.save_iterations = args.save_iterations
        params.checkpoint_iterations = args.checkpoint_iterations
        params.start_checkpoint = args.start_checkpoint
        
        # 确保最后一次迭代也保存
        if hasattr(params.optimization, 'iterations') and params.optimization.iterations not in params.save_iterations:
            params.save_iterations.append(params.optimization.iterations)
        
        return params
