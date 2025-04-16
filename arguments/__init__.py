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
            value = value if not fill_none else None
            help_text = descriptions.get(key, None)  # 获取描述
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true", help=help_text)
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t, help=help_text)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true", help=help_text)
                else:
                    group.add_argument("--" + key, default=value, type=t, help=help_text)

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
    def __init__(self, parser):
        # 迭代次数
        self.iterations = 14_000
        
        # 学习率参数
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000008
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 14_000
        self.feature_lr = 0.0025
        self.opacity_lr = 0.05
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        
        # 稠密化参数
        self.densification_interval = 100
        self.opacity_reset_interval = 1000
        self.densify_from_iter = 500
        self.densify_until_iter = 13_000
        self.densify_grad_threshold = 0.0001
        
        # 正则化参数
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        self.lambda_dist = 0.0005
        self.lambda_normal = 0.1
        self.opacity_cull = 0.00005
        
        # 深度收敛损失参数
        self.use_depth_convergence = True  # 是否启用深度收敛损失
        self.lambda_depth_convergence = 0.1  # 深度收敛损失权重
        self.conv_start_iter = 3000  # 从哪个迭代开始应用深度收敛损失
        
        # 背景点处理参数
        self.bg_start_iter = 500  # 从哪个迭代开始应用背景点处理
        self.set_background_opacity_to_zero = True  # 启用背景高斯点不透明度直接设为0，默认开启
        
        # 数据增强相关参数
        self.use_data_augmentation = False  # 是否使用数据增强
        
        # 其他参数保持不变
        self.use_edge_aware_normal = True
        self.edge_weight_exponent = 4.0
        self.lambda_consistency = 0.5
        self.use_ms_ssim = True
        
        # 为参数添加帮助信息
        descriptions = {
            "use_edge_aware_normal": "启用边缘感知法向损失，在边缘区域放松约束，平坦区域增强约束",
            "edge_weight_exponent": "边缘权重指数q (默认: 4.0)，控制边缘敏感度",
            "lambda_consistency": "法线一致性权重 (默认: 0.5)，控制原始法线一致性损失的权重",
            "use_ms_ssim": "启用多尺度SSIM损失，提升多视角一致性",
            "lambda_dssim": "SSIM在颜色损失中的权重 (默认: 0.2)",
            "lambda_normal": "法线正则化强度 (默认: 0.05)",
            "lambda_dist": "深度失真正则化强度 (默认: 0.0)",
            "use_depth_convergence": "启用深度收敛损失，强制相邻高斯基元深度接近，使表面更平滑",
            "lambda_depth_convergence": "深度收敛损失权重 (默认: 0.01)",
            "conv_start_iter": "从哪个迭代开始应用深度收敛损失 (默认: 3000)",
            "bg_start_iter": "从哪个迭代开始应用背景点处理 (默认: 500)",
            "set_background_opacity_to_zero": "启用背景高斯点不透明度清零 (默认开启)",
            "opacity_cull": "不透明度剪枝阈值 (默认: 0.005)，小于此值的点会被裁剪",
            "use_data_augmentation": "是否使用数据增强",
        }
        
        super().__init__(parser, "Optimization Parameters", descriptions=descriptions)

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
