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
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None 
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

class ModelParams(ParamGroup): 
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3  # 球谐函数的阶数
        self._source_path = ""  # 输入数据源路径
        self._model_path = ""  # 模型保存路径
        self._images = "images"  # 图像文件夹名称
        self._resolution = -1  # 图像分辨率(-1表示原始分辨率)
        self._white_background = False  # 是否使用白色背景
        self.data_device = "cuda"  # 数据加载设备(cuda/cpu)
        self.eval = False  # 是否为评估模式
        self.render_items = ['RGB', 'Alpha', 'Normal', 'Depth', 'Edge', 'Curvature']  # 渲染输出项列表
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g

class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False  # 是否使用Python计算球谐函数
        self.compute_cov3D_python = False  # 是否使用Python计算3D协方差
        self.depth_ratio = 0.0  # 深度图混合比例
        self.debug = False  # 是否启用调试模式
        super().__init__(parser, "Pipeline Parameters")

class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 30_000  # 总迭代次数
        self.position_lr_init = 0.00016  # 位置学习率初始值
        self.position_lr_final = 0.0000016  # 位置学习率最终值
        self.position_lr_delay_mult = 0.01  # 位置学习率延迟乘数
        self.position_lr_max_steps = 30_000  # 位置学习率最大步数
        self.feature_lr = 0.0025  # 特征学习率
        self.opacity_lr = 0.05  # 不透明度学习率
        self.scaling_lr = 0.005  # 缩放学习率
        self.rotation_lr = 0.001  # 旋转学习率
        self.percent_dense = 0.01  # 密集化百分比
        self.lambda_dssim = 0.2  # DSSIM损失权重
        self.lambda_dist = 0 # 没用,这个是为了让表面高斯深度一致的,但是不一定是在表面
        self.lambda_normal = 0.05  # 法线损失权重,非常有用这个是为了让法线一致的,高斯深度也会被一致话
        self.lambda_alpha = 0.1 # 透明度损失权重(控制背景点透明度的权重)
        self.lambda_edge_aware = 0.1 # 边缘感知损失权重(边缘感知曲率损失的权重)
        self.opacity_cull = 0.05  # 不透明度剔除阈值

        # Contribution-based pruning parameters
        self.prune_ratio = 0.01 # 修剪比例(0-1)
        self.contribution_gamma = 0.25 # 贡献度计算参数
        self.contribution_prune_interval = 1000 # 修剪间隔(迭代次数)

        self.densification_interval = 100  # 密集化间隔
        self.opacity_reset_interval = 0  # 不透明度重置间隔
        self.densify_from_iter = 500  # 开始密集化的迭代次数
        self.densify_until_iter = 29_000  # 停止密集化的迭代次数
        self.densify_grad_threshold = 0.0002  # 密集化梯度阈值
        super().__init__(parser, "Optimization Parameters")

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
