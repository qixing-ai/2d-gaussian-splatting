import os
import torch
import numpy as np
from plyfile import PlyData
import matplotlib.pyplot as plt
import io
import trimesh

def add_ply_to_tensorboard(writer, tag, ply_path, global_step=None):
    """将PLY文件添加到TensorBoard中可视化
    
    Args:
        writer: TensorBoard SummaryWriter实例
        tag: 在TensorBoard中显示的标签
        ply_path: PLY文件路径
        global_step: 全局步数
    """
    try:
        # 读取点云并添加到TensorBoard
        mesh = trimesh.load(ply_path)
        
        # 提取点和颜色
        vertices = mesh.vertices
        colors = mesh.colors / 255.0 if mesh.colors is not None else np.ones_like(vertices)
        
        # 添加到TensorBoard
        writer.add_mesh(
            tag,
            vertices=[vertices],
            colors=[colors],
            global_step=global_step
        )
        
    except Exception as e:
        print(f"无法添加PLY到TensorBoard: {e}")
        # 记录文件路径
        writer.add_text(f'{tag}/ply_path', f"PLY文件: {ply_path}", global_step) 