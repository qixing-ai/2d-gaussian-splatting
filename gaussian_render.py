import numpy as np
import torch
import open3d as o3d
from scene import Scene
from argparse import ArgumentParser
from gaussian_renderer import GaussianModel
from arguments import ModelParams

from utils.general_utils import build_rotation

def quat_to_rot_matrix(q):
    """将四元数转换为旋转矩阵"""
    # 使用GaussianModel中的build_rotation函数
    q_tensor = torch.tensor(q, device="cuda").unsqueeze(0)
    rot_matrix = build_rotation(q_tensor).squeeze(0).cpu().numpy()
    return rot_matrix

if __name__ == "__main__":
    # 简化参数解析
    parser = ArgumentParser()
    model = ModelParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--output", default="output.ply", type=str)
    parser.add_argument("--sample_density", type=int, default=10,
                      help="椭圆面采样密度(每边点数)")
    parser.add_argument("--camera_pos", nargs=3, type=float, default=[0.0, 0.0, 0.0],
                      help="相机位置 (x y z)")
    args = parser.parse_args()

    # 加载场景和参数
    dataset = model.extract(args)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
    
    # 提取高斯参数
    centers = gaussians._xyz.detach().cpu().numpy()
    scales = gaussians.get_scaling.detach().cpu().numpy()
    rotations = gaussians.get_rotation.detach().cpu().numpy()
    # 获取SH系数并转换为RGB颜色
    shs = gaussians.get_features.detach().cpu().numpy()
    # 只使用DC分量作为基础颜色
    colors = shs[:,0,:3]  # 取第一个球谐系数作为颜色
    
    # 相机位置
    camera_position = np.array(args.camera_pos)
    
    # 创建最终网格
    final_mesh = o3d.geometry.TriangleMesh()
    
    # 生成圆盘顶点的基础模板(可复用)
    theta = np.linspace(0, 2*np.pi, args.sample_density, endpoint=False)
    template_x = np.cos(theta)
    template_y = np.sin(theta)
    template_vertices = np.column_stack([template_x, template_y, np.zeros_like(theta)])
    template_vertices = np.vstack([template_vertices, [0, 0, 0]])
    
    # 生成基础三角形索引(可复用)
    n = len(theta)
    triangles = [[j, (j+1)%n, n] for j in range(n)]
    
    for i in range(len(centers)):
        # 缩放和应用变换
        vertices = template_vertices.copy()
        vertices[:-1,0] *= scales[i,0]
        vertices[:-1,1] *= scales[i,1]
        
        # 应用旋转和平移
        rot_matrix = quat_to_rot_matrix(rotations[i])
        vertices = np.dot(vertices, rot_matrix.T) + centers[i]
        
        # 计算从高斯中心到相机的方向向量
        view_direction = camera_position - centers[i]
        view_direction = view_direction / np.linalg.norm(view_direction)  # 归一化
        
        # 获取高斯圆盘的法线方向(旋转矩阵的第三列，即局部z轴)
        normal_direction = rot_matrix[:,2]
        
        # 确保法线朝向相机：如果法线与视线方向夹角大于90度，则翻转法线
        if np.dot(normal_direction, view_direction) < 0:
            normal_direction = -normal_direction
        
        # 创建单个圆盘网格
        disk = o3d.geometry.TriangleMesh()
        disk.vertices = o3d.utility.Vector3dVector(vertices)
        disk.triangles = o3d.utility.Vector3iVector(triangles)
        disk.vertex_normals = o3d.utility.Vector3dVector(np.tile(normal_direction, (len(vertices), 1)))
        
        # 设置顶点颜色
        rgb_color = 1/(1+np.exp(-colors[i]))  # sigmoid激活颜色
        disk.vertex_colors = o3d.utility.Vector3dVector(np.tile(rgb_color.astype(np.float64), (len(vertices), 1)))
        
        # 合并到最终网格
        final_mesh += disk
    
    # 保存网格
    o3d.io.write_triangle_mesh(args.output, final_mesh, write_vertex_normals=True)
    print(f"Disk mesh saved to {args.output}")
    print(f"Camera position: {camera_position}")
