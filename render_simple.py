import numpy as np
import torch
import open3d as o3d
from scene import Scene
from argparse import ArgumentParser
from gaussian_renderer import GaussianModel
from arguments import ModelParams

if __name__ == "__main__":
    # 简化参数解析
    parser = ArgumentParser()
    model = ModelParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--output", default="output.ply", type=str)
    parser.add_argument("--search_radius", type=float, default=0.1, 
                      help="法线估计搜索半径")
    parser.add_argument("--max_distance", type=float, default=0.15,
                      help="最大连接距离")
    args = parser.parse_args()

    # 加载场景和高斯模型
    dataset = model.extract(args)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
    
    # 提取高斯点中心坐标(分离计算图)
    points = gaussians._xyz.detach().cpu().numpy()
    
    # 创建点云并生成简单网格
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # 法线估计(基于K近邻)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    
    # 基于邻近点生成网格
    points = np.asarray(pcd.points)
    triangles = []
    
    # 构建KDTree加速邻近点搜索
    tree = o3d.geometry.KDTreeFlann(pcd)
    
    # 生成三角面片
    for i in range(len(points)):
        # 找到半径内的邻近点
        [k, idx, _] = tree.search_radius_vector_3d(pcd.points[i], args.max_distance)
        
        # 简单连接邻近点形成三角面
        if k > 3:  # 至少有3个邻近点
            triangles.append([i, idx[1], idx[2]])
            if k > 4:
                triangles.append([i, idx[2], idx[3]])
    
    # 创建网格对象
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(points)
    mesh.triangles = o3d.utility.Vector3iVector(np.array(triangles))
    
    # 保存网格文件
    o3d.io.write_triangle_mesh(args.output, mesh)
    print(f"Mesh saved to {args.output}")
