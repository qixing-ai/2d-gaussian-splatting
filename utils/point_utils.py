import torch

def depths_to_points(view, depthmap):
    c2w = (view.world_view_transform.T).inverse()
    W, H = view.image_width, view.image_height
    ndc2pix = torch.tensor([
        [W / 2, 0, 0, (W) / 2],
        [0, H / 2, 0, (H) / 2],
        [0, 0, 0, 1]]).float().cuda().T
    projection_matrix = c2w.T @ view.full_proj_transform
    intrins = (projection_matrix @ ndc2pix)[:3,:3].T
    
    grid_x, grid_y = torch.meshgrid(torch.arange(W, device='cuda').float(), torch.arange(H, device='cuda').float(), indexing='xy')
    points = torch.stack([grid_x, grid_y, torch.ones_like(grid_x)], dim=-1).reshape(-1, 3)
    rays_d = points @ intrins.inverse().T @ c2w[:3,:3].T
    rays_o = c2w[:3,3]
    points = depthmap.reshape(-1, 1) * rays_d + rays_o
    return points

def depth_to_normal_legacy(view, depth):
    """
    原始的2像素间距法向量计算算法
    view: view camera
    depth: depthmap 
    """
    points = depths_to_points(view, depth).reshape(*depth.shape[1:], 3)
    output = torch.zeros_like(points)
    dx = torch.cat([points[2:, 1:-1] - points[:-2, 1:-1]], dim=0)
    dy = torch.cat([points[1:-1, 2:] - points[1:-1, :-2]], dim=1)
    normal_map = torch.nn.functional.normalize(torch.cross(dx, dy, dim=-1), dim=-1)
    output[1:-1, 1:-1, :] = normal_map
    return output

def depth_to_normal_precise(view, depth):
    """
    精确的1像素间距法向量计算算法
    view: view camera
    depth: depthmap 
    """
    points = depths_to_points(view, depth).reshape(*depth.shape[1:], 3)
    output = torch.zeros_like(points)
    
    # 使用1像素间距计算更精确的梯度，但保持原始的输出区域
    # 计算1像素间距的梯度
    dx = points[1:, 1:-1, :] - points[:-1, 1:-1, :]  # 垂直方向 (H-1, W-2, 3)
    dy = points[1:-1, 1:, :] - points[1:-1, :-1, :]  # 水平方向 (H-2, W-1, 3)
    
    # 取重叠区域进行叉积计算
    dx_crop = dx[:-1, :, :]  # (H-2, W-2, 3)
    dy_crop = dy[:, :-1, :]  # (H-2, W-2, 3)
    
    normal_map = torch.nn.functional.normalize(torch.cross(dx_crop, dy_crop, dim=-1), dim=-1)
    output[1:-1, 1:-1, :] = normal_map
    return output

def depth_to_normal(view, depth, use_precise=True):
    """
    根据参数选择使用哪种法向量计算算法
    view: view camera
    depth: depthmap 
    use_precise: 是否使用精确的1像素间距算法，False则使用原始的2像素间距算法
    """
    if use_precise:
        return depth_to_normal_precise(view, depth)
    else:
        return depth_to_normal_legacy(view, depth)