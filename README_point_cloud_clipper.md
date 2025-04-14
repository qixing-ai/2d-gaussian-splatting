# 点云裁剪工具使用说明

## 简介

点云裁剪工具(`point_cloud_clipper.py`)是一个使用COLMAP相机参数和图像透明度信息来裁剪点云的实用程序。它的主要目的是通过图像背景的透明度信息来过滤点云，只保留那些投影到图像前景区域的点，从而获得更准确的3D模型形状。

这个工具特别适用于需要从一团均匀密集的点云中裁剪出3D模型形状的场景。它可以：
- 使用外部密集点云文件作为输入
- 在指定边界内生成均匀分布的点云
- 根据COLMAP相机参数和透明背景图像进行裁剪
- 输出裁剪后的点云和可视化结果

## 安装依赖

在使用此工具前，请确保安装以下依赖库：

```bash
pip install numpy pillow scipy matplotlib open3d tqdm
```

同时，请确保你的环境中已经包含了COLMAP的输出数据，包括相机参数和对应的透明背景图像。

## 使用方法

### 基本用法

```bash
# 使用均匀生成的点云进行裁剪
python point_cloud_clipper.py --data_dir /path/to/colmap/data --images_dir /path/to/transparent/images

# 使用外部点云文件进行裁剪
python point_cloud_clipper.py --data_dir /path/to/colmap/data --input_pointcloud dense.ply
```

### 命令行参数

以下是可用的命令行参数：

- `--data_dir`: (必需) COLMAP数据目录，包含sparse/0/等子目录
- `--images_dir`: 包含透明背景图像的目录，默认为data_dir/images
- `--output_ply`: 输出的裁剪后点云文件名，默认为'clipped_pointcloud.ply'
- `--input_pointcloud`: 输入的密集点云文件路径(.ply, .pcd等)
- `--min_views`: 一个点需要在至少多少个视角中可见才会被保留，默认为3
- `--alpha_threshold`: 透明度掩码阈值（0.0-1.0），默认为0.5
- `--dense_points`: 生成的均匀密集点云点数，默认为1000000
- `--vis_output`: 可视化可见度的点云输出文件名，默认为'visibility_pointcloud.ply'
- `--num_workers`: 并行工作进程数，默认为CPU核心数减1
- `--batch_size`: 批处理大小，默认为1000
- `--scene_scale`: 场景边界框缩放因子，默认为1.5
- `--bounds_min`: 手动指定边界框最小值，格式为"x,y,z"
- `--bounds_max`: 手动指定边界框最大值，格式为"x,y,z"

### 使用示例

1. 使用外部密集点云文件进行裁剪：

```bash
python point_cloud_clipper.py --data_dir ./colmap_output --input_pointcloud ./dense.ply --min_views 2
```

2. 在场景边界内生成均匀分布的密集点云并进行裁剪：

```bash
python point_cloud_clipper.py --data_dir ./colmap_output --dense_points 2000000 --alpha_threshold 0.7
```

3. 手动指定点云边界：

```bash
python point_cloud_clipper.py --data_dir ./colmap_output --bounds_min "-1,-1,-1" --bounds_max "1,1,1" --dense_points 1500000
```

4. 指定自定义图像目录和输出文件：

```bash
python point_cloud_clipper.py --data_dir ./colmap_output --images_dir ./transparent_images --output_ply ./output/model.ply
```

## 输出文件

该工具会生成两个主要输出文件：

1. 裁剪后的点云文件（由`--output_ply`指定）
2. 可视化可见度的点云文件（由`--vis_output`指定）：用热力图颜色表示每个点被多少相机观察到

## 工作原理

1. 加载COLMAP相机参数
2. 获取输入点云（从文件加载或生成均匀分布的点云）
3. 从图像的透明通道生成二值掩码
4. 将点投影到每个相机视图，检查它们是否落在图像前景区域内
5. 计算每个点在多少个视角中可见
6. 筛选出在足够多视角中可见的点
7. 保存裁剪后的点云

## 点云来源选择

根据具体需求，可以选择以下方式获取输入点云：

1. **外部密集点云**：使用`--input_pointcloud`参数指定外部PLY或PCD文件
2. **均匀生成点云**：程序自动在场景边界内生成均匀分布的点云，数量由`--dense_points`指定
3. **边界确定方式**：
   - 使用`--bounds_min`和`--bounds_max`手动指定边界
   - 使用COLMAP稀疏点云的边界（如果存在）
   - 根据相机位置估计场景边界

## 注意事项

- 图像必须包含透明背景（alpha通道），通常为PNG格式
- 相机参数必须是COLMAP格式
- 处理大量点或图像可能需要较长时间，可以调整`--num_workers`参数提高性能
- 对于大场景，使用外部点云文件通常比生成均匀点云更准确

## 故障排除

如果遇到问题：

1. 确保COLMAP输出格式正确
2. 检查图像是否包含透明通道
3. 尝试调整`--min_views`和`--alpha_threshold`参数
4. 尝试手动指定合理的边界范围
5. 如果处理速度太慢，可以减少点云数量或增加工作进程数 