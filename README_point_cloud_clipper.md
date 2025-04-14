# 点云裁剪工具使用说明

## 简介

点云裁剪工具(`point_cloud_clipper.py`)是一个使用COLMAP相机参数和图像透明度信息来裁剪点云的实用程序。它的主要目的是通过图像背景的透明度信息来过滤点云，只保留那些投影到图像前景区域的点，从而获得更准确的3D模型形状。

这个工具特别适用于需要从一团均匀密集的点云中裁剪出3D模型形状的场景。它可以：
- 在指定边界内生成均匀分布的点云
- 根据COLMAP相机参数和透明背景图像给点云评分
- 根据评分筛选和输出裁剪后的点云

## 安装依赖

在使用此工具前，请确保安装以下依赖库：

```bash
pip install numpy pillow scipy matplotlib open3d tqdm
```

同时，请确保你的环境中已经包含了COLMAP的输出数据，包括相机参数和对应的透明背景图像。

## 使用方法

### 基本用法

```bash
# 最简单的用法
python point_cloud_clipper.py --data_dir /path/to/colmap/data
```

### 命令行参数

以下是可用的命令行参数：

- `--data_dir`: (必需) COLMAP数据目录，包含sparse/0/等子目录
- `--images_dir`: 包含透明背景图像的目录，默认为data_dir/images
- `--output_ply`: 输出的裁剪后点云文件名，默认为'scored_pointcloud.ply'
- `--dense_points`: 生成的均匀密集点云点数，默认为1000000
- `--num_workers`: 并行工作进程数，默认为CPU核心数减1
- `--batch_size`: 批处理大小，默认为1000
- `--scene_scale`: 场景边界框缩放因子，默认为1.5
- `--bounds_min`: 手动指定边界框最小值，格式为"x,y,z"
- `--bounds_max`: 手动指定边界框最大值，格式为"x,y,z"
- `--keep_percentage`: 保留得分最高的点云百分比，默认为10%

### 使用示例

1. 在场景边界内生成均匀分布的密集点云并进行裁剪：

```bash
python point_cloud_clipper.py --data_dir ./colmap_output --dense_points 2000000 --keep_percentage 20
```

2. 手动指定点云边界：

```bash
python point_cloud_clipper.py --data_dir ./colmap_output --bounds_min "-1,-1,-1" --bounds_max "1,1,1"
```

3. 指定自定义图像目录和输出文件：

```bash
python point_cloud_clipper.py --data_dir ./colmap_output --images_dir ./transparent_images --output_ply ./output/model.ply
```

## 输出文件

该工具会生成以下主要输出文件：

- 裁剪后的点云文件（由`--output_ply`指定）：包含根据得分筛选后的点，颜色表示得分高低

## 工作原理

1. 加载COLMAP相机参数（内参和外参）
2. 确定场景边界（用户指定或从相机位置估计）
3. 在确定的边界内生成均匀分布的点云
4. 投影点到每个相机视角，并按以下规则评分:
   - 如果点投影到图像的前景区域（非透明区域），得分+1
   - 如果点投影到图像的背景区域（透明区域），得分-1
   - 累积所有相机视角下的得分
5. 根据累积得分，筛选出得分最高的前N%点（由`--keep_percentage`参数指定）
6. 根据得分为点云着色，保存为PLY文件

## 得分计算方法

系统通过以下步骤为每个点计算得分：
1. 将3D点投影到每个相机的图像平面上
2. 检查投影点是否落在图像范围内
3. 对于有效的投影点，检查其在图像中对应位置的透明度：
   - 非透明区域（前景）：得分+1
   - 透明区域（背景）：得分-1
4. 累加所有相机视角下的得分，得分高的点更可能是对象表面点

## 点云边界确定方式

程序自动在场景边界内生成均匀分布的点云，边界确定方式有：
- 使用`--bounds_min`和`--bounds_max`手动指定边界
- 根据相机位置估计场景边界：
  1. 提取所有相机的世界坐标位置
  2. 计算相机位置的中心点和最大距离
  3. 以中心点为基准，创建一个立方体边界框，边长为最大距离的几倍（由`--scene_scale`参数控制）

## 可视化方式

输出的点云根据得分进行着色：
- 使用蓝色通道表示得分高低
- 得分高的点颜色更深（更蓝），代表更可能是对象表面
- 得分低的点颜色更浅，代表更可能是背景

## 注意事项

- 图像必须包含透明背景（alpha通道），通常为PNG格式
- 相机参数必须是COLMAP格式
- 处理大量点或图像可能需要较长时间，可以调整`--num_workers`参数提高性能
- 默认保留得分最高的10%点，可通过`--keep_percentage`调整

## 故障排除

如果遇到问题：

1. 确保COLMAP输出格式正确
2. 检查图像是否包含透明通道
3. 尝试调整`--keep_percentage`参数以保留更多或更少的点
4. 尝试手动指定合理的边界范围
5. 如果处理速度太慢，可以减少点云数量或增加工作进程数

```bash
python point_cloud_clipper.py --data_dir /workspace/2dgs/reoutput --images_dir /workspace/2dgs/reoutput/images/ --output_ply ./output/model.ply
``` 


使用切割线来直接创建表面点云