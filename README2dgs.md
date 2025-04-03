# 2D高斯散射（2D Gaussian Splatting）项目文档

## 一、项目概述

2D高斯散射（2D Gaussian Splatting）是一种用于几何精确辐射场表示的方法。与传统的3D高斯散射不同，该方法使用2D定向圆盘（表面元素，即surfels）来表示场景，并通过透视正确的可微分光栅化渲染这些surfels。该方法还开发了增强重建质量的正则化方法，并设计了从高斯散射中提取网格的方法。

主要特点：
- 使用2D定向圆盘（surfels）表示场景
- 透视正确的可微分光栅化
- 特殊的正则化方法提高重建质量
- 支持从高斯散射中提取网格
- 支持有界和无界场景的网格提取

## 二、2DGS算法的数学公式

### 1. 2D高斯表示

在2DGS中，每个高斯元素是一个2D定向圆盘，它由以下参数定义：

- 位置：$\mathbf{x} \in \mathbb{R}^3$
- 缩放：$\mathbf{s} \in \mathbb{R}^2$（表示圆盘的主轴长度）
- 旋转：$\mathbf{r} \in \mathbb{R}^4$（四元数，定义圆盘的方向）
- 不透明度：$\alpha \in [0,1]$
- 特征：$\mathbf{f}$（用于计算颜色）

### 2. 协方差矩阵计算

从缩放和旋转构建协方差矩阵的函数：
```
build_covariance_from_scaling_rotation(center, scaling, scaling_modifier, rotation)
```

协方差矩阵计算公式：
$$RS = \text{build\_scaling\_rotation}(\text{concat}([\text{scaling} \times \text{scaling\_modifier}, \mathbf{1}]), \text{rotation})^T$$

然后构建变换矩阵：
$$\text{trans} = \begin{bmatrix} RS & \mathbf{center} \\ \mathbf{0} & 1 \end{bmatrix}$$

### 3. 深度和法线计算

从深度图计算法线的公式：
```python
def depth_to_normal(view, depth):
    points = depths_to_points(view, depth).reshape(*depth.shape[1:], 3)
    output = torch.zeros_like(points)
    dx = torch.cat([points[2:, 1:-1] - points[:-2, 1:-1]], dim=0)
    dy = torch.cat([points[1:-1, 2:] - points[1:-1, :-2]], dim=1)
    normal_map = torch.nn.functional.normalize(torch.cross(dx, dy, dim=-1), dim=-1)
    output[1:-1, 1:-1, :] = normal_map
    return output
```

### 4. 正则化项

正则化项包括：
- 法线一致性损失：确保渲染法线与表面法线一致
  $$L_{\text{normal}} = 1 - (\text{rend\_normal} \cdot \text{surf\_normal})$$

- 深度失真损失：减少深度失真
  $$L_{\text{distortion}} = \text{rend\_dist}$$

### 5. 总体损失函数

总体损失函数为：
$$L_{\text{total}} = L_{\text{color}} + \lambda_{\text{dist}} \cdot L_{\text{distortion}} + \lambda_{\text{normal}} \cdot L_{\text{normal}}$$

其中颜色损失结合了L1损失和SSIM：
$$L_{\text{color}} = (1 - \lambda_{\text{dssim}}) \cdot L1 + \lambda_{\text{dssim}} \cdot (1 - \text{SSIM})$$

## 三、训练步骤顺序

2DGS的训练过程如下：

1. **初始化**：
   - 从点云创建初始高斯元素
   - 设置训练参数并配置优化器

2. **迭代训练**：
   - 随机选择训练集中的一个相机视角
   - 渲染该视角的图像和相关属性
   - 计算损失（颜色损失、法线一致性损失、深度失真损失）
   - 执行反向传播
   - 更新学习率
   - 更新球谐系数（每1000次迭代）
   - 执行稠密化（densification）和剪枝（pruning）：
     - 记录最大2D半径
     - 添加稠密化统计信息
     - 在特定迭代后执行稠密化和剪枝
     - 重置不透明度（在特定迭代步骤）

3. **评估与保存**：
   - 定期保存模型检查点
   - 评估测试集上的性能
   - 记录训练指标（PSNR、L1损失等）

## 四、项目中的重要代码文件

1. **主要训练脚本**：
   - `train.py`：包含主要训练循环，负责高斯元素的优化和稠密化

2. **场景表示和高斯模型**：
   - `scene/gaussian_model.py`：定义高斯模型的核心类，包含参数设置、优化和稠密化方法
   - `scene/__init__.py`：场景表示，负责加载和管理场景数据
   - `scene/cameras.py`：相机管理类，处理相机参数和变换
   - `scene/colmap_loader.py`：从COLMAP数据集加载相机和点云数据

3. **渲染器**：
   - `gaussian_renderer/__init__.py`：高斯渲染器的核心实现，使用可微分光栅化渲染场景
   - `render.py`：用于从训练后的模型渲染图像和提取网格

4. **工具函数**：
   - `utils/loss_utils.py`：损失函数的实现
   - `utils/point_utils.py`：处理点云和深度图的工具函数
   - `utils/mesh_utils.py`：网格提取和处理的工具函数

5. **评估脚本**：
   - `scripts/dtu_eval.py`：在DTU数据集上评估几何重建质量
   - `scripts/tnt_eval.py`：在TnT数据集上评估
   - `scripts/m360_eval.py`：在MipNeRF360数据集上评估

## 五、训练和使用指南

### 安装

```bash
# 下载代码
git clone https://github.com/hbb1/2d-gaussian-splatting.git --recursive

# 创建环境
conda env create --file environment.yml
conda activate surfel_splatting
```

### 训练模型

```bash
python train.py -s <COLMAP或NeRF合成数据集的路径>
```

主要命令行参数：
- `--lambda_normal`：法线一致性正则化的超参数
- `--lambda_distortion`：深度失真正则化的超参数
- `--depth_ratio`：0表示平均深度，1表示中值深度（大多数情况下0效果较好）

### 网格提取

有界网格提取：
```bash
python render.py -m <预训练模型路径> -s <COLMAP数据集路径>
```

无界网格提取：
```bash
python render.py -m <预训练模型路径> -s <COLMAP数据集路径> --unbounded --mesh_res 1024
```

## 六、项目特点和局限性

### 特点
1. 使用2D定向圆盘表示场景，比3D高斯散射更符合表面几何
2. 透视正确的可微分光栅化
3. 通过正则化增强几何重建质量
4. 支持网格提取，便于后续处理和应用
5. 支持有界和无界场景

### 局限性
1. 如果相机的主点不在图像中心，可能会遇到收敛问题
2. 有界网格提取模式需要调整`depth_trunc`参数
3. 与3DGS的查看器不完全兼容，可能出现失真伪影

## 总结

2D高斯散射是一种用于精确几何辐射场表示的新方法，通过使用2D定向圆盘和透视正确的可微分光栅化，结合特殊的正则化方法，实现了高质量的场景重建和网格提取。该方法在新视角合成和几何重建方面表现出色，为三维重建和渲染提供了一种高效的解决方案。
