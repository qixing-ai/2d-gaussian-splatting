# 自适应法线一致性损失

## 概述

自适应法线一致性损失是对原始法线损失的改进，它根据图像内容的复杂度自动调整损失权重：
- **平坦区域**：使用强权重，确保几何一致性
- **纹理丰富区域**：使用弱权重，保留细节特征

## 核心原理

### 1. 平坦度检测
使用Sobel算子计算图像梯度，通过梯度幅值判断区域的纹理复杂度：
- 低梯度 → 平坦区域 → 高权重
- 高梯度 → 纹理区域 → 低权重

### 2. 权重计算公式
```
weight_map = flat_weight * exp(-3.0 * gradient_norm) + texture_weight
```

其中：
- `gradient_norm`: 归一化的梯度幅值 [0,1]
- `flat_weight`: 平坦区域权重（默认2.0）
- `texture_weight`: 纹理区域权重（默认0.5）

### 3. 自适应损失
```
normal_error = (1 - (rend_normal * surf_normal).sum(dim=0))
weighted_normal_error = normal_error * flatness_weight
normal_loss = lambda_normal * weighted_normal_error.mean()
```

## 参数配置

在 `arguments/__init__.py` 中的 `OptimizationParams` 类中添加了以下参数：

```python
# 自适应法线损失参数
self.use_adaptive_normal = True  # 是否使用自适应法线损失
self.normal_flat_weight = 2.0    # 平坦区域的法线损失权重
self.normal_texture_weight = 0.5 # 纹理区域的法线损失权重
```

## 使用方法

### 1. 启用自适应法线损失（默认启用）
```bash
python train.py -s <scene_path> --use_adaptive_normal
```

### 2. 禁用自适应法线损失（使用原始方法）
```bash
python train.py -s <scene_path> --use_adaptive_normal False
```

### 3. 自定义权重参数
```bash
python train.py -s <scene_path> \
    --normal_flat_weight 3.0 \
    --normal_texture_weight 0.3
```

## 参数调优建议

### 权重设置指南

| 场景类型 | flat_weight | texture_weight | 说明 |
|---------|-------------|----------------|------|
| 建筑物/室内 | 2.0-3.0 | 0.3-0.5 | 强调几何一致性 |
| 自然场景 | 1.5-2.0 | 0.5-0.8 | 平衡几何与细节 |
| 纹理丰富 | 1.0-1.5 | 0.8-1.0 | 保留更多细节 |

### 调优步骤

1. **初始测试**：使用默认参数 `flat_weight=2.0, texture_weight=0.5`
2. **观察效果**：
   - 如果平坦区域几何不够平滑 → 增加 `flat_weight`
   - 如果纹理细节丢失 → 增加 `texture_weight`
3. **精细调整**：根据具体场景特点微调参数

## 测试验证

运行测试脚本验证功能：
```bash
python test_adaptive_normal.py
```

这将生成可视化图像，展示：
- 权重分布效果
- 不同参数设置的对比
- 统计信息

## 技术细节

### 梯度计算
- 使用Sobel算子计算图像梯度
- 应用高斯滤波平滑梯度图
- 归一化到[0,1]范围

### 权重映射
- 使用指数函数确保平滑过渡
- 避免权重突变导致的训练不稳定

### 内存优化
- 梯度计算使用反射填充避免边界效应
- 权重图与法线误差逐像素相乘，内存高效

## 预期效果

使用自适应法线损失后，您应该观察到：

1. **平坦区域**：更好的几何一致性，减少噪声
2. **纹理区域**：保留更多细节特征
3. **整体质量**：更平衡的重建效果
4. **训练稳定性**：权重平滑过渡，训练更稳定

## 故障排除

### 常见问题

1. **权重图异常**
   - 检查输入图像格式是否正确
   - 确保CUDA设备可用

2. **效果不明显**
   - 尝试增大 `flat_weight` 和 `texture_weight` 的差值
   - 检查 `lambda_normal` 基础权重是否合适

3. **训练不稳定**
   - 降低权重差值，使过渡更平滑
   - 检查其他损失项的权重平衡

### 调试技巧

1. 使用测试脚本可视化权重分布
2. 在TensorBoard中监控法线损失变化
3. 对比启用/禁用自适应损失的训练曲线 