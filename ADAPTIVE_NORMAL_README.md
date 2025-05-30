# 基于图像梯度的自适应法线一致性算法

## 概述

本项目实现了基于图像梯度的自适应法线一致性算法，**完全去除了原有的时间衰减机制**，采用2阶段策略：在特定迭代次数后启用自适应法线一致性算法。该算法能够自动识别图像中的平坦区域和纹理丰富/边缘区域，并对不同区域使用不同的法线一致性权重。

## 算法特点

1. **2阶段策略**：完全去除时间衰减，采用固定权重→自适应权重的2阶段切换
2. **自适应权重计算**：基于图像梯度自动计算每个像素的权重
3. **区域识别**：自动区分平坦区域和边缘/纹理区域
4. **平滑过渡**：使用高斯平滑避免权重突变
5. **自适应阈值**：使用梯度百分位数作为自适应阈值
6. **保持业务逻辑**：完全保持原有项目的业务逻辑不变
7. **持续监控**：始终计算自适应权重，便于监控算法行为

## 实现细节

### 核心函数

#### `compute_adaptive_normal_weights(image, flat_weight=0.1, edge_weight=1.0, threshold=0.1)`

计算基于图像梯度的自适应权重图。

**参数：**
- `image`: 输入图像 [C, H, W]
- `flat_weight`: 平坦区域的权重（默认：0.1）
- `edge_weight`: 边缘/纹理区域的权重（默认：1.0）
- `threshold`: 梯度阈值，用于区分平坦和边缘区域（默认：0.1）

**返回：**
- 权重图 [1, H, W]

### 算法流程

1. **图像预处理**：将RGB图像转换为灰度图像
2. **梯度计算**：使用Sobel算子计算图像梯度
3. **梯度幅值**：计算梯度的幅值
4. **高斯平滑**：减少噪声影响
5. **自适应阈值**：使用70%百分位数作为阈值
6. **权重分配**：高梯度区域使用edge_weight，低梯度区域使用flat_weight
7. **权重平滑**：再次应用高斯平滑确保权重过渡平滑

## 使用方法

### 在训练中使用

算法已经集成到 `compute_training_losses` 函数中，并且所有参数都已添加到 `arguments/__init__.py` 配置文件中：

```python
# 自适应法线一致性参数（已添加到OptimizationParams类中）
self.lambda_adaptive_normal = 0.03       # 第二阶段自适应法线损失权重
self.adaptive_normal_start_iter = 15000  # 开始使用自适应法线算法的迭代次数
self.normal_flat_weight = 0.1            # 平坦区域的权重
self.normal_edge_weight = 1.0            # 边缘/纹理区域的权重
self.normal_gradient_threshold = 0.1     # 梯度阈值，用于区分平坦和边缘区域
```

这些参数可以通过命令行参数进行调整：

```bash
python train.py --lambda_adaptive_normal 0.05 --adaptive_normal_start_iter 10000 --normal_flat_weight 0.05 --normal_edge_weight 1.5
```

为了向后兼容，如果 `adaptive_normal_start_iter` 未定义，会尝试使用 `normal_decay_start_iter`。如果 `lambda_adaptive_normal` 未定义，会使用 `lambda_normal` 的值。

### 测试算法

运行测试脚本验证算法功能：

```bash
python test_adaptive_normal.py
```

## 与原有代码的对比

### 原有实现（时间衰减）

```python
# 计算lambda_normal，在normal_decay_start_iter步之后指数衰减到0
if iteration <= opt.normal_decay_start_iter or opt.normal_decay_start_iter >= opt.iterations:
    lambda_normal = opt.lambda_normal
else:
    progress = (iteration - opt.normal_decay_start_iter) / (opt.iterations - opt.normal_decay_start_iter)
    lambda_normal = opt.lambda_normal * np.exp(-5 * progress)

# 法线损失
normal_error = (1 - (rend_normal * surf_normal).sum(dim=0))[None]
normal_loss = lambda_normal * normal_error.mean()
```

### 新实现（2阶段策略）

```python
# 2阶段法线一致性策略：完全去除时间衰减机制
adaptive_start_iter = getattr(opt, 'adaptive_normal_start_iter', getattr(opt, 'normal_decay_start_iter', 1000))

if iteration <= adaptive_start_iter:
    # 阶段1：使用固定的法线一致性权重
    lambda_normal = opt.lambda_normal
    use_adaptive_weights = False
else:
    # 阶段2：lambda_normal设为0，完全使用自适应权重
    lambda_normal = 0.0
    use_adaptive_weights = True

# 法线损失
normal_error = (1 - (rend_normal * surf_normal).sum(dim=0))[None]

if use_adaptive_weights:
    # 阶段2：使用自适应权重进行像素级调节，使用专门的权重参数
    lambda_adaptive_normal = getattr(opt, 'lambda_adaptive_normal', opt.lambda_normal)
    weighted_normal_error = normal_error * adaptive_weights
    normal_loss = lambda_adaptive_normal * weighted_normal_error.mean()
else:
    # 阶段1：使用固定权重
    normal_loss = lambda_normal * normal_error.mean()
```

## 训练阶段说明

### 阶段1：固定权重阶段（iteration ≤ adaptive_normal_start_iter）
- 使用固定的 `lambda_normal = opt.lambda_normal`
- 对所有像素使用相同的法线一致性权重
- **始终计算自适应权重用于监控**，但不应用到损失计算中
- 保持训练初期的稳定性

### 阶段2：自适应权重阶段（iteration > adaptive_normal_start_iter）
- **lambda_normal 设置为 0**：完全去除全局法线一致性权重
- **使用 lambda_adaptive_normal**：专门控制自适应法线损失的强度
- 根据图像内容动态调整权重
- 平坦区域使用较低权重，边缘区域使用较高权重
- 在法线损失计算中应用像素级自适应权重：`normal_loss = lambda_adaptive_normal * (normal_error * adaptive_weights).mean()`

## 优势

1. **完全去除衰减**：彻底移除了原有的指数衰减机制
2. **清晰的2阶段设计**：固定权重→纯自适应权重，逻辑简单明确
3. **持续监控**：始终计算自适应权重，便于监控算法行为
4. **更精确的权重控制**：根据图像内容自适应调整权重
5. **保持细节**：在纹理丰富区域保持较高的法线一致性约束
6. **减少过度约束**：在平坦区域减少不必要的法线约束
7. **数值稳定**：避免了指数衰减可能带来的数值问题
8. **参数可调**：提供多个参数供用户调节
9. **向后兼容**：支持原有的参数名称

## 测试结果示例

```
迭代次数 | 阶段      | lambda_normal | adaptive_weight | adaptive_stage | 说明
--------------------------------------------------------------------------------
     500 | 固定权重   | 0.050000     | 0.181383       |             1 | 使用固定法线一致性权重
    1000 | 固定权重   | 0.050000     | 0.181383       |             1 | 使用固定法线一致性权重
    1500 | 自适应权重 | 0.000000     | 0.181383       |             2 | 使用自适应法线一致性算法
    5000 | 自适应权重 | 0.000000     | 0.181383       |             2 | 使用自适应法线一致性算法
   15000 | 自适应权重 | 0.000000     | 0.181383       |             2 | 使用自适应法线一致性算法
```

可以看到：
- **阶段1**：`lambda_normal = 0.050000`（使用原始固定权重）
- **阶段2**：`lambda_normal = 0.000000`（完全使用自适应权重）

## 返回值扩展

在损失字典中新增了字段，用于监控自适应权重和当前阶段：

```python
return {
    # ... 其他损失 ...
    'adaptive_normal_weight': avg_adaptive_weight,
    'adaptive_stage': 2 if use_adaptive_weights else 1
}
```

- **所有阶段**：`adaptive_normal_weight` 始终为实际计算的自适应权重平均值
- **阶段1**：`adaptive_stage = 1`，权重仅用于监控，不影响损失计算
- **阶段2**：`adaptive_stage = 2`，权重同时用于监控和损失计算

这有助于在训练过程中监控算法的行为和阶段切换，同时可以观察自适应权重在整个训练过程中的变化趋势。 