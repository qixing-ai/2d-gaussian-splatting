### 2. 深度收敛损失（Depth Convergence Loss）
#### 2.1 动机
光滑表面上的高光区域由于反射不连续性，会导致优化过程生成偏离真实表面的高光Gaussian，进而在重建中产生孔洞。原始2DGS的深度畸变损失以不透明度作为权重，高不透明度的Gaussian主导损失计算，而位于真实表面的低不透明度Gaussian被忽视，无法有效优化。

#### 2.2 公式
深度收敛损失的定义为：
\[
\mathcal{L}_{\text{converge}} = \sum_{i=2}^{n} \min(\hat{\mathcal{G}}_i(\mathbf{x}), \hat{\mathcal{G}}_{i-1}(\mathbf{x})) \cdot D_i
\]
其中：
- \( D_i = (d_i - d_{i-1})^2 \)：相邻两个交点的深度差平方。
- \( \hat{\mathcal{G}}_i(\mathbf{x}) \)：第 \( i \) 个Gaussian的2D Gaussian值，作为权重，不参与梯度计算。

#### 2.3 特点
- **去除不透明度影响**：不同于原始方法使用不透明度权重，深度收敛损失采用Gaussian值 \(\hat{\mathcal{G}}_i(\mathbf{x})\)，确保低不透明度Gaussian也能公平参与优化。
- **平滑深度分布**：损失函数类似于Dirichlet能量最小化，鼓励Gaussian深度连续且平滑，自然收敛到真实表面。
- **梯度缩放**：计算梯度时，对 \( D_i \) 关于 \( d_i \) 的偏导数进行缩放：
  \[
  \frac{\partial D_i}{\partial d_i} \leftarrow 2k(d_i - d_{i-1}), \quad k=1.25
  \]
  这增强了高光区域Gaussian的优化力度，加速其向表面移动。

#### 2.4 效果
- **填补孔洞**：通过“边缘生长”现象，高光区域的Gaussian从孔洞边缘向真实表面逐步移动，填补缺失区域。
- **触发机制**：较大的梯度可能触发Gaussian的splitting（分裂）和cloning（克隆），生成更多Gaussian进一步完善重建。
