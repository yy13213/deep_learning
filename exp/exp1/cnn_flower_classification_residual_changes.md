# CNN 花卉分类实验进一步优化说明：加入残差连接

## 1. 修改目标

本次修改基于上一版 `cnn_flower_classification_optimized.py` 继续优化，不改变已经完成的核心改动，包括：

- 输入尺寸保持 `224×224`；
- 训练集和验证集按类别随机打乱后分层切分；
- 训练阶段保留数据增强；
- 保留 BatchNorm、Dropout、AdamW、label smoothing、梯度裁剪、余弦退火学习率调度；
- 保留 best checkpoint、混淆矩阵、每类准确率与 summary 输出。

本次新增的重点是：**在 CNN 主干中加入残差连接，将普通堆叠卷积结构升级为 Residual-CNN 结构**。

---

## 2. 为什么要加入残差连接

上一版 CNN 已经比原始浅层网络更深，但更深的卷积网络在训练时可能出现以下问题：

1. 梯度传播路径变长，训练不够稳定；
2. 网络加深后不一定带来更高准确率，甚至可能出现退化；
3. 中低层纹理、边缘特征在多层卷积后可能被削弱；
4. 小数据集上深层网络更容易过拟合。

残差连接通过引入 shortcut，让网络学习：

```text
输出 = 卷积变换(x) + shortcut(x)
```

这样模型不必每一层都重新学习完整映射，而是学习相对于输入特征的“增量修正”。这通常可以让深层 CNN 更容易训练，也更适合 224×224 图像分类任务。

---

## 3. 新增 ResidualBlock

新增基础残差块：

```python
class ResidualBlock(nn.Module):
    ...
```

每个残差块主体结构为：

```text
Conv2d
BatchNorm2d
ReLU
Conv2d
BatchNorm2d
Dropout2d
+ shortcut
ReLU
```

其中 shortcut 分为两种情况：

### 3.1 输入输出维度一致

如果输入通道数、输出通道数和特征图尺寸一致，则直接使用恒等映射：

```python
self.shortcut = nn.Identity()
```

这类残差连接几乎不增加参数量，但可以明显改善梯度传播。

### 3.2 输入输出维度不一致或需要下采样

如果通道数变化，或者 `stride=2` 进行下采样，则使用 `1×1` 卷积匹配维度：

```python
self.shortcut = nn.Sequential(
    nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
    nn.BatchNorm2d(out_channels),
)
```

这样可以保证主分支和 shortcut 分支的张量形状一致，从而可以相加。

---

## 4. CNN 主干结构修改

上一版普通 CNN 的特征提取方式主要是：

```text
Conv + BN + ReLU + MaxPool
Conv + BN + ReLU + MaxPool
...
```

本次改为 Residual-CNN：

```text
Stem:
Conv 3→32 + BatchNorm + ReLU + MaxPool

Residual stages:
ResidualBlock 32→32, stride=1
ResidualBlock 32→64, stride=2
ResidualBlock 64→64, stride=1
ResidualBlock 64→128, stride=2
ResidualBlock 128→128, stride=1
ResidualBlock 128→256, stride=2
ResidualBlock 256→256, stride=1
ResidualBlock 256→256, stride=2

Classifier:
Flatten
Dropout
Linear(256×7×7 → 512)
BatchNorm1d
ReLU
Dropout
Linear(512 → n_classes)
```

---

## 5. 输入输出尺寸适配说明

输入图像尺寸仍然是：

```text
224×224
```

特征图尺寸变化如下：

```text
输入图像:      224×224
Stem 池化后:   112×112
Stage 1:       112×112
Stage 2:        56×56
Stage 3:        28×28
Stage 4:        14×14
Stage 5:         7×7
```

最终通道数为 256，因此全连接层输入维度仍然是：

```python
fc_in_features = 256 * 7 * 7
```

这保证了模型可以直接适配 `224×224` 输入，不会出现维度不匹配问题。

---

## 6. 保留并增强的正则化设计

本次残差版本仍然保留上一版中的正则化策略：

- 卷积层使用 `BatchNorm2d`；
- 残差块中使用 `Dropout2d`；
- 分类器中使用 `Dropout`；
- 分类器中额外加入 `BatchNorm1d(512)`；
- 损失函数使用 `label_smoothing`；
- 优化器使用 `AdamW + weight_decay`。

这些设计的目标是：在增加网络表达能力的同时，尽量控制过拟合风险。

---

## 7. 输出文件与保存路径变化

为了避免覆盖上一版普通 CNN 的实验结果，本次残差版本修改了模型和实验输出路径。

最佳模型保存为：

```text
model/cnn_residual_best.pt
```

最后一轮模型保存为：

```text
model/cnn_residual_last.pt
```

实验输出目录为：

```text
experiment_outputs/cnn_flower_residual/
```

其中包括：

```text
fig_residual_loss_acc.png
fig_best_confusion_matrix.png
best_confusion_matrix.csv
summary.json
```

---

## 8. 推荐运行方式

基础训练：

```bash
python cnn_flower_classification_residual.py \
  --data-dir data/flowers \
  --epochs 30 \
  --batch-size 32 \
  --learning-rate 3e-4
```

如果显存不足，可以降低 batch size：

```bash
python cnn_flower_classification_residual.py \
  --data-dir data/flowers \
  --epochs 30 \
  --batch-size 16
```

如果使用 zip 数据集：

```bash
python cnn_flower_classification_residual.py \
  --zip-path flowers.zip \
  --data-dir data/flowers \
  --epochs 30 \
  --batch-size 32
```

快速检查模型前向和反向传播：

```bash
python cnn_flower_classification_residual.py --smoke-test
```

---

## 9. 与上一版相比的主要变化

| 对比项 | 上一版 CNN | 本次 Residual-CNN |
|---|---|---|
| 输入尺寸 | 224×224 | 224×224 |
| 数据增强 | 有 | 有 |
| BatchNorm | 有 | 有 |
| Dropout | 有 | 有 |
| 残差连接 | 无 | 有 |
| 下采样方式 | MaxPool 为主 | stride=2 残差块 + stem MaxPool |
| 主干结构 | 普通卷积堆叠 | 残差块堆叠 |
| 全连接输入 | 256×7×7 | 256×7×7 |
| best 模型 | cnn_optimized_best.pt | cnn_residual_best.pt |
| 输出目录 | cnn_flower_optimized | cnn_flower_residual |

---

## 10. 预期效果

加入残差连接后，预期带来的改进包括：

1. 梯度传播更顺畅，训练更稳定；
2. 允许网络在不明显退化的情况下加深；
3. 有助于保留浅层纹理、边缘和颜色信息；
4. 对 224×224 花卉图像的细节建模能力更强；
5. 在数据增强、Dropout、AdamW 配合下，有望获得比普通 CNN 更好的验证集表现。

---

## 11. 后续实验建议

建议在实验报告中做如下对比：

```text
原始 CNN
→ 224×224 + 数据增强 + BatchNorm/Dropout CNN
→ Residual-CNN
→ 预训练 ResNet18 / MobileNetV3 / EfficientNet-B0
```

其中前三个版本可以体现你对 CNN 结构逐步优化的过程，最后一个版本可以作为进一步提升结果的强基线。
