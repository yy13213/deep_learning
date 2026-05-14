# DCGAN MNIST 实验优化修改说明

## 1. 修改目标

本次修改基于原始 `dcgan_biggan_mnist.py`，围绕三个阶段完成优化：

1. **阶段 1：训练超参数优化**  
   提高默认训练充分性，并允许独立控制生成器和判别器的学习率。

2. **阶段 2：网络结构优化**  
   增强判别器结构，引入 SpectralNorm、Dropout，并提供更稳定的生成器上采样结构。

3. **阶段 3：损失函数优化**  
   默认由 BCE GAN 切换为更稳定的 Hinge GAN，同时保留 BCE 作为 baseline 对照。

此外，还补充了 EMA、轻量数据增强、条件生成、分类器辅助评估、训练过程可视化等更适合写实验报告和做消融实验的功能。

---

## 2. 阶段 1：超参数与训练流程修改

### 2.1 默认训练轮数增加

原始方案默认训练：

```bash
--epochs 10
```

修改后默认训练：

```bash
--epochs 50
```

原因是 MNIST DCGAN 在 10 个 epoch 下通常只能看到初步数字轮廓，训练尚未完全稳定。增加到 30～50 个 epoch 后，生成图像的轮廓、边缘和数字类别稳定性通常会明显提升。

### 2.2 噪声维度增加

原始方案：

```bash
--noise-dim 100
```

修改后默认：

```bash
--noise-dim 128
```

更高的潜变量维度可以给生成器提供更大的表示空间，有利于生成多样化数字。

### 2.3 独立控制 G/D 学习率

新增参数：

```bash
--g-lr
--d-lr
```

原始代码只使用统一学习率 `--lr`。修改后可以分别控制生成器和判别器学习率，支持 TTUR 风格训练。

推荐尝试：

```bash
python dcgan_mnist_improved.py --g-lr 2e-4 --d-lr 1e-4
```

如果判别器过强，降低 `d-lr`；如果判别器过弱，提高 `d-lr`。

### 2.4 新增训练控制参数

新增：

```bash
--beta1
--beta2
--d-steps
--grad-clip
--instance-noise
--sample-every
```

这些参数便于进行稳定性实验和消融对比。

---

## 3. 阶段 2：网络结构优化

### 3.1 生成器结构优化

原始生成器采用标准 `ConvTranspose2d` 上采样。修改后提供两种模式：

```bash
--generator-mode deconv
--generator-mode upsample
```

默认使用：

```bash
--generator-mode upsample
```

`Upsample + Conv2d` 相比单纯反卷积更不容易出现棋盘格伪影，适合提升生成数字边缘的平滑性。

### 3.2 增强判别器

新增两类判别器：

```bash
--discriminator basic
--discriminator strong
```

其中 `basic` 对应原始风格判别器，`strong` 是默认改进版。

改进版判别器结构大致为：

```text
1 × 28 × 28
→ 64 × 14 × 14
→ 128 × 7 × 7
→ 256 × 4 × 4
→ Linear / Projection head
```

### 3.3 引入 SpectralNorm

默认启用 SpectralNorm：

```bash
--discriminator strong
```

如需关闭：

```bash
--no-spectral-norm
```

SpectralNorm 可以约束判别器的 Lipschitz 性，避免判别器输出过于激进，从而提升 GAN 训练稳定性。

### 3.4 判别器 Dropout

新增：

```bash
--dropout 0.20
```

Dropout 可以缓解判别器过拟合真实 MNIST 样本，减少生成器训练早期被判别器完全压制的情况。

---

## 4. 阶段 3：损失函数优化

### 4.1 默认切换为 Hinge Loss

原始方案使用：

```python
BCEWithLogitsLoss
```

修改后默认使用：

```bash
--loss hinge
```

Hinge Loss 的核心形式为：

```python
d_loss = relu(1 - D(real)).mean() + relu(1 + D(fake)).mean()
g_loss = -D(fake).mean()
```

它比普通 BCE GAN 更适合搭配 SpectralNorm 判别器，训练通常更加稳定。

### 4.2 保留 BCE baseline

为了方便写报告和做消融实验，仍然可以切换回原始 BCE 训练：

```bash
python dcgan_mnist_improved.py --loss bce --discriminator basic --generator-mode deconv --epochs 10
```

这样可以复现实验 baseline，并与改进版对比。

---

## 5. 其他重要增强

### 5.1 EMA Generator

默认启用生成器 EMA：

```bash
--ema-decay 0.999
```

EMA 不直接改变训练梯度，而是在采样和保存时使用生成器权重的指数滑动平均版本。它通常能让生成图像更稳定、更清晰。

如需关闭：

```bash
--no-ema
```

训练结束会同时保存：

```text
generator.pt
generator_ema.pt
```

### 5.2 轻量数据增强

新增：

```bash
--augment
```

启用后对 MNIST 做轻量 `RandomAffine`，包括小角度旋转、平移和缩放。没有使用水平翻转，因为翻转会改变数字语义。

### 5.3 条件 DCGAN

新增：

```bash
--conditional
```

启用后，生成器和判别器都会使用 MNIST 标签信息。生成器输入变为：

```text
noise + label embedding
```

判别器使用 projection conditioning。这通常能改善类别一致性和模式覆盖问题。

推荐最终实验可以使用：

```bash
python dcgan_mnist_improved.py --conditional --augment --eval-classifier
```

### 5.4 分类器辅助评估

新增：

```bash
--eval-classifier
```

启用后会训练一个轻量 MNIST CNN 分类器，并用它评估生成图像：

- 生成样本的平均分类置信度
- 预测类别数量分布
- 预测类别熵
- 分类器在 MNIST 测试集上的准确率

评估结果会保存到：

```text
summary.json
```

这比只看 GAN loss 更适合实验报告，因为 GAN loss 不一定直接反映生成质量。

---

## 6. 推荐运行命令

### 6.1 改进版默认实验

```bash
python dcgan_mnist_improved.py
```

默认配置等价于：

```bash
python dcgan_mnist_improved.py \
  --epochs 50 \
  --batch-size 128 \
  --noise-dim 128 \
  --generator-mode upsample \
  --discriminator strong \
  --loss hinge
```

### 6.2 更推荐的最终版本

```bash
python dcgan_mnist_improved.py \
  --epochs 50 \
  --batch-size 128 \
  --noise-dim 128 \
  --conditional \
  --augment \
  --eval-classifier
```

### 6.3 复现原始 baseline 风格

```bash
python dcgan_mnist_improved.py \
  --epochs 10 \
  --noise-dim 100 \
  --generator-mode deconv \
  --discriminator basic \
  --loss bce \
  --no-ema \
  --instance-noise 0
```

### 6.4 判别器过强时

如果观察到 `D_loss` 很低、`G_loss` 持续变大、生成图像很差，可以尝试：

```bash
python dcgan_mnist_improved.py --g-lr 2e-4 --d-lr 1e-4
```

### 6.5 判别器过弱时

如果生成图像长期模糊、判别器区分能力不足，可以尝试：

```bash
python dcgan_mnist_improved.py --g-lr 1e-4 --d-lr 2e-4
```

### 6.6 CPU 训练卡住或过慢时

某些 PyTorch CPU 环境可能因为线程设置导致训练很慢，可以使用：

```bash
python dcgan_mnist_improved.py --torch-num-threads 1
```

---

## 7. 输出文件说明

运行后默认输出目录：

```text
experiment_outputs/dcgan_mnist_improved
```

主要输出包括：

| 文件 | 说明 |
|---|---|
| `snapshot_epoch_xxx.png` | 每隔若干 epoch 保存的生成样本图 |
| `generated_final.png` | 最终生成样本图 |
| `fig_train_loss.png` | 生成器/判别器损失曲线 |
| `fig_discriminator_logits.png` | 判别器对真实/生成样本的平均 logit 曲线 |
| `history.npz` | 训练历史数组 |
| `generator.pt` | 最终生成器权重 |
| `generator_ema.pt` | EMA 生成器权重 |
| `summary.json` | 实验配置和最终指标 |

---

## 8. 建议用于实验报告的消融表

| 实验编号 | 设置 | 目的 |
|---|---|---|
| Exp-0 | 原始 BCE + basic D + deconv G | baseline |
| Exp-1 | epoch 10 → 50 | 验证训练充分性 |
| Exp-2 | strong D + SpectralNorm | 验证判别器稳定化效果 |
| Exp-3 | Hinge Loss | 验证损失函数改进 |
| Exp-4 | Upsample Generator | 验证生成器结构改进 |
| Exp-5 | EMA | 验证采样稳定性提升 |
| Exp-6 | Conditional DCGAN | 验证类别一致性和模式覆盖提升 |
| Exp-7 | Conditional + Augment + Eval | 推荐最终版本 |

---

## 9. 总结

本次修改后的代码不只是把原始 DCGAN 参数调大，而是形成了一个更完整的实验框架：

- 可以复现原始 baseline；
- 可以逐步加入判别器增强、Hinge Loss、EMA、条件生成；
- 可以保存更丰富的训练曲线和评估指标；
- 更适合写实验报告中的“优化过程、消融实验、结果分析”。

最推荐最终配置：

```bash
python dcgan_mnist_improved.py \
  --epochs 50 \
  --batch-size 128 \
  --noise-dim 128 \
  --conditional \
  --augment \
  --eval-classifier
```
