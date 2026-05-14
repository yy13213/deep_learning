# 快速开始 - DCGAN MNIST 实验

## 前置条件

- Python 3.8+
- PyTorch 1.x+（推荐 GPU 版本）
- torchvision, matplotlib, numpy

## 一键启动

```bash
cd exp3

# 1. 安装依赖（三选一）
conda env create -f ../environment.yml
conda activate deep_learning
# 或
pip install torch torchvision matplotlib numpy
# 或
pip install -r ../requirements-dl-exp3.txt

# 2. 直接运行优化版（默认 50 epoch，约几分钟完成）
python dcgan_mnist_improved.py
```

运行完成后，在 `experiment_outputs/dcgan_mnist_improved/` 下查看结果。

## 常用命令速查

```bash
# ========== 快速实验（5 分钟） ==========

# 原始 Baseline（10 epoch）
python dcgan_biggan_mnist.py

# 优化版推荐配置（50 epoch）
python dcgan_mnist_improved.py

# ========== 高级配置 ==========

# 条件生成 + 数据增强 + 分类器评估
python dcgan_mnist_improved.py --conditional --augment --eval-classifier

# 使用反卷积生成器 + BCE 损失（复现原始 DCGAN）
python dcgan_mnist_improved.py --generator-mode deconv --loss bce --discriminator basic --epochs 10 --no-ema

# 自定义学习率（G 学习率高于 D）
python dcgan_mnist_improved.py --g-lr 2e-4 --d-lr 1e-4

# 使用 CPU 训练（线程受限）
python dcgan_mnist_improved.py --torch-num-threads 1
```

## 参数速查

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--epochs` | 50 | 训练轮数 |
| `--batch-size` | 128 | 批次大小 |
| `--noise-dim` | 128 | 噪声向量维度 |
| `--g-lr` | 2e-4 | 生成器学习率 |
| `--d-lr` | 2e-4 | 判别器学习率 |
| `--generator-mode` | upsample | 生成器模式（deconv/upsample） |
| `--discriminator` | strong | 判别器类型（basic/strong） |
| `--loss` | hinge | 损失函数（hinge/bce） |
| `--conditional` | off | 启用条件生成 |
| `--augment` | off | 启用数据增强 |
| `--eval-classifier` | off | 启用分类器评估 |
| `--ema-decay` | 0.999 | EMA 衰减系数（`--no-ema` 关闭） |
| `--instance-noise` | 0.03 | 实例噪声强度 |
| `--dropout` | 0.20 | 判别器 Dropout 率 |

## 预期输出

运行优化版 50 epoch 后，生成结果应如下：

- **最终生成图像**：清晰的 0-9 手写数字
- **训练损失**：G_loss 收敛至 ~0.06，D_loss 收敛至 ~1.94
- **判别器 logits**：D(real) 和 D(fake) 差距逐渐缩小

## 验证安装

```bash
python -c "import torch; print('PyTorch:', torch.__version__, '| Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```
