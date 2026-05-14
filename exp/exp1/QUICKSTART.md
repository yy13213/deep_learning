# 快速开始 - CNN 花卉分类实验

## 前置条件

- Python 3.8+
- PyTorch 1.x+（推荐 GPU 版本）
- torchvision, matplotlib, numpy, Pillow

## 一键启动

```bash
cd exp1

# 1. 安装依赖
pip install torch torchvision matplotlib numpy Pillow

# 2. 确保数据集位于 data/flowers/（按类别编号的子目录）

# 3. 运行优化版
python cnn_flower_classification_optimized.py \
  --data-dir data/flowers \
  --epochs 50 \
  --batch-size 32
```

## 常用命令速查

```bash
# ========== Baseline（快速验证） ==========

# 原始 2 卷积层 CNN，28x28 输入，30 epoch
python cnn_flower_classification.py --data-dir data/flowers

# ========== 优化版 ==========

# 5 卷积块 + BatchNorm + Dropout + 数据增强 + AdamW
python cnn_flower_classification_optimized.py --data-dir data/flowers

# 自定义训练轮数和批次大小
python cnn_flower_classification_optimized.py \
  --data-dir data/flowers \
  --epochs 80 \
  --batch-size 64

# 关闭学习率调度器
python cnn_flower_classification_optimized.py --no-scheduler

# 只跑 smoke test（随机输入验证模型结构）
python cnn_flower_classification_optimized.py --smoke-test

# ========== 残差版 ==========

# ResNet 风格残差块，默认 50 epoch
python cnn_flower_classification_residual.py --data-dir data/flowers

# ========== zip 数据集 ==========

# 从 zip 包解压并训练
python cnn_flower_classification_optimized.py \
  --zip-path flowers.zip \
  --extract-to data/
```

## 参数速查

| 参数 | Baseline | Optimized | Residual | 说明 |
|------|----------|-----------|----------|------|
| `--data-dir` | data/flowers | data/flowers | data/flowers | 数据集目录 |
| `--epochs` | 30 | 50 | 50 | 训练轮数 |
| `--batch-size` | 20 | 32 | 32 | 批次大小 |
| `--learning-rate` | 1e-3 | 3e-4 | 3e-4 | 学习率 |
| `--dropout` | - | 0.4 | 0.4 | Dropout 率 |
| `--weight-decay` | - | 1e-4 | 1e-4 | 权重衰减 |
| `--label-smoothing` | - | 0.1 | 0.1 | 标签平滑 |
| `--grad-clip` | - | 1.0 | 1.0 | 梯度裁剪 |
| `--no-scheduler` | - | off | off | 关闭 LR 调度 |
| `--smoke-test` | on | on | on | 结构验证 |

## 预期输出

运行优化版 50 epoch 后：

```
Epoch 50/50 | lr 1.23e-05 | train loss 1.8943 acc 0.6523 |
             val loss 1.8312 acc 0.7087 | best acc 0.7087@45
saved to /path/to/experiment_outputs/cnn_flower_optimized
```

输出目录包含：
- 损失/准确率曲线图
- 混淆矩阵热力图 + CSV
- 最佳模型权重 `cnn_optimized_best.pt`
- `summary.json`（含 102 类每类准确率）

## 常见问题

**Q: 显存不足？**
```bash
python cnn_flower_classification_optimized.py --batch-size 16
```

**Q: 数据集在 zip 包里？**
```bash
python cnn_flower_classification_optimized.py --zip-path flowers.zip
```

**Q: 只想跑 smoke test 验证环境？**
```bash
python cnn_flower_classification_optimized.py --smoke-test
```
