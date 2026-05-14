# 快速开始

## 环境安装

```bash
# Conda (推荐)
conda env create -f environment.yml
conda activate dl-exp3

# 或 pip
pip install -r requirements-dl-exp3.txt
```

## 一键运行

```bash
# 实验 1: CNN 花卉分类 (102 类, 50 epoch)
cd exp/exp1
python cnn_flower_classification_optimized.py --data-dir data/flowers

# 实验 2: RNN 正弦波预测 (RNN/LSTM/GRU, ~3 min)
cd exp/exp2
python rnn_lstm_sin_optimized.py

# 实验 3: DCGAN 数字生成 (50 epoch, ~3 min)
cd exp/exp3
python dcgan_mnist_improved.py
```

## 快速实验命令速查

### 实验 1: CNN 花卉分类

```bash
cd exp/exp1

# Baseline (28x28, 2 卷积层, 30 epoch)
python cnn_flower_classification.py --data-dir data/flowers

# Optimized (224x224, 5 卷积块, 50 epoch)
python cnn_flower_classification_optimized.py --data-dir data/flowers

# Residual (ResNet 残差块, 50 epoch)
python cnn_flower_classification_residual.py --data-dir data/flowers

# Smoke test (随机输入验证模型)
python cnn_flower_classification_optimized.py --smoke-test
```

### 实验 2: RNN 正弦波预测

```bash
cd exp/exp2

# Baseline (look_back=3, 单层 Linear)
python rnn_lstm_sin.py

# Optimized (look_back=30, MLP Head, AdamW)
python rnn_lstm_sin_optimized.py

# 只训练 GRU (最快)
python rnn_lstm_sin_optimized.py --models gru

# 超参搜索
python rnn_lstm_sin_optimized.py --search
```

### 实验 3: DCGAN 数字生成

```bash
cd exp/exp3

# Baseline (10 epoch, BCE)
python dcgan_biggan_mnist.py

# Optimized (50 epoch, Hinge + Strong D)
python dcgan_mnist_improved.py

# 完整功能 (条件生成 + 增强 + 分类器评估)
python dcgan_mnist_improved.py --conditional --augment --eval-classifier

# 复现实验任务书 Baseline
python dcgan_mnist_improved.py --generator-mode deconv --loss bce --discriminator basic --epochs 10 --no-ema
```

## 关键参数速查

### 实验 1 (CNN)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--epochs` | 30 → 50 | 训练轮数 |
| `--batch-size` | 20 → 32 | 批次大小 |
| `--learning-rate` | 1e-3 → 3e-4 | 学习率 |
| `--dropout` | - | Dropout 率 (优化版) |
| `--weight-decay` | - | 权重衰减 |

### 实验 2 (RNN)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--epochs` | 300 → 1000 | 最大训练轮数 (早停) |
| `--lr` | 1e-2 → 1e-3 | 学习率 |
| `--hidden-size` | 20 → 64 | 隐藏层大小 |
| `--look-back` | 3 → 30 | 输入序列长度 |
| `--models` | rnn,lstm,gru | 训练的模型 |

### 实验 3 (DCGAN)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--epochs` | 10 → 50 | 训练轮数 |
| `--batch-size` | 128 | 批次大小 |
| `--noise-dim` | 100 → 128 | 噪声向量维度 |
| `--generator-mode` | deconv → upsample | 上采样方式 |
| `--discriminator` | basic → strong | 判别器类型 |
| `--loss` | bce → hinge | 损失函数 |
| `--conditional` | off | 条件生成 |

## 输出位置

| 实验 | 结果目录 |
|------|---------|
| 实验 1 | `exp/exp1/experiment_outputs/` |
| 实验 2 | `exp/exp2/experiment_outputs/` |
| 实验 3 | `exp/exp3/experiment_outputs/` |

每个目录包含：预测/生成曲线图、损失曲线、模型权重 `.pt`、`summary.json`。
