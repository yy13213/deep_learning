# 快速开始 - RNN/LSTM/GRU 正弦波预测实验

## 前置条件

- Python 3.8+
- PyTorch 1.x+（推荐 GPU 版本，但 CPU 也可运行）
- matplotlib, numpy

## 一键启动

```bash
cd exp2

# 1. 安装依赖
pip install torch torchvision matplotlib numpy

# 2. 直接运行优化版（默认 LSTM+GRU+RNN，约几分钟完成）
python rnn_lstm_sin_optimized.py
```

运行完成后，在 `experiment_outputs/rnn_sin_optimized/` 下查看预测曲线和损失图。

## 常用命令速查

```bash
# ========== 快速实验 ==========

# 原始 Baseline（300 epoch）
python rnn_lstm_sin.py

# 优化版推荐配置（完整 RNN+LSTM+GRU 对比）
python rnn_lstm_sin_optimized.py

# ========== 常用配置 ==========

# 只训练 GRU（最快）
python rnn_lstm_sin_optimized.py --models gru

# 只训练 LSTM 和 GRU
python rnn_lstm_sin_optimized.py --models lstm,gru

# 自定义窗口长度和隐藏层大小
python rnn_lstm_sin_optimized.py --look-back 50 --hidden-size 128

# 关闭残差预测（直接预测下一个值）
python rnn_lstm_sin_optimized.py --no-residual-prediction

# ========== 超参数搜索 ==========

# 紧凑型网格搜索（遍历 look_back × hidden_size × lr × model 组合）
python rnn_lstm_sin_optimized.py --search \
  --models lstm,gru \
  --search-look-backs 20,30,50 \
  --search-hidden-sizes 32,64,128 \
  --search-lrs 0.001,0.003
```

## 参数速查

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--epochs` | 1000 | 最大训练轮数（早停可能提前） |
| `--lr` | 1e-3 | 学习率 |
| `--hidden-size` | 64 | 隐藏层大小 |
| `--num-layers` | 2 | RNN 层数 |
| `--dropout` | 0.10 | Dropout 率 |
| `--batch-size` | 32 | 批次大小 |
| `--look-back` | 30 | 输入序列长度 |
| `--patience` | 80 | Early stopping 容忍轮数 |
| `--weight-decay` | 1e-4 | 权重衰减 |
| `--grad-clip` | 1.0 | 梯度裁剪阈值 |
| `--train-ratio` | 0.70 | 训练集比例 |
| `--val-ratio` | 0.15 | 验证集比例 |
| `--models` | rnn,lstm,gru | 要训练的模型（逗号分隔） |
| `--residual-prediction` | on | 残差预测（`--no-residual-prediction` 关闭） |
| `--seed` | 42 | 随机种子 |

## 预期输出

运行优化版后，你将看到：

```
Epoch 0001 | train 0.00272352 | val 0.00259847 | lr 1.00e-03
Epoch 0050 | train 0.00000012 | val 0.00000009 | lr 1.00e-03
...
Early stopping at epoch 186. Best epoch: 106, best val MSE: 0.00000000
GRU test normalized MSE: 1.95e-06 | raw RMSE: 0.00139808 | R2: 0.999998
```

输出目录中将包含：
- 三张预测曲线图（RNN/LSTM/GRU，蓝色真实值 vs 红色预测值）
- 一张训练/验证损失对比图
- 模型权重文件 `.pt`
- `summary_optimized.json` 含完整指标

## 验证安装

```bash
python -c "import torch; print('PyTorch:', torch.__version__, '| Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```
