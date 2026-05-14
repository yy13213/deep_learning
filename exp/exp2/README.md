# 实验 2：循环神经网络 RNN 实验

基于 RNN / LSTM / GRU 对正弦波时间序列进行预测的实验。

## 实验目的

- 学习掌握循环神经网络（RNN）的基本原理及 LSTM 的基本结构
- 掌握利用 LSTM 神经元构造循环神经网络进行训练和预测时间序列
- 对比分析 RNN、LSTM 和 GRU 三种模型

## 实验环境

- **硬件**：x86_64 Centos 3.10.0 服务器/GPU 服务器/PC
- **软件**：Python 3.5+、PyTorch 1.0+、NumPy 1.12.1+

## 实验原理

RNN 是一种用于处理时序数据的神经网络模型，通过携带指向自身的循环结构传递当前时刻的信息给下一时刻。其展开结构如图所示：

```
输入 xt → [处理单元 A] → 输出 ht
                ↑
                └── 上一时刻隐藏状态 ht-1
```

RNN 在预测点与依赖信息距离较远时难以学习，因此引入了 LSTM（Long Short-Term Memory）。LSTM 的核心是 memory block（记忆块），包含三个门（forget gate、input gate、output gate）和一个记忆单元（cell state）。

## 项目结构

```
exp2/
├── rnn_lstm_sin.py                    # 原始 RNN 实现（实验任务书 baseline）
├── rnn_lstm_sin_optimized.py          # 三阶段优化版
├── rnn_lstm_sin_optimized说明.md      # 优化修改说明
├── requirements.txt                   # 依赖列表
└── experiment_outputs/                # 训练输出
    ├── rnn_sin_py/                    # 原始 baseline 结果
    └── rnn_sin_optimized/             # 优化版结果
```

## 原始实现 vs 优化版

`rnn_lstm_sin.py` 是实验任务书定义的基础实现，`rnn_lstm_sin_optimized.py` 围绕八个方面进行了系统增强：

| 优化维度 | 原始方案 | 优化方案 |
|---------|---------|---------|
| 输入窗口 | look_back = 3 | look_back = 30（可配置） |
| 数据划分 | 训练集/测试集 | 训练集(70%)/验证集(15%)/测试集(15%) |
| 归一化 | 全序列归一化 | 仅训练集拟合参数 |
| 预测头 | RNN → Linear | RNN → LayerNorm → Dropout → MLP(GELU) |
| 预测方式 | 直接预测 y_t+1 | 残差预测 y_hat = x_t + Δ |
| 优化器 | Adam | AdamW + weight decay + 梯度裁剪 + LR 调度 + Early Stopping |
| 评价指标 | MSE | MSE/RMSE/MAE/R²（含反归一化 raw 指标） |
| 超参数搜索 | 无 | 内置紧凑型网格搜索 |

## 模型架构

优化版模型结构：

```
SequenceRegressor:
  ┌─────────────────────┐
  │  RNN / LSTM / GRU    │  (num_layers=2, dropout=0.10)
  └─────────┬───────────┘
            │ last hidden state (hidden_size=64)
            ▼
  ┌─────────────────────┐
  │  LayerNorm           │
  │  Dropout             │
  │  Linear → GELU       │
  │  Linear              │
  └─────────┬───────────┘
            │
  残差预测:  output = last_input + head_output
```

## 训练结果

### 测试集预测对比

**原始 Baseline（look_back=3, 单层 Linear）：** LSTM 和 GRU 都能较好拟合正弦曲线，但窗口过短限制了长期预测能力。

![原始 Baseline LSTM 预测](../experiment_outputs/rnn_sin_py/fig_pred_lstm.png)

**优化版（look_back=30, MLP Head + 残差预测 + AdamW）：** LSTM 和 GRU 预测曲线与真实值高度重合，预测精度显著提升。

![优化版 LSTM 预测](../experiment_outputs/rnn_sin_optimized/fig_pred_lstm.png)
![优化版 GRU 预测](../experiment_outputs/rnn_sin_optimized/fig_pred_gru.png)

### 训练损失曲线

优化版训练与验证损失快速收敛，三种模型均在 50 个 epoch 内降至接近 0，无过拟合：

![训练和验证损失曲线](../experiment_outputs/rnn_sin_optimized/fig_loss_train_val_all.png)

### 模型性能对比

| 模型 | 测试 MSE (normalized) | 测试 RMSE (raw) |
|------|----------------------|-----------------|
| RNN  | ~1.1e-4 | ~0.010 |
| LSTM | ~3.1e-7 | ~0.0006 |
| GRU  | ~2.0e-6 | ~0.0014 |

LSTM 在 sine-wave 预测任务上表现最佳，GRU 次之，RNN 因梯度消散问题精度最低。

## 快速开始

### 环境准备

```bash
# 使用 pip
pip install torch torchvision matplotlib numpy

# 或查看 requirements.txt
pip install -r requirements.txt
```

### 运行原始 Baseline

```bash
python rnn_lstm_sin.py \
  --epochs 300 \
  --lr 1e-2 \
  --hidden-size 20 \
  --num-layers 2 \
  --batch-size 32 \
  --seed 42
```

### 运行优化版（推荐）

```bash
python rnn_lstm_sin_optimized.py
```

等价于：

```bash
python rnn_lstm_sin_optimized.py \
  --epochs 1000 \
  --lr 1e-3 \
  --hidden-size 64 \
  --num-layers 2 \
  --look-back 30 \
  --dropout 0.10 \
  --patience 80
```

### 只训练 GRU（快速验证）

```bash
python rnn_lstm_sin_optimized.py --models gru
```

### 超参数搜索

```bash
python rnn_lstm_sin_optimized.py \
  --search \
  --models lstm,gru \
  --search-look-backs 20,30,50 \
  --search-hidden-sizes 32,64,128 \
  --search-lrs 0.001,0.003,0.0005
```

## 输出文件说明

| 文件 | 说明 |
|------|------|
| `fig_pred_rnn.png` | RNN 测试集预测曲线（蓝色=真实值，红色=预测值） |
| `fig_pred_lstm.png` | LSTM 测试集预测曲线 |
| `fig_pred_gru.png` | GRU 测试集预测曲线 |
| `fig_loss_train_val_all.png` | 所有模型训练/验证损失曲线 |
| `predictions_optimized.npz` | 测试集真实值和各模型预测值 |
| `rnn_sin_optimized.pt` | RNN 最优模型权重 |
| `lstm_sin_optimized.pt` | LSTM 最优模型权重 |
| `gru_sin_optimized.pt` | GRU 最优模型权重 |
| `summary_optimized.json` | 实验配置、训练历史和测试指标 |
| `search_summary.json` | 超参数搜索结果（使用 `--search` 时生成） |
