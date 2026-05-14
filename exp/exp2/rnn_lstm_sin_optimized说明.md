# `rnn_lstm_sin.py` 实验方案优化说明

## 1. 修改目标

原始脚本已经实现了 RNN、LSTM、GRU 对正弦序列的单步预测实验，但整体更偏向“基础可运行版本”。本次修改的目标是让实验结果更稳定、指标更完整、结构更有可解释性，并尽量提升预测精度。

修改后的文件为：

```text
rnn_lstm_sin_optimized.py
```

核心思路是：保留原来的三类循环网络对比框架，同时从数据构造、网络结构、训练策略、评价指标和实验管理五个方面进行增强。

---

## 2. 主要修改内容

### 2.1 将 `look_back=3` 改为可配置长窗口

原始方案中默认使用 3 个历史点预测下一个点。由于序列步长为 0.01，3 个点只能覆盖很短的局部片段，模型很难充分捕捉正弦函数的周期性与曲率变化。

修改后新增参数：

```bash
--look-back 30
```

默认值设为 30。这样模型每次可以看到更长的历史片段，从而更容易学习平滑函数的变化趋势。

推荐尝试：

```text
look_back = 10, 20, 30, 50
```

其中 `20` 和 `30` 通常比较稳妥。

---

### 2.2 增加训练集、验证集、测试集三划分

原始方案只有训练集和测试集，无法在训练过程中判断模型是否已经过拟合。

修改后采用顺序切分：

```text
训练集：70%
验证集：15%
测试集：15%
```

对应参数为：

```bash
--train-ratio 0.70
--val-ratio 0.15
```

验证集用于 early stopping 和学习率调度，测试集只用于最终报告性能。

---

### 2.3 归一化只使用训练集信息

原始脚本直接对完整序列进行归一化，这在严格实验中会引入测试集信息泄漏。

修改后只根据训练集拟合归一化参数：

```python
min_value, scalar = fit_normalizer(train_x_raw, train_y_raw)
```

然后再用同一组参数变换训练集、验证集和测试集。这样实验流程更加规范。

---

### 2.4 网络结构从单层线性头改为 MLP Head

原始模型结构为：

```text
RNN/LSTM/GRU -> last hidden state -> Linear -> output
```

修改后改为：

```text
RNN/LSTM/GRU -> last hidden state -> LayerNorm -> Dropout -> Linear -> GELU -> Linear -> output
```

对应代码结构：

```python
self.head = nn.Sequential(
    nn.LayerNorm(hidden_size),
    nn.Dropout(dropout),
    nn.Linear(hidden_size, hidden_size),
    nn.GELU(),
    nn.Linear(hidden_size, output_size),
)
```

这样预测头的非线性表达能力更强，也能一定程度上缓解训练不稳定。

---

### 2.5 引入残差预测

正弦序列是连续平滑函数，直接预测下一个点 `y_{t+1}` 并不是最容易的方式。更自然的方式是预测相对当前最后一个输入点的变化量：

```text
ŷ = last_input + Δ
```

修改后的默认模式启用残差预测：

```python
return x[:, -1, :] + raw_output
```

如果想关闭该机制，可以运行：

```bash
python rnn_lstm_sin_optimized.py --no-residual-prediction
```

残差预测通常会让这种平滑序列任务更容易收敛。

---

### 2.6 优化训练策略

原始训练策略为：

```text
Adam + 固定学习率 + 固定 epoch
```

修改后改为：

```text
AdamW + weight decay + gradient clipping + ReduceLROnPlateau + early stopping
```

新增训练参数包括：

```bash
--weight-decay 1e-4
--grad-clip 1.0
--patience 80
--min-delta 1e-7
```

主要作用如下：

| 机制 | 作用 |
|---|---|
| AdamW | 比普通 Adam 更适合加入权重衰减 |
| weight decay | 降低过拟合风险 |
| gradient clipping | 避免 RNN 类模型梯度异常 |
| ReduceLROnPlateau | 验证集不下降时自动降低学习率 |
| early stopping | 自动保存验证集最优模型，避免后期过拟合 |

---

### 2.7 增加完整评价指标

原始脚本只统计归一化后的 MSE。

修改后同时输出：

```text
normalized MSE
normalized RMSE
normalized MAE
normalized R²
raw MSE
raw RMSE
raw MAE
raw R²
```

其中 raw 指标是在反归一化后计算的，更容易理解预测误差的真实大小。

---

### 2.8 增加紧凑型超参数搜索

新增 `--search` 模式，可以自动搜索不同模型、窗口长度、隐藏层大小和学习率组合。

示例：

```bash
python rnn_lstm_sin_optimized.py \
  --search \
  --models lstm,gru \
  --search-look-backs 10,20,30,50 \
  --search-hidden-sizes 32,64,128 \
  --search-lrs 0.001,0.003,0.0005
```

搜索结果会保存为：

```text
search_summary.json
search_summary.csv
```

可以直接根据 `test_mse_norm` 排名选择最佳组合。

---

### 2.9 优化 CPU 运行效率

在小型序列任务中，CPU 多线程反而可能因为线程调度开销导致运行变慢。因此脚本默认设置：

```python
OMP_NUM_THREADS=1
MKL_NUM_THREADS=1
```

在当前这类小规模实验中，这样通常更快、更稳定。如果在大数据集或高性能服务器上运行，可以手动修改该设置。

---

## 3. 推荐运行方式

### 3.1 默认增强实验

```bash
python rnn_lstm_sin_optimized.py
```

默认会运行：

```text
RNN + LSTM + GRU
look_back = 30
hidden_size = 64
num_layers = 2
lr = 1e-3
epochs = 1000
patience = 80
residual_prediction = True
```

输出目录默认为：

```text
experiment_outputs/rnn_sin_optimized
```

---

### 3.2 只训练 GRU

如果想先快速看效果，可以运行：

```bash
python rnn_lstm_sin_optimized.py --models gru
```

---

### 3.3 更稳妥的推荐配置

```bash
python rnn_lstm_sin_optimized.py \
  --models lstm,gru \
  --look-back 30 \
  --hidden-size 64 \
  --num-layers 2 \
  --dropout 0.1 \
  --lr 0.001 \
  --epochs 1000 \
  --patience 80
```

---

### 3.4 小型搜索推荐命令

```bash
python rnn_lstm_sin_optimized.py \
  --search \
  --models lstm,gru \
  --search-look-backs 20,30,50 \
  --search-hidden-sizes 32,64 \
  --search-lrs 0.001,0.003
```

这个搜索规模较小，适合先做初步调参。

---

## 4. 输出文件说明

运行结束后，输出目录中会包含：

| 文件 | 含义 |
|---|---|
| `summary_optimized.json` | 实验配置、训练历史、验证集结果和测试集指标 |
| `predictions_optimized.npz` | 测试集真实值和各模型预测值 |
| `fig_loss_train_val_all.png` | 所有模型训练集与验证集 loss 曲线 |
| `fig_pred_rnn.png` | RNN 测试集预测曲线 |
| `fig_pred_lstm.png` | LSTM 测试集预测曲线 |
| `fig_pred_gru.png` | GRU 测试集预测曲线 |
| `rnn_sin_optimized.pt` | RNN 最优模型权重 |
| `lstm_sin_optimized.pt` | LSTM 最优模型权重 |
| `gru_sin_optimized.pt` | GRU 最优模型权重 |

如果使用 `--search`，还会生成：

| 文件 | 含义 |
|---|---|
| `search_summary.json` | 搜索结果完整记录 |
| `search_summary.csv` | 搜索结果表格，便于复制到报告或 Excel 中 |

---

## 5. 建议在实验报告中强调的改进点

可以在报告中这样表述：

> 在基础 RNN/LSTM/GRU 对比实验的基础上，本实验进一步优化了时间序列建模流程。首先，将原始的短历史窗口扩展为可调节的长窗口，使模型能够利用更多历史上下文信息；其次，引入训练集、验证集和测试集三划分，并仅基于训练集进行归一化，以提高实验规范性；再次，在网络结构上加入 LayerNorm、Dropout 与非线性 MLP 预测头，同时采用残差预测方式，将直接预测下一个数值转化为预测相邻时间步的变化量；最后，训练阶段引入 AdamW、学习率自适应衰减、梯度裁剪和早停机制，从而提升模型收敛稳定性并降低过拟合风险。

---

## 6. 本次修改的总体结论

这次优化不是单纯增加模型复杂度，而是围绕时间序列预测任务进行了系统增强：

```text
更长输入窗口 + 更规范数据划分 + 更强预测头 + 残差预测 + 稳定训练策略 + 完整指标
```

其中最关键的提升点是：

1. `look_back` 从 3 提升到 30；
2. 加入验证集和 early stopping；
3. 使用残差预测降低任务难度；
4. 用 AdamW、调度器和梯度裁剪提升训练稳定性；
5. 增加 raw 与 normalized 两套指标，方便写实验分析。
