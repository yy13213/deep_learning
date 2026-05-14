# 深度学习实验

本仓库包含三个深度学习实验，涵盖 CNN 图像分类、RNN 时间序列预测、DCGAN 图像生成。

## 实验总览

| 实验 | 主题 | 框架 | 数据集 | 模型数 | 关键结果 |
|------|------|------|--------|--------|---------|
| [实验 1](./exp/exp1/README.md) | CNN 图像分类 | PyTorch | Flower Recognition (102 类) | 3 | Baseline 40% → Optimized 71% → Residual 76% |
| [实验 2](./exp/exp2/README.md) | RNN 时间序列预测 | PyTorch | 正弦波 sin(pi*x) | 3 | LSTM 最优 MSE 3.1e-7, R2 > 0.99999 |
| [实验 3](./exp/exp3/README.md) | GAN 图像生成 | PyTorch | MNIST (10 类数字) | 2 | 从噪声到清晰数字，50 epoch 收敛 |

## 项目结构

```
deep_learning/
├── exp/
│   ├── exp1/          # 实验 1: CNN 花卉分类
│   │   ├── cnn_flower_classification.py              # Baseline (2 卷积层, 28x28)
│   │   ├── cnn_flower_classification_optimized.py    # 优化版 (5 卷积块, 224x224)
│   │   ├── cnn_flower_classification_residual.py     # 残差版 (ResNet 风格)
│   │   └── experiment_outputs/
│   ├── exp2/          # 实验 2: RNN 正弦波预测
│   │   ├── rnn_lstm_sin.py                           # Baseline (RNN/LSTM/GRU)
│   │   ├── rnn_lstm_sin_optimized.py                 # 优化版 (AdamW + 残差预测)
│   │   └── experiment_outputs/
│   └── exp3/          # 实验 3: DCGAN 数字生成
│       ├── dcgan_biggan_mnist.py                     # Baseline DCGAN
│       ├── dcgan_mnist_improved.py                   # 优化版 (Hinge + Strong D)
│       └── experiment_outputs/
├── model/              # 训练保存的模型权重
├── data/               # 数据集 (git 忽略)
├── experiment_outputs/ # 实验结果汇总
├── environment.yml     # Conda 环境配置
├── requirements-dl-exp3.txt  # pip 依赖
└── setup_dl_exp3_env.sh  # 环境安装脚本
```

## 快速开始

### 环境安装

```bash
# Conda (推荐)
conda env create -f environment.yml
conda activate dl-exp3

# 或 pip
pip install -r requirements-dl-exp3.txt
```

### 一键运行各实验

```bash
# 实验 1: CNN 花卉分类
cd exp/exp1
python cnn_flower_classification_optimized.py --data-dir data/flowers

# 实验 2: RNN 正弦波预测
cd exp/exp2
python rnn_lstm_sin_optimized.py

# 实验 3: DCGAN 数字生成
cd exp/exp3
python dcgan_mnist_improved.py
```

## 各实验详细说明

- [实验 1: CNN 花卉分类](./exp/exp1/README.md) — 从简单 CNN 到残差网络的演进，数据增强、BatchNorm、Dropout、AdamW 等训练技巧的系统引入
- [实验 2: RNN 时间序列预测](./exp/exp2/README.md) — RNN/LSTM/GRU 对正弦波的对比预测，长窗口、残差预测、早停、超参搜索
- [实验 3: GAN 图像生成](./exp/exp3/README.md) — DCGAN 生成 MNIST 数字，Hinge Loss、SpectralNorm、EMA、条件生成、实例噪声
