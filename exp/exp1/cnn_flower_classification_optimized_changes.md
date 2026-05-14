# CNN 花卉分类实验优化说明

## 1. 修改目标

本次修改基于原始 `cnn_flower_classification.py` 脚本，围绕提升花卉图像分类实验效果进行优化，主要从图像输入尺寸、数据划分、数据增强、CNN 网络结构、训练超参数、模型保存与结果分析几个方面进行改进。

---

## 2. 核心修改内容

### 2.1 输入尺寸由 28×28 修改为 224×224

原始脚本中使用：

```python
transforms.Resize((28, 28))
```

该尺寸过小，容易丢失花瓣纹理、边缘结构、颜色分布等关键信息。修改后统一使用：

```python
IMAGE_SIZE = 224
transforms.Resize((IMAGE_SIZE, IMAGE_SIZE))
```

这样可以保留更多图像细节，更适合花卉分类任务。

---

### 2.2 区分训练集与验证集 transform

原始脚本中训练集和验证集使用同一套 transform。修改后拆分为：

```python
TRAIN_TRANSFORMER_IMAGE
VAL_TRANSFORMER_IMAGE
```

其中：

- 训练集使用数据增强，提高泛化能力；
- 验证集只做确定性预处理，保证评估结果稳定。

---

### 2.3 训练阶段加入数据增强

训练集新增以下增强操作：

```python
transforms.RandomHorizontalFlip(p=0.5)
transforms.RandomRotation(degrees=20)
transforms.ColorJitter(...)
transforms.RandomAffine(...)
transforms.RandomErasing(...)
```

这些增强可以模拟花卉图像在真实采集过程中的角度、位置、颜色和局部遮挡变化，有助于降低过拟合。

---

### 2.4 归一化方式修改为 ImageNet 统计量

原始脚本使用：

```python
Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
```

修改后使用更常见的 ImageNet 均值和方差：

```python
Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)
```

这样做的好处是：

1. 更适合自然图像任务；
2. 后续如果切换到预训练 ResNet、EfficientNet 等模型，可以直接复用当前数据管线。

---

### 2.5 训练集和验证集切分加入随机打乱

原始脚本按每个类别的文件顺序进行切分，可能受到文件名、采集批次或存储顺序影响。

修改后对每个类别内部先随机打乱，再按 8:2 进行分层切分：

```python
rng = random.Random(seed)
rng.shuffle(paths)
```

这样可以减少数据分布偏差，使验证集更能反映模型真实性能。

---

### 2.6 改进 CNN 网络结构

原始 CNN 较浅，且第二层卷积通道数从 24 降到 12，表达能力偏弱。修改后采用更适合 224×224 输入的小型 CNN：

```text
Conv 3→32   + BatchNorm + ReLU + MaxPool
Conv 32→64  + BatchNorm + ReLU + MaxPool
Conv 64→128 + BatchNorm + ReLU + MaxPool + Dropout2d
Conv 128→256 + BatchNorm + ReLU + MaxPool + Dropout2d
Conv 256→256 + BatchNorm + ReLU + MaxPool + Dropout2d
Flatten
Dropout
Linear(256×7×7 → 512)
ReLU
Dropout
Linear(512 → n_classes)
```

输入图像为 224×224，经过 5 次池化后尺寸变化为：

```text
224 → 112 → 56 → 28 → 14 → 7
```

因此全连接层输入维度修改为：

```python
fc_in_features = 256 * 7 * 7
```

这满足 224×224 输入下的维度适配要求。

---

### 2.7 加入 BatchNorm 与 Dropout

新增：

```python
nn.BatchNorm2d(...)
nn.Dropout2d(...)
nn.Dropout(...)
```

其中：

- BatchNorm 用于稳定训练、加快收敛；
- Dropout / Dropout2d 用于抑制过拟合；
- 对小规模花卉分类数据集尤其有帮助。

---

### 2.8 优化训练超参数

默认超参数由原来的：

```text
epochs = 10
batch_size = 20
learning_rate = 1e-3
optimizer = Adam
```

修改为：

```text
epochs = 30
batch_size = 32
learning_rate = 3e-4
optimizer = AdamW
weight_decay = 1e-4
label_smoothing = 0.1
grad_clip = 1.0
```

优化点包括：

- AdamW 比 Adam 更适合配合权重衰减；
- label smoothing 可以缓解模型过度自信；
- gradient clipping 可以提高训练稳定性；
- 训练轮数增加到 30，更适合 224×224 下的 CNN 训练。

---

### 2.9 加入余弦退火学习率调度器

新增：

```python
CosineAnnealingLR
```

学习率会在训练过程中逐渐下降，有助于模型后期收敛到更稳定的解。

如果不想使用调度器，可以运行时添加：

```bash
--no-scheduler
```

---

### 2.10 保存最佳模型而不是只保存最后模型

原始脚本只保存训练结束后的模型，可能不是验证集表现最好的模型。

修改后同时保存：

```text
model/cnn_optimized_best.pt
model/cnn_optimized_last.pt
```

其中：

- `cnn_optimized_best.pt`：验证集准确率最高的模型；
- `cnn_optimized_last.pt`：最后一轮模型。

实验报告中建议优先使用 best model 的结果。

---

### 2.11 新增混淆矩阵与每类准确率

脚本会额外输出：

```text
experiment_outputs/cnn_flower_optimized/fig_best_confusion_matrix.png
experiment_outputs/cnn_flower_optimized/best_confusion_matrix.csv
experiment_outputs/cnn_flower_optimized/summary.json
```

这些结果可以用于分析：

- 哪些类别容易混淆；
- 哪些类别准确率较低；
- 数据增强和模型结构改进是否真正改善了泛化能力。

---

## 3. 推荐运行方式

基础运行：

```bash
python cnn_flower_classification_optimized.py \
  --data-dir data/flowers \
  --epochs 30 \
  --batch-size 32 \
  --learning-rate 3e-4
```

如果使用 zip 数据集：

```bash
python cnn_flower_classification_optimized.py \
  --zip-path flowers.zip \
  --data-dir data/flowers \
  --epochs 30 \
  --batch-size 32
```

快速检查模型结构：

```bash
python cnn_flower_classification_optimized.py --smoke-test
```

如果显存不足，可以降低 batch size：

```bash
python cnn_flower_classification_optimized.py --batch-size 16
```

---

## 4. 预期效果

相比原始版本，本次修改预计会带来以下改进：

1. 224×224 输入保留更多图像细节；
2. 随机分层切分使验证结果更可靠；
3. 数据增强提升泛化能力；
4. 更深的 CNN + BatchNorm 提高特征表达能力；
5. Dropout、weight decay、label smoothing 减轻过拟合；
6. best checkpoint 保存机制避免最后一轮模型退化；
7. 混淆矩阵和每类准确率使实验分析更完整。

---

## 5. 后续进一步优化建议

如果课程或实验要求允许，下一步最值得尝试的是：

1. 使用 ImageNet 预训练 ResNet18 / MobileNetV3 / EfficientNet-B0；
2. 先冻结 backbone 训练分类头，再全量微调；
3. 尝试 RandAugment、MixUp、CutMix 等更强增强；
4. 多随机种子重复实验，报告平均值和标准差；
5. 使用早停机制 Early Stopping，避免后期过拟合。
