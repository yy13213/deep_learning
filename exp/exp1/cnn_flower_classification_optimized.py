import argparse
import csv
import json
import os
import random
import zipfile
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder


IMAGE_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 采用 ImageNet 统计量，后续如果切换到预训练 ResNet/EfficientNet 也可以直接复用。
NORMALIZE = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
)

TRAIN_TRANSFORMER_IMAGE = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=20),
        transforms.ColorJitter(
            brightness=0.20,
            contrast=0.20,
            saturation=0.20,
            hue=0.05,
        ),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.05, 0.05),
            scale=(0.95, 1.05),
        ),
        transforms.ToTensor(),
        NORMALIZE,
        transforms.RandomErasing(p=0.15, scale=(0.02, 0.08), ratio=(0.3, 3.3)),
    ]
)

VAL_TRANSFORMER_IMAGE = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        NORMALIZE,
    ]
)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


class FlowerDataset(Dataset):
    def __init__(self, filenames, labels, transform):
        self.filenames = filenames
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        image = Image.open(self.filenames[idx]).convert("RGB")
        image = self.transform(image)
        return image, int(self.labels[idx])


def _find_imagefolder_root(base: Path) -> Path:
    candidates = [base]
    nested = base / "flowers"
    if nested.is_dir():
        candidates.append(nested)
    for p in sorted([x for x in base.iterdir() if x.is_dir()]):
        if p not in candidates:
            candidates.append(p)
    for c in candidates:
        try:
            ds = ImageFolder(str(c))
        except Exception:
            continue
        if len(ds.classes) >= 2:
            return c
    raise RuntimeError(f"无法在 {base} 下定位多类别 ImageFolder 根目录")


def split_train_val_data(
    data_dir: str,
    ratio,
    batch_size: int = 32,
    train_transform=None,
    val_transform=None,
    zip_path: Optional[str] = None,
    extract_to: Optional[str] = None,
    num_workers: int = 4,
    seed: int = 42,
):
    """按类别进行随机分层切分，避免原始文件顺序造成训练/验证分布偏差。"""
    if train_transform is None:
        train_transform = TRAIN_TRANSFORMER_IMAGE
    if val_transform is None:
        val_transform = VAL_TRANSFORMER_IMAGE

    root = Path(data_dir)
    if zip_path:
        zp = Path(zip_path)
        if not zp.is_file():
            raise FileNotFoundError(f"zip 不存在: {zp}")
        out_dir = Path(extract_to) if extract_to else root.parent / "flowers_extract"
        out_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zp, "r") as zf:
            zf.extractall(out_dir)
        dataset_root = _find_imagefolder_root(out_dir)
    else:
        dataset_root = root

    dataset = ImageFolder(str(dataset_root))
    n_classes = len(dataset.classes)
    per_class_paths = [[] for _ in range(n_classes)]
    for path, y in dataset.samples:
        per_class_paths[y].append(path)

    rng = random.Random(seed)
    train_images, val_images = [], []
    train_labels, val_labels = [], []

    for class_idx, paths in enumerate(per_class_paths):
        paths = list(paths)
        rng.shuffle(paths)

        if len(paths) <= 1:
            n_train = len(paths)
        else:
            n_train = int(len(paths) * ratio[0])
            n_train = max(1, min(len(paths) - 1, n_train))

        for p in paths[:n_train]:
            train_images.append(p)
            train_labels.append(class_idx)
        for p in paths[n_train:]:
            val_images.append(p)
            val_labels.append(class_idx)

    # 再整体打乱一次训练集，保证 DataLoader 首轮读取前顺序也不带类别块结构。
    train_pairs = list(zip(train_images, train_labels))
    rng.shuffle(train_pairs)
    if train_pairs:
        train_images, train_labels = map(list, zip(*train_pairs))

    train_ds = FlowerDataset(train_images, train_labels, train_transform)
    val_ds = FlowerDataset(val_images, val_labels, val_transform)

    loader_kwargs = {
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        **loader_kwargs,
    )
    return train_loader, val_loader, dataset.class_to_idx, n_classes


class CNN(nn.Module):
    """
    改进版 CNN：输入 224x224。

    5 次 MaxPool2d 后，特征图尺寸为：
    224 -> 112 -> 56 -> 28 -> 14 -> 7
    因此全连接层输入维度为 256 * 7 * 7。
    """

    def __init__(self, in_channels: int = 3, n_classes: int = 102, p_drop: float = 0.4):
        super().__init__()

        def conv_block(c_in: int, c_out: int, dropout2d: float = 0.0):
            layers = [
                nn.Conv2d(c_in, c_out, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(c_out),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2),
            ]
            if dropout2d > 0:
                layers.append(nn.Dropout2d(p=dropout2d))
            return nn.Sequential(*layers)

        self.features = nn.Sequential(
            conv_block(in_channels, 32),       # 224 -> 112
            conv_block(32, 64),                # 112 -> 56
            conv_block(64, 128, 0.05),         # 56 -> 28
            conv_block(128, 256, 0.10),        # 28 -> 14
            conv_block(256, 256, 0.10),        # 14 -> 7
        )

        fc_in_features = 256 * 7 * 7
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=p_drop),
            nn.Linear(fc_in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=p_drop),
            nn.Linear(512, n_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class CNNDeep(CNN):
    """保留兼容名称，实际复用改进后的 CNN 结构。"""

    pass


@torch.no_grad()
def accuracy_from_logits(logits, y):
    pred = logits.argmax(dim=1)
    return (pred == y).float().mean().item()


def train_one_epoch(model, loader, optimizer, loss_fn, device, grad_clip: float = 0.0):
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    n = 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        if grad_clip and grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()
        bs = x.size(0)
        total_loss += loss.item() * bs
        total_acc += accuracy_from_logits(logits, y) * bs
        n += bs
    return total_loss / max(1, n), total_acc / max(1, n)


@torch.no_grad()
def evaluate(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    n = 0
    all_preds = []
    all_labels = []
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits = model(x)
        loss = loss_fn(logits, y)
        pred = logits.argmax(dim=1)
        bs = x.size(0)
        total_loss += loss.item() * bs
        total_acc += (pred == y).float().mean().item() * bs
        n += bs
        all_preds.append(pred.cpu())
        all_labels.append(y.cpu())

    if all_preds:
        all_preds = torch.cat(all_preds).numpy()
        all_labels = torch.cat(all_labels).numpy()
    else:
        all_preds = np.array([])
        all_labels = np.array([])
    return total_loss / max(1, n), total_acc / max(1, n), all_preds, all_labels


def build_confusion_matrix(y_true, y_pred, n_classes: int):
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def per_class_accuracy(cm: np.ndarray, idx_to_class: dict):
    result = {}
    for i in range(cm.shape[0]):
        total = int(cm[i].sum())
        correct = int(cm[i, i])
        result[idx_to_class[i]] = float(correct / total) if total > 0 else None
    return result


def save_history_figure(history: dict, out_path: Path) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(history["train_loss"], label="train")
    ax[0].plot(history["val_loss"], label="val")
    ax[0].set_title("Loss")
    ax[0].set_xlabel("Epoch")
    ax[0].legend()

    ax[1].plot(history["train_acc"], label="train")
    ax[1].plot(history["val_acc"], label="val")
    ax[1].set_title("Accuracy")
    ax[1].set_xlabel("Epoch")
    ax[1].legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_confusion_matrix_figure(cm: np.ndarray, class_names, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm)
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    threshold = cm.max() / 2 if cm.size and cm.max() > 0 else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > threshold else "black",
                fontsize=9,
            )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_confusion_matrix_csv(cm: np.ndarray, class_names, out_path: Path) -> None:
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["true\\pred"] + list(class_names))
        for name, row in zip(class_names, cm.tolist()):
            writer.writerow([name] + row)


def run_smoke_test() -> None:
    x = torch.randn(4, 3, IMAGE_SIZE, IMAGE_SIZE, device=DEVICE)
    y = torch.randint(0, 102, (4,), device=DEVICE)
    loss_fn = nn.CrossEntropyLoss()
    model = CNN(3, 102).to(DEVICE)
    logits = model(x)
    loss = loss_fn(logits, y)
    loss.backward()
    print(f"[smoke] CNN: input={(4, 3, IMAGE_SIZE, IMAGE_SIZE)}, logits={tuple(logits.shape)}, loss={loss.item():.6f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Optimized CNN flower classification script")
    parser.add_argument("--data-dir", default=os.environ.get("FLOWERS_DATA_DIR", "data/flowers"))
    parser.add_argument("--zip-path", default=os.environ.get("FLOWERS_ZIP_PATH", ""))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--no-scheduler", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    print("device:", DEVICE)
    print(f"image size: {IMAGE_SIZE}x{IMAGE_SIZE}")

    if args.smoke_test:
        run_smoke_test()
        return

    train_loader, val_loader, class_to_idx, n_classes = split_train_val_data(
        args.data_dir,
        ratio=[0.8, 0.2],
        batch_size=args.batch_size,
        train_transform=TRAIN_TRANSFORMER_IMAGE,
        val_transform=VAL_TRANSFORMER_IMAGE,
        zip_path=args.zip_path or None,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    idx_to_class = {idx: cls for cls, idx in class_to_idx.items()}
    class_names = [idx_to_class[i] for i in range(n_classes)]

    model = CNN(in_channels=3, n_classes=n_classes, p_drop=args.dropout).to(DEVICE)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = None
    if not args.no_scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    loss_fn = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": []}

    out_model = Path("model")
    out_model.mkdir(parents=True, exist_ok=True)
    exp_dir = Path("experiment_outputs") / "cnn_flower_optimized"
    exp_dir.mkdir(parents=True, exist_ok=True)

    best_val_acc = -1.0
    best_epoch = 0
    best_cm = None
    best_per_class_acc = None

    for epoch in range(args.epochs):
        current_lr = optimizer.param_groups[0]["lr"]
        tl, ta = train_one_epoch(
            model,
            train_loader,
            optimizer,
            loss_fn,
            DEVICE,
            grad_clip=args.grad_clip,
        )
        vl, va, val_preds, val_labels = evaluate(model, val_loader, loss_fn, DEVICE)

        if scheduler is not None:
            scheduler.step()

        history["train_loss"].append(tl)
        history["train_acc"].append(ta)
        history["val_loss"].append(vl)
        history["val_acc"].append(va)
        history["lr"].append(current_lr)

        cm = build_confusion_matrix(val_labels, val_preds, n_classes)
        cls_acc = per_class_accuracy(cm, idx_to_class)

        if va > best_val_acc:
            best_val_acc = va
            best_epoch = epoch + 1
            best_cm = cm
            best_per_class_acc = cls_acc
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "class_to_idx": class_to_idx,
                    "history": history,
                    "best_val_acc": float(best_val_acc),
                    "best_epoch": best_epoch,
                    "image_size": IMAGE_SIZE,
                    "args": vars(args),
                },
                out_model / "cnn_optimized_best.pt",
            )

        print(
            f"Epoch {epoch + 1}/{args.epochs} | "
            f"lr {current_lr:.6g} | "
            f"train loss {tl:.4f} acc {ta:.4f} | "
            f"val loss {vl:.4f} acc {va:.4f} | "
            f"best acc {best_val_acc:.4f}@{best_epoch}"
        )

    torch.save(
        {
            "model_state": model.state_dict(),
            "class_to_idx": class_to_idx,
            "history": history,
            "final_val_acc": float(history["val_acc"][-1]),
            "best_val_acc": float(best_val_acc),
            "best_epoch": best_epoch,
            "image_size": IMAGE_SIZE,
            "args": vars(args),
        },
        out_model / "cnn_optimized_last.pt",
    )

    save_history_figure(history, exp_dir / "fig_optimized_loss_acc.png")
    if best_cm is not None:
        save_confusion_matrix_figure(best_cm, class_names, exp_dir / "fig_best_confusion_matrix.png")
        save_confusion_matrix_csv(best_cm, class_names, exp_dir / "best_confusion_matrix.csv")

    summary = {
        "image_size": IMAGE_SIZE,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "dropout": args.dropout,
        "label_smoothing": args.label_smoothing,
        "grad_clip": args.grad_clip,
        "scheduler": None if args.no_scheduler else "CosineAnnealingLR",
        "class_to_idx": class_to_idx,
        "final_val_acc": float(history["val_acc"][-1]),
        "best_val_acc": float(best_val_acc),
        "best_epoch": best_epoch,
        "best_per_class_acc": best_per_class_acc,
        "saved_best_model": str((out_model / "cnn_optimized_best.pt").resolve()),
        "saved_last_model": str((out_model / "cnn_optimized_last.pt").resolve()),
    }
    (exp_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("saved to", exp_dir.resolve())
    print(f"best val acc: {best_val_acc:.4f} at epoch {best_epoch}")


if __name__ == "__main__":
    main()
