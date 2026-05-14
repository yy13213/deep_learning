import argparse
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


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NORMALIZE = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
TRANSFORMER_IMAGE = transforms.Compose(
    [
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        NORMALIZE,
    ]
)


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
    batch_size: int = 20,
    transform=None,
    zip_path: Optional[str] = None,
    extract_to: Optional[str] = None,
    num_workers: int = 0,
):
    if transform is None:
        transform = TRANSFORMER_IMAGE

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
    character = [[] for _ in range(n_classes)]
    for path, y in dataset.samples:
        character[y].append(path)

    train_images, val_images = [], []
    train_labels, val_labels = [], []
    for i, paths in enumerate(character):
        n_train = int(len(paths) * ratio[0])
        for p in paths[:n_train]:
            train_images.append(p)
            train_labels.append(i)
        for p in paths[n_train:]:
            val_images.append(p)
            val_labels.append(i)

    train_ds = FlowerDataset(train_images, train_labels, transform)
    val_ds = FlowerDataset(val_images, val_labels, transform)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=max(1, len(val_ds)),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader, dataset.class_to_idx, n_classes


class CNN(nn.Module):
    def __init__(self, in_channels: int = 3, n_classes: int = 102):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 24, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(24, 12, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.fc1 = nn.Sequential(nn.Linear(12 * 7 * 7, 196), nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(196, 84), nn.ReLU())
        self.fc3 = nn.Linear(84, n_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


class CNNDeep(nn.Module):
    def __init__(self, in_channels: int = 3, n_classes: int = 102, p_drop: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 3 * 3, 256),
            nn.ReLU(),
            nn.Dropout(p_drop),
            nn.Linear(256, n_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.fc(x)


@torch.no_grad()
def accuracy_from_logits(logits, y):
    pred = logits.argmax(dim=1)
    return (pred == y).float().mean().item()


def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    n = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
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
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = loss_fn(logits, y)
        bs = x.size(0)
        total_loss += loss.item() * bs
        total_acc += accuracy_from_logits(logits, y) * bs
        n += bs
    return total_loss / max(1, n), total_acc / max(1, n)


def save_history_figure(history: dict, out_path: Path) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(history["train_loss"], label="train")
    ax[0].plot(history["val_loss"], label="val")
    ax[0].set_title("Loss")
    ax[0].legend()
    ax[1].plot(history["train_acc"], label="train")
    ax[1].plot(history["val_acc"], label="val")
    ax[1].set_title("Accuracy")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_smoke_test() -> None:
    x = torch.randn(8, 3, 28, 28, device=DEVICE)
    y = torch.randint(0, 102, (8,), device=DEVICE)
    loss_fn = nn.CrossEntropyLoss()
    for name, model in [("baseline", CNN(3, 102)), ("deep", CNNDeep(3, 102))]:
        model = model.to(DEVICE)
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        print(f"[smoke] {name}: logits={tuple(logits.shape)}, loss={loss.item():.6f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="exp1 CNN flower classification script")
    parser.add_argument("--data-dir", default=os.environ.get("FLOWERS_DATA_DIR", "data/flowers"))
    parser.add_argument("--zip-path", default=os.environ.get("FLOWERS_ZIP_PATH", ""))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    print("device:", DEVICE)

    if args.smoke_test:
        run_smoke_test()
        return

    train_loader, val_loader, class_to_idx, n_classes = split_train_val_data(
        args.data_dir,
        ratio=[0.8, 0.2],
        batch_size=args.batch_size,
        transform=TRANSFORMER_IMAGE,
        zip_path=args.zip_path or None,
        num_workers=args.num_workers,
    )

    model = CNN(in_channels=3, n_classes=n_classes).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_fn = nn.CrossEntropyLoss()
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(args.epochs):
        tl, ta = train_one_epoch(model, train_loader, optimizer, loss_fn, DEVICE)
        vl, va = evaluate(model, val_loader, loss_fn, DEVICE)
        history["train_loss"].append(tl)
        history["train_acc"].append(ta)
        history["val_loss"].append(vl)
        history["val_acc"].append(va)
        print(
            f"Epoch {epoch + 1}/{args.epochs} | train loss {tl:.4f} acc {ta:.4f} | "
            f"val loss {vl:.4f} acc {va:.4f}"
        )

    out_model = Path("model")
    out_model.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model_state": model.state_dict(), "class_to_idx": class_to_idx, "history": history},
        out_model / "cnn_baseline.pt",
    )

    exp_dir = Path("experiment_outputs") / "cnn_flower_py"
    exp_dir.mkdir(parents=True, exist_ok=True)
    save_history_figure(history, exp_dir / "fig_baseline_loss_acc.png")
    (exp_dir / "summary.json").write_text(
        json.dumps(
            {
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "class_to_idx": class_to_idx,
                "final_val_acc": float(history["val_acc"][-1]),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print("saved to", exp_dir.resolve())


if __name__ == "__main__":
    main()
