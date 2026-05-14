import argparse
import copy
import json
import math
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

# Compatibility shim for environments where torchvision was installed without
# compiled custom ops. It is harmless when the real operator already exists and
# allows torchvision.datasets/transforms/utils to be imported for this MNIST task.
try:
    from torch.library import Library

    _torchvision_stub_lib = Library("torchvision", "DEF")
    _torchvision_stub_lib.define("nms(Tensor dets, Tensor scores, float iou_threshold) -> Tensor")
except Exception:
    _torchvision_stub_lib = None

import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms
from torchvision.utils import make_grid


def set_seed(seed: int = 42, deterministic: bool = False) -> None:
    """Set random seeds for reproducible experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = torch.cuda.is_available()


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 10


def load_dataset(
    batch_size: int = 256,
    download: bool = True,
    root: str = "data",
    limit_train: Optional[int] = None,
    augment: bool = False,
    num_workers: int = 2,
):
    """Load MNIST with optional light augmentation.

    For MNIST, horizontal flip is intentionally avoided because it changes digit semantics.
    """
    train_transform_steps = []
    if augment:
        train_transform_steps.append(
            transforms.RandomAffine(
                degrees=10,
                translate=(0.08, 0.08),
                scale=(0.90, 1.10),
                fill=0,
            )
        )
    train_transform_steps.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
    train_transform = transforms.Compose(train_transform_steps)
    eval_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    trainset = torchvision.datasets.MNIST(
        root=root,
        train=True,
        download=download,
        transform=train_transform,
    )
    if limit_train is not None and limit_train < len(trainset):
        trainset = torch.utils.data.Subset(trainset, list(range(limit_train)))

    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )

    testset = torchvision.datasets.MNIST(
        root=root,
        train=False,
        download=download,
        transform=eval_transform,
    )
    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )
    return trainloader, testloader


class UpsampleBlock(nn.Module):
    """Upsample + Conv block, generally less prone to checkerboard artifacts."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DCGANGenerator(nn.Module):
    """Generator with either classic ConvTranspose or improved Upsample+Conv mode.

    Conditional mode concatenates a label embedding to the noise vector. This allows the
    model to generate a specified digit class and usually improves mode coverage on MNIST.
    """

    def __init__(
        self,
        noise_dim: int = 128,
        img_channels: int = 1,
        base_channels: int = 64,
        mode: str = "upsample",
        conditional: bool = False,
        label_embed_dim: int = 32,
        num_classes: int = NUM_CLASSES,
    ):
        super().__init__()
        if mode not in {"deconv", "upsample"}:
            raise ValueError(f"Unknown generator mode: {mode}")

        self.noise_dim = noise_dim
        self.conditional = conditional
        self.label_embed_dim = label_embed_dim if conditional else 0
        self.mode = mode
        input_dim = noise_dim + self.label_embed_dim

        self.label_embed = (
            nn.Embedding(num_classes, self.label_embed_dim) if conditional else None
        )

        if mode == "deconv":
            self.net = nn.Sequential(
                nn.ConvTranspose2d(input_dim, base_channels * 4, 7, 1, 0, bias=False),
                nn.BatchNorm2d(base_channels * 4),
                nn.ReLU(True),
                nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(base_channels * 2),
                nn.ReLU(True),
                nn.ConvTranspose2d(base_channels * 2, base_channels, 4, 2, 1, bias=False),
                nn.BatchNorm2d(base_channels),
                nn.ReLU(True),
                nn.Conv2d(base_channels, img_channels, 3, 1, 1, bias=False),
                nn.Tanh(),
            )
        else:
            self.project = nn.Sequential(
                nn.Linear(input_dim, base_channels * 4 * 7 * 7, bias=False),
                nn.BatchNorm1d(base_channels * 4 * 7 * 7),
                nn.ReLU(True),
            )
            self.net = nn.Sequential(
                UpsampleBlock(base_channels * 4, base_channels * 2),
                UpsampleBlock(base_channels * 2, base_channels),
                nn.Conv2d(base_channels, img_channels, 3, 1, 1, bias=False),
                nn.Tanh(),
            )

    def _prepare_input(self, z: torch.Tensor, labels: Optional[torch.Tensor]) -> torch.Tensor:
        if not self.conditional:
            return z
        if labels is None:
            raise ValueError("Conditional generator requires labels.")
        label_vec = self.label_embed(labels).view(labels.size(0), self.label_embed_dim, 1, 1)
        return torch.cat([z, label_vec], dim=1)

    def forward(self, z: torch.Tensor, labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        z = self._prepare_input(z, labels)
        if self.mode == "deconv":
            return self.net(z)
        z = z.flatten(start_dim=1)
        x = self.project(z).view(z.size(0), -1, 7, 7)
        return self.net(x)


class BasicDiscriminator(nn.Module):
    """Original-style DCGAN discriminator kept for ablation/baseline comparison."""

    def __init__(self, img_channels: int = 1, base_channels: int = 64):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(img_channels, base_channels, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels, base_channels * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.classifier = nn.Linear(base_channels * 2 * 7 * 7, 1)

    def forward(self, x: torch.Tensor, labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        del labels
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        return self.classifier(x)


class StrongDiscriminator(nn.Module):
    """Deeper discriminator with optional SpectralNorm and projection conditioning."""

    def __init__(
        self,
        img_channels: int = 1,
        base_channels: int = 64,
        spectral_norm: bool = True,
        dropout: float = 0.20,
        conditional: bool = False,
        num_classes: int = NUM_CLASSES,
    ):
        super().__init__()
        self.conditional = conditional
        norm = nn.utils.spectral_norm if spectral_norm else (lambda module: module)

        self.features = nn.Sequential(
            norm(nn.Conv2d(img_channels, base_channels, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            norm(nn.Conv2d(base_channels, base_channels * 2, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
            norm(nn.Conv2d(base_channels * 2, base_channels * 4, 3, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
        )
        feature_dim = base_channels * 4 * 4 * 4
        self.classifier = norm(nn.Linear(feature_dim, 1))
        self.label_projection = (
            norm(nn.Embedding(num_classes, feature_dim)) if conditional else None
        )
        self.feature_dim = feature_dim

    def forward(self, x: torch.Tensor, labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        features = self.features(x).flatten(start_dim=1)
        logits = self.classifier(features)
        if self.conditional:
            if labels is None:
                raise ValueError("Conditional discriminator requires labels.")
            projected = (features * self.label_projection(labels)).sum(dim=1, keepdim=True)
            logits = logits + projected / math.sqrt(self.feature_dim)
        return logits


def init_weights(module: nn.Module) -> None:
    """DCGAN-style initialization; skips already wrapped params safely."""
    classname = module.__class__.__name__
    if "Conv" in classname or "Linear" in classname:
        if hasattr(module, "weight") and module.weight is not None:
            nn.init.normal_(module.weight.data, 0.0, 0.02)
        if getattr(module, "bias", None) is not None:
            nn.init.constant_(module.bias.data, 0)
    elif "BatchNorm" in classname:
        nn.init.normal_(module.weight.data, 1.0, 0.02)
        nn.init.constant_(module.bias.data, 0)
    elif "Embedding" in classname:
        nn.init.normal_(module.weight.data, 0.0, 0.02)


def gen_noise(n_instance: int, noise_dim: int, device: torch.device) -> torch.Tensor:
    return torch.randn(n_instance, noise_dim, 1, 1, device=device)


def make_fixed_labels(n_instance: int, device: torch.device) -> torch.Tensor:
    labels = torch.arange(NUM_CLASSES, device=device)
    repeat = math.ceil(n_instance / NUM_CLASSES)
    return labels.repeat(repeat)[:n_instance]


@torch.no_grad()
def sample_grid(
    generator: nn.Module,
    fixed_noise: torch.Tensor,
    fixed_labels: Optional[torch.Tensor] = None,
    nrow: int = 8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    was_training = generator.training
    generator.eval()
    fake = generator(fixed_noise, fixed_labels).detach().cpu()
    grid = make_grid(fake, nrow=nrow, normalize=True, value_range=(-1, 1))
    if was_training:
        generator.train()
    return fake, grid


class EMAGenerator:
    """Exponential moving average of generator weights for cleaner sampling."""

    def __init__(self, generator: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.module = copy.deepcopy(generator).eval()
        for param in self.module.parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    def update(self, generator: nn.Module) -> None:
        ema_state = self.module.state_dict()
        model_state = generator.state_dict()
        for name, ema_value in ema_state.items():
            model_value = model_state[name].detach()
            if not torch.is_floating_point(ema_value):
                ema_value.copy_(model_value)
            else:
                ema_value.mul_(self.decay).add_(model_value, alpha=1.0 - self.decay)


def add_instance_noise(images: torch.Tensor, std: float) -> torch.Tensor:
    if std <= 0:
        return images
    return (images + torch.randn_like(images) * std).clamp(-1.0, 1.0)


def discriminator_loss(
    real_logits: torch.Tensor,
    fake_logits: torch.Tensor,
    loss_type: str,
    real_smoothing: float,
) -> torch.Tensor:
    if loss_type == "hinge":
        return torch.relu(1.0 - real_logits).mean() + torch.relu(1.0 + fake_logits).mean()
    if loss_type == "bce":
        criterion = nn.BCEWithLogitsLoss()
        real_targets = torch.full_like(real_logits, real_smoothing)
        fake_targets = torch.zeros_like(fake_logits)
        return criterion(real_logits, real_targets) + criterion(fake_logits, fake_targets)
    raise ValueError(f"Unknown loss type: {loss_type}")


def generator_loss(fake_logits: torch.Tensor, loss_type: str) -> torch.Tensor:
    if loss_type == "hinge":
        return -fake_logits.mean()
    if loss_type == "bce":
        criterion = nn.BCEWithLogitsLoss()
        targets = torch.ones_like(fake_logits)
        return criterion(fake_logits, targets)
    raise ValueError(f"Unknown loss type: {loss_type}")


def save_grid_image(grid: torch.Tensor, path: Path, title: Optional[str] = None) -> None:
    fig = plt.figure(figsize=(6, 6))
    plt.imshow(grid.permute(1, 2, 0).squeeze(), cmap="gray")
    if title:
        plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def train_dcgan(
    generator: nn.Module,
    discriminator: nn.Module,
    trainloader,
    epochs: int,
    noise_dim: int,
    g_lr: float,
    d_lr: float,
    beta1: float,
    beta2: float,
    loss_type: str,
    real_smoothing: float,
    d_steps: int,
    grad_clip: float,
    instance_noise: float,
    print_every: int,
    sample_every: int,
    output_dir: Path,
    fixed_noise: torch.Tensor,
    fixed_labels: Optional[torch.Tensor],
    conditional: bool,
    ema: Optional[EMAGenerator] = None,
) -> Dict[str, List[float]]:
    g_optimizer = optim.Adam(generator.parameters(), lr=g_lr, betas=(beta1, beta2))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=d_lr, betas=(beta1, beta2))

    history: Dict[str, List[float]] = {
        "D_loss": [],
        "G_loss": [],
        "D_real_logit": [],
        "D_fake_logit": [],
        "instance_noise_std": [],
    }
    snapshots: List[torch.Tensor] = []

    for epoch in range(epochs):
        generator.train()
        discriminator.train()
        d_epoch_loss = 0.0
        g_epoch_loss = 0.0
        real_logit_epoch = 0.0
        fake_logit_epoch = 0.0
        step_count = 0
        current_noise_std = instance_noise * max(0.0, 1.0 - epoch / max(epochs, 1))

        for step, (real_images, real_labels) in enumerate(trainloader, start=1):
            real_images = real_images.to(DEVICE, non_blocking=True)
            real_labels = real_labels.to(DEVICE, non_blocking=True)
            batch_size = real_images.size(0)

            latest_d_loss = None
            latest_real_logits = None
            latest_fake_logits = None
            for _ in range(max(1, d_steps)):
                d_optimizer.zero_grad(set_to_none=True)

                fake_labels = (
                    torch.randint(0, NUM_CLASSES, (batch_size,), device=DEVICE)
                    if conditional
                    else None
                )
                noise = gen_noise(batch_size, noise_dim, DEVICE)
                with torch.no_grad():
                    fake_images = generator(noise, fake_labels)

                real_for_d = add_instance_noise(real_images, current_noise_std)
                fake_for_d = add_instance_noise(fake_images, current_noise_std)

                d_real_labels = real_labels if conditional else None
                real_logits = discriminator(real_for_d, d_real_labels)
                fake_logits = discriminator(fake_for_d.detach(), fake_labels)
                d_loss = discriminator_loss(
                    real_logits=real_logits,
                    fake_logits=fake_logits,
                    loss_type=loss_type,
                    real_smoothing=real_smoothing,
                )
                d_loss.backward()
                if grad_clip > 0:
                    nn.utils.clip_grad_norm_(discriminator.parameters(), grad_clip)
                d_optimizer.step()

                latest_d_loss = d_loss.detach()
                latest_real_logits = real_logits.detach()
                latest_fake_logits = fake_logits.detach()

            g_optimizer.zero_grad(set_to_none=True)
            gen_labels = (
                torch.randint(0, NUM_CLASSES, (batch_size,), device=DEVICE)
                if conditional
                else None
            )
            noise = gen_noise(batch_size, noise_dim, DEVICE)
            generated = generator(noise, gen_labels)
            gen_logits = discriminator(add_instance_noise(generated, current_noise_std), gen_labels)
            g_loss = generator_loss(gen_logits, loss_type)
            g_loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(generator.parameters(), grad_clip)
            g_optimizer.step()
            if ema is not None:
                ema.update(generator)

            d_epoch_loss += float(latest_d_loss.item())
            g_epoch_loss += float(g_loss.item())
            real_logit_epoch += float(latest_real_logits.mean().item())
            fake_logit_epoch += float(latest_fake_logits.mean().item())
            step_count += 1

            if print_every > 0 and step % print_every == 0:
                print(
                    f"[epoch {epoch + 1:03d}, step {step:04d}] "
                    f"D loss: {latest_d_loss.item():.4f} | "
                    f"G loss: {g_loss.item():.4f} | "
                    f"D(real): {latest_real_logits.mean().item():.3f} | "
                    f"D(fake): {latest_fake_logits.mean().item():.3f}"
                )

        history["D_loss"].append(d_epoch_loss / max(step_count, 1))
        history["G_loss"].append(g_epoch_loss / max(step_count, 1))
        history["D_real_logit"].append(real_logit_epoch / max(step_count, 1))
        history["D_fake_logit"].append(fake_logit_epoch / max(step_count, 1))
        history["instance_noise_std"].append(current_noise_std)

        sample_model = ema.module if ema is not None else generator
        _, grid = sample_grid(sample_model, fixed_noise, fixed_labels, nrow=8)
        snapshots.append(grid)
        print(
            f"epoch {epoch + 1:03d}/{epochs:03d} finished | "
            f"D loss={history['D_loss'][-1]:.4f}, "
            f"G loss={history['G_loss'][-1]:.4f}, "
            f"D(real)={history['D_real_logit'][-1]:.3f}, "
            f"D(fake)={history['D_fake_logit'][-1]:.3f}"
        )

        if sample_every > 0 and ((epoch + 1) % sample_every == 0 or epoch + 1 == epochs):
            save_grid_image(grid, output_dir / f"snapshot_epoch_{epoch + 1:03d}.png")

    history["snapshots"] = snapshots
    return history


class MNISTClassifier(nn.Module):
    """Small classifier used only for optional quantitative evaluation of generated digits."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 1),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(True),
            nn.Linear(128, NUM_CLASSES),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_classifier(
    trainloader,
    testloader,
    epochs: int,
    lr: float = 1e-3,
) -> Tuple[MNISTClassifier, Dict[str, float]]:
    classifier = MNISTClassifier().to(DEVICE)
    optimizer = optim.Adam(classifier.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        classifier.train()
        total_loss = 0.0
        total = 0
        correct = 0
        for images, labels in trainloader:
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = classifier(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * images.size(0)
            pred = logits.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += images.size(0)
        print(
            f"classifier epoch {epoch + 1:02d}/{epochs:02d} | "
            f"train loss={total_loss / max(total, 1):.4f}, "
            f"train acc={correct / max(total, 1):.4f}"
        )

    classifier.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for images, labels in testloader:
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)
            logits = classifier(images)
            pred = logits.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += images.size(0)
    metrics = {"classifier_test_acc": correct / max(total, 1)}
    return classifier, metrics


@torch.no_grad()
def evaluate_generator_with_classifier(
    generator: nn.Module,
    classifier: nn.Module,
    noise_dim: int,
    conditional: bool,
    n_samples: int = 2000,
    batch_size: int = 256,
) -> Dict[str, object]:
    generator.eval()
    classifier.eval()
    counts = torch.zeros(NUM_CLASSES, dtype=torch.long)
    confidence_sum = 0.0
    total = 0

    for start in range(0, n_samples, batch_size):
        current_bs = min(batch_size, n_samples - start)
        noise = gen_noise(current_bs, noise_dim, DEVICE)
        labels = (
            torch.arange(start, start + current_bs, device=DEVICE) % NUM_CLASSES
            if conditional
            else None
        )
        fake = generator(noise, labels)
        logits = classifier(fake)
        probs = logits.softmax(dim=1)
        confidence, pred = probs.max(dim=1)
        counts += torch.bincount(pred.cpu(), minlength=NUM_CLASSES)
        confidence_sum += confidence.sum().item()
        total += current_bs

    distribution = (counts.float() / max(total, 1)).tolist()
    entropy = float(-(torch.tensor(distribution) + 1e-12).log().mul(torch.tensor(distribution)).sum().item())
    return {
        "n_generated_samples": total,
        "mean_classifier_confidence": confidence_sum / max(total, 1),
        "predicted_class_counts": counts.tolist(),
        "predicted_class_distribution": distribution,
        "predicted_class_entropy": entropy,
    }


def save_training_artifacts(
    history: Dict[str, List[float]],
    generator: nn.Module,
    fixed_noise: torch.Tensor,
    fixed_labels: Optional[torch.Tensor],
    output_dir: Path,
    config: dict,
    ema: Optional[EMAGenerator] = None,
    eval_metrics: Optional[Dict[str, object]] = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "history.npz",
        D_loss=np.array(history["D_loss"], dtype=np.float32),
        G_loss=np.array(history["G_loss"], dtype=np.float32),
        D_real_logit=np.array(history["D_real_logit"], dtype=np.float32),
        D_fake_logit=np.array(history["D_fake_logit"], dtype=np.float32),
        instance_noise_std=np.array(history["instance_noise_std"], dtype=np.float32),
    )

    sample_model = ema.module if ema is not None else generator
    _, final_grid = sample_grid(sample_model, fixed_noise, fixed_labels, nrow=8)
    save_grid_image(final_grid, output_dir / "generated_final.png", title="Final generated samples")

    fig = plt.figure(figsize=(8, 4))
    plt.plot(history["D_loss"], label="Discriminator loss")
    plt.plot(history["G_loss"], label="Generator loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("GAN training loss")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_dir / "fig_train_loss.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig = plt.figure(figsize=(8, 4))
    plt.plot(history["D_real_logit"], label="D(real) logit")
    plt.plot(history["D_fake_logit"], label="D(fake) logit")
    plt.xlabel("epoch")
    plt.ylabel("mean logit")
    plt.title("Discriminator logits")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_dir / "fig_discriminator_logits.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    torch.save(generator.state_dict(), output_dir / "generator.pt")
    if ema is not None:
        torch.save(ema.module.state_dict(), output_dir / "generator_ema.pt")

    config_to_save = dict(config)
    if eval_metrics is not None:
        config_to_save["eval_metrics"] = eval_metrics
    (output_dir / "summary.json").write_text(
        json.dumps(config_to_save, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def run_biggan_demo() -> None:
    try:
        from pytorch_pretrained_biggan import BigGAN, one_hot_from_names, truncated_noise_sample
    except Exception as exc:
        print("BIGGAN 依赖不可用:", exc)
        return

    model = BigGAN.from_pretrained("biggan-deep-128")
    model.to(DEVICE)
    model.eval()
    class_vector = one_hot_from_names(["golden retriever"] * 4, batch_size=4)
    noise_vector = truncated_noise_sample(truncation=0.4, batch_size=4)
    noise_vector = torch.from_numpy(noise_vector).to(DEVICE)
    class_vector = torch.from_numpy(class_vector).to(DEVICE)
    with torch.no_grad():
        output = model(noise_vector, class_vector, 0.4)
    grid = make_grid(output.cpu(), nrow=2, normalize=True, value_range=(-1, 1))
    plt.figure(figsize=(6, 6))
    plt.imshow(grid.permute(1, 2, 0))
    plt.title("BIGGAN demo")
    plt.axis("off")
    plt.show()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Improved DCGAN on MNIST")

    # Stage 1: more controllable training hyperparameters.
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--noise-dim", type=int, default=128)
    parser.add_argument("--g-lr", type=float, default=2e-4)
    parser.add_argument("--d-lr", type=float, default=2e-4)
    parser.add_argument("--lr", type=float, default=None, help="Compatibility alias: set both g_lr and d_lr.")
    parser.add_argument("--beta1", type=float, default=0.5)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--d-steps", type=int, default=1)
    parser.add_argument("--grad-clip", type=float, default=0.0)

    # Stage 2: model architecture and discriminator stabilization.
    parser.add_argument("--generator-mode", choices=["deconv", "upsample"], default="upsample")
    parser.add_argument("--discriminator", choices=["basic", "strong"], default="strong")
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--d-base-channels", type=int, default=64)
    parser.add_argument("--no-spectral-norm", action="store_true")
    parser.add_argument("--dropout", type=float, default=0.20)

    # Stage 3: loss-level stabilization.
    parser.add_argument("--loss", choices=["hinge", "bce"], default="hinge")
    parser.add_argument("--real-smoothing", type=float, default=0.9)
    parser.add_argument("--instance-noise", type=float, default=0.03)

    # Additional useful upgrades.
    parser.add_argument("--conditional", action="store_true")
    parser.add_argument("--label-embed-dim", type=int, default=32)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--no-ema", action="store_true")
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--eval-classifier", action="store_true")
    parser.add_argument("--classifier-epochs", type=int, default=2)
    parser.add_argument("--eval-samples", type=int, default=2000)

    # Reproducibility and I/O.
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--torch-num-threads", type=int, default=None, help="Limit CPU threads if local PyTorch CPU training hangs or is too slow.")
    parser.add_argument("--print-every", type=int, default=100)
    parser.add_argument("--sample-every", type=int, default=1)
    parser.add_argument("--output-dir", default="experiment_outputs/dcgan_mnist_improved")
    parser.add_argument("--biggan-demo", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.lr is not None:
        args.g_lr = args.lr
        args.d_lr = args.lr

    if args.torch_num_threads is not None and args.torch_num_threads > 0:
        torch.set_num_threads(args.torch_num_threads)

    set_seed(args.seed, deterministic=args.deterministic)
    print("device:", DEVICE)
    print("torch:", torch.__version__)
    print("torchvision:", torchvision.__version__)

    if args.biggan_demo:
        run_biggan_demo()
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    trainloader, testloader = load_dataset(
        batch_size=args.batch_size,
        download=True,
        root=args.data_root,
        limit_train=args.limit_train,
        augment=args.augment,
        num_workers=args.num_workers,
    )

    generator = DCGANGenerator(
        noise_dim=args.noise_dim,
        base_channels=args.base_channels,
        mode=args.generator_mode,
        conditional=args.conditional,
        label_embed_dim=args.label_embed_dim,
    ).to(DEVICE)

    if args.discriminator == "basic":
        if args.conditional:
            raise ValueError("Conditional training requires --discriminator strong.")
        discriminator = BasicDiscriminator(base_channels=args.d_base_channels).to(DEVICE)
    else:
        discriminator = StrongDiscriminator(
            base_channels=args.d_base_channels,
            spectral_norm=not args.no_spectral_norm,
            dropout=args.dropout,
            conditional=args.conditional,
        ).to(DEVICE)

    generator.apply(init_weights)
    discriminator.apply(init_weights)

    fixed_noise = gen_noise(64, args.noise_dim, DEVICE)
    fixed_labels = make_fixed_labels(64, DEVICE) if args.conditional else None
    ema = None if args.no_ema else EMAGenerator(generator, decay=args.ema_decay)

    history = train_dcgan(
        generator=generator,
        discriminator=discriminator,
        trainloader=trainloader,
        epochs=args.epochs,
        noise_dim=args.noise_dim,
        g_lr=args.g_lr,
        d_lr=args.d_lr,
        beta1=args.beta1,
        beta2=args.beta2,
        loss_type=args.loss,
        real_smoothing=args.real_smoothing,
        d_steps=args.d_steps,
        grad_clip=args.grad_clip,
        instance_noise=args.instance_noise,
        print_every=args.print_every,
        sample_every=args.sample_every,
        output_dir=output_dir,
        fixed_noise=fixed_noise,
        fixed_labels=fixed_labels,
        conditional=args.conditional,
        ema=ema,
    )

    eval_metrics = None
    if args.eval_classifier:
        classifier, clf_metrics = train_classifier(
            trainloader=trainloader,
            testloader=testloader,
            epochs=args.classifier_epochs,
        )
        sample_model = ema.module if ema is not None else generator
        gen_metrics = evaluate_generator_with_classifier(
            generator=sample_model,
            classifier=classifier,
            noise_dim=args.noise_dim,
            conditional=args.conditional,
            n_samples=args.eval_samples,
            batch_size=args.batch_size,
        )
        eval_metrics = {**clf_metrics, **gen_metrics}
        print("evaluation metrics:", json.dumps(eval_metrics, ensure_ascii=False, indent=2))

    summary = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "noise_dim": args.noise_dim,
        "g_lr": args.g_lr,
        "d_lr": args.d_lr,
        "beta1": args.beta1,
        "beta2": args.beta2,
        "d_steps": args.d_steps,
        "grad_clip": args.grad_clip,
        "generator_mode": args.generator_mode,
        "discriminator": args.discriminator,
        "base_channels": args.base_channels,
        "d_base_channels": args.d_base_channels,
        "spectral_norm": not args.no_spectral_norm,
        "dropout": args.dropout,
        "loss": args.loss,
        "real_smoothing": args.real_smoothing,
        "instance_noise": args.instance_noise,
        "conditional": args.conditional,
        "label_embed_dim": args.label_embed_dim,
        "ema": ema is not None,
        "ema_decay": args.ema_decay if ema is not None else None,
        "augment": args.augment,
        "device": str(DEVICE),
        "seed": args.seed,
        "deterministic": args.deterministic,
        "limit_train": args.limit_train,
        "final_D_loss": float(history["D_loss"][-1]),
        "final_G_loss": float(history["G_loss"][-1]),
        "final_D_real_logit": float(history["D_real_logit"][-1]),
        "final_D_fake_logit": float(history["D_fake_logit"][-1]),
    }
    save_training_artifacts(
        history=history,
        generator=generator,
        fixed_noise=fixed_noise,
        fixed_labels=fixed_labels,
        output_dir=output_dir,
        config=summary,
        ema=ema,
        eval_metrics=eval_metrics,
    )
    print("saved to", output_dir.resolve())


if __name__ == "__main__":
    main()
