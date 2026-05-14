import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms
from torchvision.utils import make_grid


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_dataset(
    batch_size: int = 256,
    download: bool = True,
    root: str = "data",
    limit_train: Optional[int] = None,
):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
    trainset = torchvision.datasets.MNIST(
        root=root,
        train=True,
        download=download,
        transform=transform,
    )
    if limit_train is not None and limit_train < len(trainset):
        trainset = torch.utils.data.Subset(trainset, list(range(limit_train)))
    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        drop_last=True,
        pin_memory=torch.cuda.is_available(),
    )
    testset = torchvision.datasets.MNIST(
        root=root,
        train=False,
        download=download,
        transform=transform,
    )
    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )
    return trainloader, testloader


class DCGANGenerator(nn.Module):
    def __init__(self, noise_dim: int = 100, img_channels: int = 1, base_channels: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(noise_dim, base_channels * 4, 7, 1, 0, bias=False),
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

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class DCGANDiscriminator(nn.Module):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        return self.classifier(x)


def init_weights(module: nn.Module) -> None:
    classname = module.__class__.__name__
    if "Conv" in classname or "Linear" in classname:
        nn.init.normal_(module.weight.data, 0.0, 0.02)
        if getattr(module, "bias", None) is not None:
            nn.init.constant_(module.bias.data, 0)
    elif "BatchNorm" in classname:
        nn.init.normal_(module.weight.data, 1.0, 0.02)
        nn.init.constant_(module.bias.data, 0)


def gen_noise(n_instance: int, noise_dim: int, device: torch.device) -> torch.Tensor:
    return torch.randn(n_instance, noise_dim, 1, 1, device=device)


@torch.no_grad()
def sample_grid(generator: nn.Module, fixed_noise: torch.Tensor, nrow: int = 8):
    was_training = generator.training
    generator.eval()
    fake = generator(fixed_noise).detach().cpu()
    grid = make_grid(fake, nrow=nrow, normalize=True, value_range=(-1, 1))
    if was_training:
        generator.train()
    return fake, grid


def train_dcgan(
    generator: nn.Module,
    discriminator: nn.Module,
    trainloader,
    epochs: int,
    noise_dim: int,
    lr: float,
    print_every: int,
    output_dir: Path,
    fixed_noise: torch.Tensor,
) -> Dict[str, List[float]]:
    criterion = nn.BCEWithLogitsLoss()
    g_optimizer = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

    history = {"D_loss": [], "G_loss": []}
    snapshots: List[torch.Tensor] = []

    for epoch in range(epochs):
        generator.train()
        discriminator.train()
        d_epoch_loss = 0.0
        g_epoch_loss = 0.0
        step_count = 0

        for step, (real_images, _) in enumerate(trainloader, start=1):
            real_images = real_images.to(DEVICE, non_blocking=True)
            batch_size = real_images.size(0)
            real_targets = torch.full((batch_size, 1), 0.9, device=DEVICE)
            fake_targets = torch.zeros((batch_size, 1), device=DEVICE)

            # 训练判别器：真实图像判为 1，假图像判为 0
            d_optimizer.zero_grad(set_to_none=True)
            real_logits = discriminator(real_images)
            d_real_loss = criterion(real_logits, real_targets)

            noise = gen_noise(batch_size, noise_dim, DEVICE)
            fake_images = generator(noise)
            fake_logits = discriminator(fake_images.detach())
            d_fake_loss = criterion(fake_logits, fake_targets)
            d_loss = d_real_loss + d_fake_loss
            d_loss.backward()
            d_optimizer.step()

            # 训练生成器：让假图像尽量被判为真
            g_optimizer.zero_grad(set_to_none=True)
            noise = gen_noise(batch_size, noise_dim, DEVICE)
            generated = generator(noise)
            gen_logits = discriminator(generated)
            g_targets = torch.ones((batch_size, 1), device=DEVICE)
            g_loss = criterion(gen_logits, g_targets)
            g_loss.backward()
            g_optimizer.step()

            d_epoch_loss += d_loss.item()
            g_epoch_loss += g_loss.item()
            step_count += 1

            if step % print_every == 0:
                print(
                    f"[epoch {epoch + 1:02d}, step {step:04d}] "
                    f"D loss: {d_loss.item():.4f} | G loss: {g_loss.item():.4f}"
                )

        history["D_loss"].append(d_epoch_loss / max(step_count, 1))
        history["G_loss"].append(g_epoch_loss / max(step_count, 1))
        _, grid = sample_grid(generator, fixed_noise, nrow=8)
        snapshots.append(grid)
        print(
            f"epoch {epoch + 1:02d}/{epochs:02d} finished | "
            f"D loss={history['D_loss'][-1]:.4f}, G loss={history['G_loss'][-1]:.4f}"
        )

        fig = plt.figure(figsize=(6, 6))
        plt.imshow(grid.permute(1, 2, 0).squeeze(), cmap="gray")
        plt.axis("off")
        plt.tight_layout()
        fig.savefig(output_dir / f"snapshot_epoch_{epoch + 1:03d}.png", dpi=160, bbox_inches="tight")
        plt.close(fig)

    history["snapshots"] = snapshots
    return history


def save_training_artifacts(
    history: Dict[str, List[float]],
    generator: nn.Module,
    fixed_noise: torch.Tensor,
    output_dir: Path,
    config: dict,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "history.npz",
        D_loss=np.array(history["D_loss"], dtype=np.float32),
        G_loss=np.array(history["G_loss"], dtype=np.float32),
    )

    _, final_grid = sample_grid(generator, fixed_noise, nrow=8)
    fig = plt.figure(figsize=(6, 6))
    plt.imshow(final_grid.permute(1, 2, 0).squeeze(), cmap="gray")
    plt.title("Final generated samples")
    plt.axis("off")
    plt.tight_layout()
    fig.savefig(output_dir / "generated_final.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    fig = plt.figure(figsize=(8, 4))
    plt.plot(history["D_loss"], label="Discriminator loss")
    plt.plot(history["G_loss"], label="Generator loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("DCGAN training loss")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_dir / "fig_train_loss.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    torch.save(generator.state_dict(), output_dir / "generator.pt")
    (output_dir / "summary.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
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


def main() -> None:
    parser = argparse.ArgumentParser(description="exp3 DCGAN on MNIST")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--noise-dim", type=int, default=100)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--print-every", type=int, default=100)
    parser.add_argument("--output-dir", default="experiment_outputs/dcgan_mnist_py")
    parser.add_argument("--biggan-demo", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    print("device:", DEVICE)
    print("torch:", torch.__version__)
    print("torchvision:", torchvision.__version__)

    if args.biggan_demo:
        run_biggan_demo()
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    trainloader, _ = load_dataset(
        batch_size=args.batch_size,
        download=True,
        root=args.data_root,
        limit_train=args.limit_train,
    )
    generator = DCGANGenerator(noise_dim=args.noise_dim).to(DEVICE)
    discriminator = DCGANDiscriminator().to(DEVICE)
    generator.apply(init_weights)
    discriminator.apply(init_weights)

    fixed_noise = gen_noise(64, args.noise_dim, DEVICE)
    history = train_dcgan(
        generator=generator,
        discriminator=discriminator,
        trainloader=trainloader,
        epochs=args.epochs,
        noise_dim=args.noise_dim,
        lr=args.lr,
        print_every=args.print_every,
        output_dir=output_dir,
        fixed_noise=fixed_noise,
    )

    summary = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "noise_dim": args.noise_dim,
        "lr": args.lr,
        "device": str(DEVICE),
        "limit_train": args.limit_train,
        "final_D_loss": float(history["D_loss"][-1]),
        "final_G_loss": float(history["G_loss"][-1]),
    }
    save_training_artifacts(history, generator, fixed_noise, output_dir, summary)
    print("saved to", output_dir.resolve())


if __name__ == "__main__":
    main()
