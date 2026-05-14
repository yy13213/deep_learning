import argparse
import json
import math
import random
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_series(step: float = 0.01) -> np.ndarray:
    xs = np.arange(0, 3, step, dtype=np.float64)
    ys = np.sin(xs * math.pi)
    return ys.astype("float32")


def normalize_series(series: np.ndarray) -> Tuple[np.ndarray, float, float]:
    max_value = float(np.max(series))
    min_value = float(np.min(series))
    scalar = max_value - min_value
    if scalar < 1e-12:
        raise ValueError("scalar too small")
    return (series - min_value) / scalar, min_value, scalar


def create_dataset(data: np.ndarray, look_back: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    data_x, data_y = [], []
    for i in range(len(data) - look_back):
        data_x.append(data[i : i + look_back])
        data_y.append(data[i + look_back])
    return np.array(data_x, dtype="float32"), np.array(data_y, dtype="float32")


def split_dataset(data_x: np.ndarray, data_y: np.ndarray, train_ratio: float = 0.67):
    n_train = int(len(data_x) * train_ratio)
    train_x, train_y = data_x[:n_train], data_y[:n_train]
    test_x, test_y = data_x[n_train:], data_y[n_train:]
    return train_x, train_y, test_x, test_y


def to_tensor_xy(
    train_X: np.ndarray,
    train_Y: np.ndarray,
    test_X: np.ndarray,
    test_Y: np.ndarray,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # 修复点：
    # 1. 使用标准时序输入形状 [batch, seq_len, input_size] = [N, 3, 1]
    # 2. 监督信号保持为 [N, 1]，避免 MSELoss 发生错误广播
    train_x = torch.from_numpy(train_X.reshape(-1, train_X.shape[1], 1))
    train_y = torch.from_numpy(train_Y.reshape(-1, 1))
    test_x = torch.from_numpy(test_X.reshape(-1, test_X.shape[1], 1))
    test_y = torch.from_numpy(test_Y.reshape(-1, 1))
    return train_x, train_y, test_x, test_y


class RNNRegressor(nn.Module):
    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 20,
        output_size: int = 1,
        num_layers: int = 2,
        rnn_type: str = "rnn",
    ):
        super().__init__()
        self.rnn_type = rnn_type
        if rnn_type == "rnn":
            self.rnn = nn.RNN(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                nonlinearity="tanh",
            )
        elif rnn_type == "lstm":
            self.rnn = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
            )
        elif rnn_type == "gru":
            self.rnn = nn.GRU(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
            )
        else:
            raise ValueError("rnn_type must be one of rnn/lstm/gru")
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x)
        last = out[:, -1, :]
        return self.linear(last)


def train_model(
    model: nn.Module,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    epochs: int = 300,
    lr: float = 1e-2,
    batch_size: int = 32,
    log_every: int = 50,
) -> List[float]:
    model = model.to(DEVICE)
    train_ds = TensorDataset(train_x, train_y)
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []

    for e in range(epochs):
        model.train()
        running_loss = 0.0
        n = 0
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(DEVICE)
            batch_y = batch_y.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()
            bs = batch_x.size(0)
            running_loss += loss.item() * bs
            n += bs
        epoch_loss = running_loss / max(n, 1)
        losses.append(epoch_loss)
        if (e + 1) % log_every == 0 or e == 0:
            print(f"Epoch {e + 1:04d} | loss {epoch_loss:.8f}")
    return losses


@torch.no_grad()
def predict_series(model: nn.Module, x: torch.Tensor) -> np.ndarray:
    model.eval()
    y = model(x.to(DEVICE)).cpu().numpy().reshape(-1)
    return y


def denormalize(y_norm: np.ndarray, min_value: float, scalar: float) -> np.ndarray:
    return y_norm * scalar + min_value


def plot_true_pred(
    y_true_norm: np.ndarray,
    y_pred_norm: np.ndarray,
    min_value: float,
    scalar: float,
    title: str,
    out_path: Path,
    n_show: int = 200,
) -> None:
    yt = denormalize(y_true_norm[:n_show], min_value, scalar)
    yp = denormalize(y_pred_norm[:n_show], min_value, scalar)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(yt, "b-", label="True", linewidth=1.2)
    ax.plot(yp, "r-", label="Pred", linewidth=1.2)
    ax.legend()
    ax.set_title(title)
    ax.set_xlabel("Test timestep")
    ax.set_ylabel("Value")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_experiment(
    epochs: int = 300,
    lr: float = 1e-2,
    hidden_size: int = 20,
    num_layers: int = 2,
    batch_size: int = 32,
    seed: int = 42,
) -> Dict[str, object]:
    set_seed(seed)
    dataset = build_series()
    dataset_norm, min_value, scalar = normalize_series(dataset)
    data_X, data_Y = create_dataset(dataset_norm, look_back=3)
    train_X, train_Y, test_X, test_Y = split_dataset(data_X, data_Y, train_ratio=0.67)
    train_x, train_y, test_x, test_y = to_tensor_xy(train_X, train_Y, test_X, test_Y)

    print("device:", DEVICE)
    print("train_x:", tuple(train_x.shape), "train_y:", tuple(train_y.shape))

    results: Dict[str, object] = {
        "dataset_norm": dataset_norm,
        "train_X": train_X,
        "train_Y": train_Y,
        "test_X": test_X,
        "test_Y": test_Y,
        "min_value": min_value,
        "scalar": scalar,
        "train_x": train_x,
        "train_y": train_y,
        "test_x": test_x,
        "test_y": test_y,
    }

    for rnn_type in ("rnn", "lstm", "gru"):
        torch.manual_seed(seed)
        model = RNNRegressor(
            input_size=1,
            hidden_size=hidden_size,
            output_size=1,
            num_layers=num_layers,
            rnn_type=rnn_type,
        )
        print(f"\n== train {rnn_type.upper()} ==")
        losses = train_model(
            model,
            train_x,
            train_y,
            epochs=epochs,
            lr=lr,
            batch_size=batch_size,
            log_every=max(epochs // 6, 1),
        )
        pred = predict_series(model, test_x)
        mse_test = float(np.mean((pred - test_y.numpy().reshape(-1)) ** 2))
        print(f"{rnn_type.upper()} test MSE (normalized): {mse_test:.8f}")
        results[rnn_type] = {
            "model": model,
            "losses": losses,
            "pred": pred,
            "mse_test": mse_test,
        }
    return results


def save_outputs(results: Dict[str, object], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    min_value = float(results["min_value"])
    scalar = float(results["scalar"])
    test_y = results["test_y"].numpy().reshape(-1)

    summary = {}
    for name in ("rnn", "lstm", "gru"):
        payload = results[name]
        plot_true_pred(
            test_y,
            payload["pred"],
            min_value,
            scalar,
            f"{name.upper()} prediction vs ground truth",
            output_dir / f"fig_pred_{name}.png",
        )
        summary[f"{name}_final_train_loss"] = float(payload["losses"][-1])
        summary[f"{name}_test_mse"] = float(payload["mse_test"])

    fig, ax = plt.subplots(figsize=(8, 4))
    for name in ("rnn", "lstm", "gru"):
        ax.plot(results[name]["losses"], label=f"{name.upper()} train MSE")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.legend()
    ax.set_title("Training loss")
    fig.tight_layout()
    fig.savefig(output_dir / "fig_train_loss_all.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    torch.save(results["lstm"]["model"].state_dict(), output_dir / "lstm_sin.pt")
    np.savez_compressed(
        output_dir / "predictions.npz",
        test_y=test_y,
        pred_rnn=results["rnn"]["pred"],
        pred_lstm=results["lstm"]["pred"],
        pred_gru=results["gru"]["pred"],
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="exp2 RNN/LSTM/GRU sin prediction")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--hidden-size", type=int, default=20)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="experiment_outputs/rnn_sin_py")
    args = parser.parse_args()

    results = run_experiment(
        epochs=args.epochs,
        lr=args.lr,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    save_outputs(results, Path(args.output_dir))
    print("saved to", Path(args.output_dir).resolve())


if __name__ == "__main__":
    main()
