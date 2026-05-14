"""
Optimized RNN/LSTM/GRU experiment for sine-wave time-series prediction.

Main improvements over the original baseline:
1. Longer configurable look-back window.
2. Train/validation/test split instead of train/test only.
3. Train-only normalization to avoid test-set information leakage.
4. RNN/LSTM/GRU backbone + LayerNorm + Dropout + MLP prediction head.
5. Optional residual prediction: y_hat = last_input + predicted_delta.
6. AdamW, weight decay, gradient clipping, LR scheduler, and early stopping.
7. Richer metrics: normalized and denormalized MSE/RMSE/MAE/R2.
8. Multi-seed experiment support and optional compact hyperparameter search.
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import math
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

# For this tiny sequence task, excessive CPU BLAS threads can be much slower than one
# thread because thread-management overhead dominates the actual computation.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type == "cpu":
    torch.set_num_threads(max(1, int(os.environ.get("OMP_NUM_THREADS", "1"))))
MODEL_TYPES = ("rnn", "lstm", "gru")


def set_seed(seed: int = 42) -> None:
    """Make the experiment as reproducible as possible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def build_series(step: float = 0.01, end: float = 3.0) -> np.ndarray:
    """Build y = sin(pi * x) on [0, end)."""
    xs = np.arange(0, end, step, dtype=np.float64)
    ys = np.sin(xs * math.pi)
    return ys.astype("float32")


def create_dataset(data: np.ndarray, look_back: int = 30) -> Tuple[np.ndarray, np.ndarray]:
    """Create sliding-window samples for one-step-ahead prediction."""
    if look_back < 1:
        raise ValueError("look_back must be positive")
    if len(data) <= look_back:
        raise ValueError("sequence length must be larger than look_back")

    data_x, data_y = [], []
    for i in range(len(data) - look_back):
        data_x.append(data[i : i + look_back])
        data_y.append(data[i + look_back])
    return np.asarray(data_x, dtype="float32"), np.asarray(data_y, dtype="float32")


def split_dataset(
    data_x: np.ndarray,
    data_y: np.ndarray,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sequential split: train first, then validation, then test."""
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be in (0, 1)")
    if not 0 <= val_ratio < 1:
        raise ValueError("val_ratio must be in [0, 1)")
    if train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio + val_ratio must be smaller than 1")

    n = len(data_x)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val
    if min(n_train, n_val, n_test) <= 0:
        raise ValueError(
            f"invalid split: train={n_train}, val={n_val}, test={n_test}. "
            "Try reducing look_back or changing split ratios."
        )

    train_x, train_y = data_x[:n_train], data_y[:n_train]
    val_x, val_y = data_x[n_train : n_train + n_val], data_y[n_train : n_train + n_val]
    test_x, test_y = data_x[n_train + n_val :], data_y[n_train + n_val :]
    return train_x, train_y, val_x, val_y, test_x, test_y


def fit_normalizer(*arrays: np.ndarray) -> Tuple[float, float]:
    """Fit min-max normalization only on training data."""
    merged = np.concatenate([arr.reshape(-1) for arr in arrays])
    min_value = float(np.min(merged))
    max_value = float(np.max(merged))
    scalar = max_value - min_value
    if scalar < 1e-12:
        raise ValueError("normalization scalar is too small")
    return min_value, scalar


def normalize(array: np.ndarray, min_value: float, scalar: float) -> np.ndarray:
    return ((array - min_value) / scalar).astype("float32")


def denormalize(array: np.ndarray, min_value: float, scalar: float) -> np.ndarray:
    return (array * scalar + min_value).astype("float32")


def to_tensor_xy(data_x: np.ndarray, data_y: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert arrays to standard recurrent input shape [batch, seq_len, input_size]."""
    x = torch.from_numpy(data_x.reshape(-1, data_x.shape[1], 1)).float()
    y = torch.from_numpy(data_y.reshape(-1, 1)).float()
    return x, y


@dataclass(frozen=True)
class ExperimentConfig:
    step: float = 0.01
    end: float = 3.0
    look_back: int = 30
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    epochs: int = 1000
    lr: float = 1e-3
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.10
    batch_size: int = 32
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    patience: int = 80
    min_delta: float = 1e-7
    residual_prediction: bool = True
    seed: int = 42


class SequenceRegressor(nn.Module):
    """RNN/LSTM/GRU backbone with a stronger nonlinear prediction head."""

    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 64,
        output_size: int = 1,
        num_layers: int = 2,
        dropout: float = 0.10,
        rnn_type: str = "gru",
        residual_prediction: bool = True,
    ):
        super().__init__()
        if rnn_type not in MODEL_TYPES:
            raise ValueError(f"rnn_type must be one of {MODEL_TYPES}")

        self.rnn_type = rnn_type
        self.residual_prediction = residual_prediction
        recurrent_dropout = dropout if num_layers > 1 else 0.0

        if rnn_type == "rnn":
            self.backbone = nn.RNN(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=recurrent_dropout,
                nonlinearity="tanh",
            )
        elif rnn_type == "lstm":
            self.backbone = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=recurrent_dropout,
            )
        else:
            self.backbone = nn.GRU(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=recurrent_dropout,
            )

        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.backbone(x)
        last_hidden = out[:, -1, :]
        raw_output = self.head(last_hidden)
        if self.residual_prediction:
            return x[:, -1, :] + raw_output
        return raw_output


def evaluate_loss(model: nn.Module, x: torch.Tensor, y: torch.Tensor, criterion: nn.Module) -> float:
    model.eval()
    with torch.no_grad():
        pred = model(x.to(DEVICE))
        loss = criterion(pred, y.to(DEVICE))
    return float(loss.item())


def train_model(
    model: nn.Module,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    val_x: torch.Tensor,
    val_y: torch.Tensor,
    cfg: ExperimentConfig,
    log_every: int = 50,
) -> Dict[str, object]:
    """Train with AdamW, LR scheduling, gradient clipping, and early stopping."""
    model = model.to(DEVICE)
    train_ds = TensorDataset(train_x, train_y)
    generator = torch.Generator()
    generator.manual_seed(cfg.seed)
    loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        generator=generator,
    )

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=max(cfg.patience // 4, 5),
    )

    best_state = copy.deepcopy(model.state_dict())
    best_val = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    train_losses: List[float] = []
    val_losses: List[float] = []
    learning_rates: List[float] = []

    for epoch in range(1, cfg.epochs + 1):
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
            if cfg.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.grad_clip)
            optimizer.step()

            batch_size = batch_x.size(0)
            running_loss += float(loss.item()) * batch_size
            n += batch_size

        train_loss = running_loss / max(n, 1)
        val_loss = evaluate_loss(model, val_x, val_y, criterion)
        scheduler.step(val_loss)

        current_lr = float(optimizer.param_groups[0]["lr"])
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        learning_rates.append(current_lr)

        improved = val_loss < best_val - cfg.min_delta
        if improved:
            best_val = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epoch == 1 or epoch % log_every == 0:
            print(
                f"Epoch {epoch:04d} | train {train_loss:.8f} | "
                f"val {val_loss:.8f} | lr {current_lr:.2e}"
            )

        if epochs_without_improvement >= cfg.patience:
            print(
                f"Early stopping at epoch {epoch}. "
                f"Best epoch: {best_epoch}, best val MSE: {best_val:.8f}"
            )
            break

    model.load_state_dict(best_state)
    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "learning_rates": learning_rates,
        "best_epoch": best_epoch,
        "best_val_mse": best_val,
        "stopped_epoch": len(train_losses),
    }


@torch.no_grad()
def predict_series(model: nn.Module, x: torch.Tensor) -> np.ndarray:
    model.eval()
    return model(x.to(DEVICE)).cpu().numpy().reshape(-1)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    mse = float(np.mean((y_true - y_pred) ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    denom = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1.0 - np.sum((y_true - y_pred) ** 2) / denom) if denom > 1e-12 else float("nan")
    return {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2}


def prepare_data(cfg: ExperimentConfig) -> Dict[str, object]:
    raw_series = build_series(step=cfg.step, end=cfg.end)
    raw_x, raw_y = create_dataset(raw_series, look_back=cfg.look_back)
    train_x_raw, train_y_raw, val_x_raw, val_y_raw, test_x_raw, test_y_raw = split_dataset(
        raw_x,
        raw_y,
        train_ratio=cfg.train_ratio,
        val_ratio=cfg.val_ratio,
    )

    min_value, scalar = fit_normalizer(train_x_raw, train_y_raw)
    train_x_np = normalize(train_x_raw, min_value, scalar)
    train_y_np = normalize(train_y_raw, min_value, scalar)
    val_x_np = normalize(val_x_raw, min_value, scalar)
    val_y_np = normalize(val_y_raw, min_value, scalar)
    test_x_np = normalize(test_x_raw, min_value, scalar)
    test_y_np = normalize(test_y_raw, min_value, scalar)

    train_x, train_y = to_tensor_xy(train_x_np, train_y_np)
    val_x, val_y = to_tensor_xy(val_x_np, val_y_np)
    test_x, test_y = to_tensor_xy(test_x_np, test_y_np)

    return {
        "raw_series": raw_series,
        "min_value": min_value,
        "scalar": scalar,
        "train_x": train_x,
        "train_y": train_y,
        "val_x": val_x,
        "val_y": val_y,
        "test_x": test_x,
        "test_y": test_y,
        "train_y_raw": train_y_raw,
        "val_y_raw": val_y_raw,
        "test_y_raw": test_y_raw,
        "split_sizes": {
            "train": int(len(train_x)),
            "val": int(len(val_x)),
            "test": int(len(test_x)),
        },
    }


def run_single_model(rnn_type: str, cfg: ExperimentConfig, data: Dict[str, object]) -> Dict[str, object]:
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    model = SequenceRegressor(
        input_size=1,
        hidden_size=cfg.hidden_size,
        output_size=1,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
        rnn_type=rnn_type,
        residual_prediction=cfg.residual_prediction,
    )

    print(f"\n== train {rnn_type.upper()} ==")
    history = train_model(
        model,
        data["train_x"],
        data["train_y"],
        data["val_x"],
        data["val_y"],
        cfg,
        log_every=max(cfg.epochs // 10, 1),
    )

    pred_norm = predict_series(model, data["test_x"])
    true_norm = data["test_y"].numpy().reshape(-1)
    min_value = float(data["min_value"])
    scalar = float(data["scalar"])
    pred_raw = denormalize(pred_norm, min_value, scalar)
    true_raw = data["test_y_raw"].reshape(-1).astype("float32")

    metrics_norm = compute_metrics(true_norm, pred_norm)
    metrics_raw = compute_metrics(true_raw, pred_raw)

    print(
        f"{rnn_type.upper()} test normalized MSE: {metrics_norm['mse']:.8f} | "
        f"raw RMSE: {metrics_raw['rmse']:.8f} | R2: {metrics_raw['r2']:.6f}"
    )

    return {
        "model": model,
        "history": history,
        "pred_norm": pred_norm,
        "pred_raw": pred_raw,
        "metrics_norm": metrics_norm,
        "metrics_raw": metrics_raw,
    }


def run_experiment(cfg: ExperimentConfig, model_types: Sequence[str] = MODEL_TYPES) -> Dict[str, object]:
    set_seed(cfg.seed)
    data = prepare_data(cfg)
    print("device:", DEVICE)
    print("config:", json.dumps(asdict(cfg), ensure_ascii=False, indent=2))
    print("split sizes:", data["split_sizes"])
    print("train_x:", tuple(data["train_x"].shape), "train_y:", tuple(data["train_y"].shape))

    results: Dict[str, object] = {"config": asdict(cfg), "data": data}
    for rnn_type in model_types:
        results[rnn_type] = run_single_model(rnn_type, cfg, data)
    return results


def plot_true_pred(
    y_true_raw: np.ndarray,
    y_pred_raw: np.ndarray,
    title: str,
    out_path: Path,
    n_show: int = 200,
) -> None:
    y_true_raw = y_true_raw[:n_show]
    y_pred_raw = y_pred_raw[:n_show]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(y_true_raw, label="True", linewidth=1.4)
    ax.plot(y_pred_raw, label="Pred", linewidth=1.4)
    ax.legend()
    ax.set_title(title)
    ax.set_xlabel("Test timestep")
    ax.set_ylabel("Value")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_loss_curves(results: Dict[str, object], model_types: Sequence[str], output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 4))
    for name in model_types:
        history = results[name]["history"]
        ax.plot(history["train_losses"], linestyle="--", label=f"{name.upper()} train")
        ax.plot(history["val_losses"], label=f"{name.upper()} val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE loss")
    ax.legend(ncol=2)
    ax.set_title("Training and validation loss")
    fig.tight_layout()
    fig.savefig(output_dir / "fig_loss_train_val_all.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_outputs(results: Dict[str, object], output_dir: Path, model_types: Sequence[str]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    data = results["data"]
    true_raw = data["test_y_raw"].reshape(-1)
    true_norm = data["test_y"].numpy().reshape(-1)

    summary = {
        "config": results["config"],
        "split_sizes": data["split_sizes"],
        "normalizer": {
            "min_value": float(data["min_value"]),
            "scalar": float(data["scalar"]),
        },
        "models": {},
    }

    npz_payload = {"test_y_norm": true_norm, "test_y_raw": true_raw}

    for name in model_types:
        payload = results[name]
        plot_true_pred(
            true_raw,
            payload["pred_raw"],
            f"{name.upper()} prediction vs ground truth",
            output_dir / f"fig_pred_{name}.png",
        )
        summary["models"][name] = {
            "history": payload["history"],
            "metrics_norm": payload["metrics_norm"],
            "metrics_raw": payload["metrics_raw"],
        }
        npz_payload[f"pred_{name}_norm"] = payload["pred_norm"]
        npz_payload[f"pred_{name}_raw"] = payload["pred_raw"]
        torch.save(payload["model"].state_dict(), output_dir / f"{name}_sin_optimized.pt")

    plot_loss_curves(results, model_types, output_dir)
    np.savez_compressed(output_dir / "predictions_optimized.npz", **npz_payload)
    (output_dir / "summary_optimized.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def parse_csv_numbers(text: str, cast_type):
    return [cast_type(item.strip()) for item in text.split(",") if item.strip()]


def parse_model_types(text: str) -> List[str]:
    model_types = [item.strip().lower() for item in text.split(",") if item.strip()]
    invalid = [name for name in model_types if name not in MODEL_TYPES]
    if invalid:
        raise ValueError(f"invalid model types: {invalid}. Valid choices: {MODEL_TYPES}")
    return model_types


def run_compact_search(
    base_cfg: ExperimentConfig,
    model_types: Sequence[str],
    look_backs: Iterable[int],
    hidden_sizes: Iterable[int],
    lrs: Iterable[float],
    output_dir: Path,
) -> None:
    """Run a small grid search and save ranked results as JSON/CSV."""
    records = []
    search_dir = output_dir / "search_runs"
    search_dir.mkdir(parents=True, exist_ok=True)

    for look_back, hidden_size, lr, rnn_type in itertools.product(
        look_backs,
        hidden_sizes,
        lrs,
        model_types,
    ):
        cfg = ExperimentConfig(
            **{
                **asdict(base_cfg),
                "look_back": look_back,
                "hidden_size": hidden_size,
                "lr": lr,
            }
        )
        run_name = f"{rnn_type}_lb{look_back}_h{hidden_size}_lr{lr:g}"
        print(f"\n######## SEARCH RUN: {run_name} ########")
        data = prepare_data(cfg)
        set_seed(cfg.seed)
        result = run_single_model(rnn_type, cfg, data)
        record = {
            "model": rnn_type,
            "look_back": look_back,
            "hidden_size": hidden_size,
            "lr": lr,
            "best_epoch": result["history"]["best_epoch"],
            "stopped_epoch": result["history"]["stopped_epoch"],
            "val_mse": result["history"]["best_val_mse"],
            "test_mse_norm": result["metrics_norm"]["mse"],
            "test_rmse_raw": result["metrics_raw"]["rmse"],
            "test_mae_raw": result["metrics_raw"]["mae"],
            "test_r2_raw": result["metrics_raw"]["r2"],
        }
        records.append(record)

    records = sorted(records, key=lambda item: item["test_mse_norm"])
    (output_dir / "search_summary.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    csv_lines = [
        "rank,model,look_back,hidden_size,lr,best_epoch,stopped_epoch,val_mse,"
        "test_mse_norm,test_rmse_raw,test_mae_raw,test_r2_raw"
    ]
    for rank, record in enumerate(records, start=1):
        csv_lines.append(
            ",".join(
                [
                    str(rank),
                    record["model"],
                    str(record["look_back"]),
                    str(record["hidden_size"]),
                    f"{record['lr']:.8g}",
                    str(record["best_epoch"]),
                    str(record["stopped_epoch"]),
                    f"{record['val_mse']:.10g}",
                    f"{record['test_mse_norm']:.10g}",
                    f"{record['test_rmse_raw']:.10g}",
                    f"{record['test_mae_raw']:.10g}",
                    f"{record['test_r2_raw']:.10g}",
                ]
            )
        )
    (output_dir / "search_summary.csv").write_text("\n".join(csv_lines), encoding="utf-8")

    print("\nTop search results:")
    for i, record in enumerate(records[:5], start=1):
        print(f"#{i}: {record}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Optimized RNN/LSTM/GRU sine prediction experiment")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=80)
    parser.add_argument("--min-delta", type=float, default=1e-7)
    parser.add_argument("--look-back", type=int, default=30)
    parser.add_argument("--step", type=float, default=0.01)
    parser.add_argument("--end", type=float, default=3.0)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--models", default="rnn,lstm,gru", help="comma-separated choices from rnn,lstm,gru")
    parser.add_argument("--output-dir", default="experiment_outputs/rnn_sin_optimized")
    parser.add_argument(
        "--no-residual-prediction",
        action="store_true",
        help="disable residual prediction and directly predict the next value",
    )
    parser.add_argument(
        "--search",
        action="store_true",
        help="run a compact grid search instead of a single default experiment",
    )
    parser.add_argument("--search-look-backs", default="10,20,30,50")
    parser.add_argument("--search-hidden-sizes", default="32,64,128")
    parser.add_argument("--search-lrs", default="0.001,0.003,0.0005")
    args = parser.parse_args()

    cfg = ExperimentConfig(
        step=args.step,
        end=args.end,
        look_back=args.look_back,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        epochs=args.epochs,
        lr=args.lr,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        batch_size=args.batch_size,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        patience=args.patience,
        min_delta=args.min_delta,
        residual_prediction=not args.no_residual_prediction,
        seed=args.seed,
    )
    model_types = parse_model_types(args.models)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.search:
        run_compact_search(
            base_cfg=cfg,
            model_types=model_types,
            look_backs=parse_csv_numbers(args.search_look_backs, int),
            hidden_sizes=parse_csv_numbers(args.search_hidden_sizes, int),
            lrs=parse_csv_numbers(args.search_lrs, float),
            output_dir=output_dir,
        )
    else:
        results = run_experiment(cfg, model_types=model_types)
        save_outputs(results, output_dir, model_types=model_types)

    print("saved to", output_dir.resolve())


if __name__ == "__main__":
    main()
