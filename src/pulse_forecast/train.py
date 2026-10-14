"""CPU training, validation-only checkpoint selection, and test reporting."""

import copy
import json
import platform
import random
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .data import SAMPLE_HZ, SHA256, SOURCE, SPLITS, download, prepare, read_ppg
from .model import GRUForecaster, fit_ridge, predict_ridge

CONTEXT = 50
HORIZON = 10
STRIDE = 10
HIDDEN_SIZE = 32
BATCH_SIZE = 128
LEARNING_RATE = 0.001


def predict(model, x: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return np.concatenate(
            [
                model(torch.from_numpy(x[start : start + BATCH_SIZE])).numpy()
                for start in range(0, len(x), BATCH_SIZE)
            ]
        )


def metrics(y: np.ndarray, prediction: np.ndarray, std: float) -> dict:
    error = (prediction.astype(np.float64) - y) * std
    return {
        "mae": float(np.abs(error).mean()),
        "rmse": float(np.sqrt(np.square(error).mean())),
        "rmse_by_horizon": np.sqrt(np.square(error).mean(axis=0)).tolist(),
    }


def train_model(datasets: dict, epochs: int, seed: int, threads: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    x_train, y_train = datasets["train"]
    x_val, y_val = datasets["validation"]
    model = GRUForecaster(HIDDEN_SIZE, y_train.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    history, best_state, best_epoch = [], None, 0
    best_loss = float("inf")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for x, y in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.mse_loss(model(x), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item() * len(x)
        val_loss = float(np.mean(np.square(predict(model, x_val) - y_val)))
        if not np.isfinite(val_loss):
            raise RuntimeError("Validation loss became nonfinite.")
        history.append(
            {"epoch": epoch, "train_mse": total_loss / len(x_train), "validation_mse": val_loss}
        )
        if val_loss < best_loss:
            best_loss, best_epoch = val_loss, epoch
            best_state = copy.deepcopy(model.state_dict())
        print(
            f"Epoch {epoch:02d}/{epochs}: train MSE={history[-1]['train_mse']:.4f}, "
            f"validation MSE={val_loss:.4f}",
            flush=True,
        )
    model.load_state_dict(best_state)
    return model, history, best_epoch


def plots(output_dir, history, x, y, predictions, mean, std, results):
    plt.rcParams.update(
        {"axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 160, "font.size": 10}
    )
    fig, ax = plt.subplots(figsize=(7, 3.6), layout="constrained")
    ax.plot([h["epoch"] for h in history], [h["train_mse"] for h in history], label="Training")
    ax.plot(
        [h["epoch"] for h in history], [h["validation_mse"] for h in history], label="Validation"
    )
    ax.set(xlabel="Epoch", ylabel="MSE (training-standardized units²)", title="GRU learning curve")
    ax.legend()
    fig.savefig(output_dir / "learning_curve.png")
    plt.close(fig)

    # Fixed, evenly spaced examples avoid selection by observed forecast error.
    indices = np.linspace(0, len(x) - 1, 3, dtype=int)
    fig, axes = plt.subplots(3, 1, figsize=(8, 7), layout="constrained")
    history_time = np.arange(-x.shape[1] + 1, 1) / SAMPLE_HZ
    future_time = np.arange(1, y.shape[1] + 1) / SAMPLE_HZ
    for ax, index in zip(axes, indices):
        ax.plot(history_time, x[index, :, 0] * std + mean, color="#34495e", label="History")
        ax.plot(future_time, y[index] * std + mean, color="#34495e", label="Observed future")
        for name, color in [("gru", "#007f86"), ("ridge", "#b26a00"), ("persistence", "#8c70a8")]:
            ax.plot(
                future_time,
                predictions[name][index] * std + mean,
                color=color,
                linestyle="--",
                label=name.upper(),
            )
        ax.axvline(0, color="#bbbbbb", linewidth=0.8)
        ax.set(ylabel="PPG (source units)", title=f"Test recording 07, window {index}")
    axes[0].legend(ncol=3, fontsize=8)
    axes[-1].set_xlabel("Time from forecast origin (seconds)")
    fig.savefig(output_dir / "forecasts.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 3.6), layout="constrained")
    for name in predictions:
        ax.plot(future_time, results[name]["rmse_by_horizon"], marker="o", label=name.upper())
    ax.set(
        xlabel="Forecast horizon (seconds)",
        ylabel="RMSE (source PPG units)",
        title="Held-out recording 07",
    )
    ax.legend()
    fig.savefig(output_dir / "horizon_error.png")
    plt.close(fig)


def experiment(
    records: dict, output_dir: Path, epochs=20, seed=42, threads=2, provenance=None
) -> dict:
    if epochs < 1 or threads < 1:
        raise ValueError("Epochs and threads must be positive.")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"{output_dir} is not empty. Choose a new output directory.")
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    datasets, mean, std = prepare(records, CONTEXT, HORIZON, STRIDE)
    model, history, best_epoch = train_model(datasets, epochs, seed, threads)
    x_train, y_train = datasets["train"]
    ridge = fit_ridge(x_train, y_train)
    # Test observations are accessed after model fitting and checkpoint selection.
    x_test, y_test = datasets["test"]
    predictions = {
        "persistence": np.repeat(x_test[:, -1, :], HORIZON, axis=1),
        "ridge": predict_ridge(x_test, ridge),
        "gru": predict(model, x_test),
    }
    results = {name: metrics(y_test, pred, std) for name, pred in predictions.items()}
    config = {
        "seed": seed,
        "epochs": epochs,
        "best_epoch": best_epoch,
        "threads": threads,
        "context": CONTEXT,
        "horizon": HORIZON,
        "stride": STRIDE,
        "sample_hz": SAMPLE_HZ,
        "hidden_size": HIDDEN_SIZE,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "ridge_alpha": 1.0,
        "device": "cpu",
        "normalization": {"mean": mean, "std": std},
        "splits": SPLITS,
    }
    report = {
        "config": config,
        "test_metrics": results,
        "windows": {split: len(pair[0]) for split, pair in datasets.items()},
        "parameters": sum(p.numel() for p in model.parameters()),
        "history": history,
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "matplotlib": matplotlib.__version__,
            "platform": platform.platform(),
        },
        "data": provenance or {"source": "test fixture"},
    }
    torch.save({"state_dict": model.state_dict(), "config": config}, output_dir / "model.pt")
    np.save(output_dir / "ridge.npy", ridge)
    np.savez_compressed(
        output_dir / "test_predictions.npz",
        target=y_test * std + mean,
        **{name: pred * std + mean for name, pred in predictions.items()},
    )
    plots(output_dir, history, x_test, y_test, predictions, mean, std, results)
    report["elapsed_seconds"] = time.perf_counter() - started
    (output_dir / "metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Experiment results",
        "",
        "The test metrics describe forecasting on held-out BIDMC recording 07. "
        "Errors are measured in the source PPG amplitude units.",
        "",
        "| Model | MAE | RMSE |",
        "| --- | ---: | ---: |",
    ]
    for name, scores in results.items():
        lines.append(f"| {name.upper()} | {scores['mae']:.6f} | {scores['rmse']:.6f} |")
    lines.extend(
        [
            "",
            f"Validation selected epoch {best_epoch} of {epochs}. "
            f"The model contains {report['parameters']:,} parameters.",
            "",
            "A single held-out recording cannot establish patient-level generalization. "
            "The experiment has no diagnostic or clinical endpoint.",
            "",
        ]
    )
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n" + "\n".join(lines), flush=True)
    return report


def run(data_dir: Path, output_dir: Path, epochs: int, seed: int, threads: int):
    paths = download(data_dir)
    records = {record: read_ppg(path) for record, path in paths.items()}
    provenance = {
        "source": SOURCE,
        "version": "1.0.0",
        "sha256": SHA256,
        "raw_samples_hz": 125,
        "downsampling": "nonoverlapping means of five samples",
        "samples_per_record": {record: len(signal) for record, signal in records.items()},
    }
    return experiment(records, output_dir, epochs, seed, threads, provenance)
