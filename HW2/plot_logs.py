#!/usr/bin/env python3
"""从 HW2/logs 读取 CSV 并生成 matplotlib 图表。"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt

LOGS_ROOT = Path(__file__).resolve().parent / "logs"
OUT_DIR = LOGS_ROOT / "figures"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def to_float(rows: list[dict[str, str]], key: str) -> list[float]:
    return [float(r[key]) for r in rows]


def to_int_epochs(rows: list[dict[str, str]]) -> list[int]:
    return [int(float(r["epoch"])) for r in rows]


def plot_basic_style(log_subdir: str, out_stem: str) -> None:
    """读取 logs/<log_subdir>/train_log_basic.csv 与 val_log_basic.csv，与 basic 相同画法。"""
    train = read_csv(LOGS_ROOT / log_subdir / "train_log_basic.csv")
    val = read_csv(LOGS_ROOT / log_subdir / "val_log_basic.csv")
    epochs_t = to_int_epochs(train)
    epochs_v = to_int_epochs(val)
    train_loss = to_float(train, "loss")
    val_loss = to_float(val, "loss")
    train_acc = to_float(train, "accuracy")
    val_acc = to_float(val, "accuracy")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs_t, train_loss, label="train_loss", color="C0")
    ax.plot(epochs_v, val_loss, label="val_loss", color="C1")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title(f"{log_subdir}: train vs val loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"{out_stem}_loss.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs_t, train_acc, label="train_acc", color="C0")
    ax.plot(epochs_v, val_acc, label="val_acc", color="C1")
    ax.set_xlabel("epoch")
    ax.set_ylabel("accuracy")
    ax.set_title(f"{log_subdir}: train vs val accuracy")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"{out_stem}_acc.png", dpi=150)
    plt.close(fig)


def config_dirs() -> list[Path]:
    out: list[Path] = []
    skip = {"basic", "extended"}
    for p in sorted(LOGS_ROOT.iterdir()):
        if not p.is_dir() or p.name in skip:
            continue
        if (p / "pretrain_log_config.csv").is_file():
            out.append(p)
    return out


def plot_config(cfg: Path) -> None:
    name = cfg.name
    pre = read_csv(cfg / "pretrain_log_config.csv")
    tr = read_csv(cfg / "train_log_config.csv")
    va = read_csv(cfg / "val_log_config.csv")

    pre_loss = to_float(pre, "train_loss")
    pre_acc = to_float(pre, "train_acc")
    tr_loss = to_float(tr, "train_loss")
    tr_acc = to_float(tr, "train_acc")
    val_epochs = to_int_epochs(va)
    val_acc = to_float(va, "val_acc")

    n_pre = len(pre_loss)
    x_pre = list(range(1, n_pre + 1))
    x_tr = list(range(n_pre + 1, n_pre + len(tr_loss) + 1))

    safe = name.replace("/", "_")
    subdir = OUT_DIR / safe
    subdir.mkdir(parents=True, exist_ok=True)

    # pretrain_loss + train_loss
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x_pre, pre_loss, label="pretrain_loss", color="C0")
    ax.plot(x_tr, tr_loss, label="train_loss", color="C1")
    ax.set_xlabel("step (pretrain epochs 1–{}, then fine-tune)".format(n_pre))
    ax.set_ylabel("loss")
    ax.set_title(f"{name}: pretrain vs train loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(subdir / "pretrain_train_loss.png", dpi=150)
    plt.close(fig)

    # pretrain_acc + train_acc
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x_pre, pre_acc, label="pretrain_acc", color="C0")
    ax.plot(x_tr, tr_acc, label="train_acc", color="C1")
    ax.set_xlabel("step (pretrain epochs 1–{}, then fine-tune)".format(n_pre))
    ax.set_ylabel("accuracy")
    ax.set_title(f"{name}: pretrain vs train accuracy")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(subdir / "pretrain_train_acc.png", dpi=150)
    plt.close(fig)

    # val_acc + horizontal line at first val_acc
    first_acc = val_acc[0]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(val_epochs, val_acc, label="val_acc", color="C0")
    ax.axhline(y=first_acc, color="red", linestyle="--", linewidth=1.5, label=f"first val_acc = {first_acc:.4f}")
    ax.set_xlabel("epoch")
    ax.set_ylabel("val accuracy")
    ax.set_title(f"{name}: val accuracy")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(subdir / "val_acc.png", dpi=150)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "sans-serif"]
    plot_basic_style("basic", "basic")
    plot_basic_style("extended", "extended")
    for cfg in config_dirs():
        plot_config(cfg)
    print(f"图表已保存到: {OUT_DIR}")


if __name__ == "__main__":
    main()
