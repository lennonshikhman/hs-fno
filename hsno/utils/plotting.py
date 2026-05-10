from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt


def _mkdirs(out: Path):
    for name in ["rollout_curves", "prediction_examples", "error_vs_delay", "error_vs_resolution", "efficiency"]:
        (out / name).mkdir(parents=True, exist_ok=True)


def save_basic_plots(metrics_df, out_dir):
    """Save publication-audit plots from the metrics table.

    The plots use actual delay/resolution/rollout-step columns emitted by the
    evaluator rather than proxy benchmark bars.
    """
    out = Path(out_dir)
    _mkdirs(out)
    if len(metrics_df) == 0:
        return

    fig, ax = plt.subplots(figsize=(8, 4))
    metrics_df.groupby("model")["one_step_rel_l2"].mean().sort_values().plot(kind="bar", ax=ax)
    ax.set_ylabel("one-step relative L2 (physical units)")
    fig.tight_layout()
    fig.savefig(out / "prediction_examples" / "one_step_by_model.png")
    plt.close(fig)

    step_cols = [c for c in metrics_df.columns if c.startswith("rollout_step_") and c.endswith("_rel_l2")]
    if step_cols:
        step_cols = sorted(step_cols, key=lambda c: int(c.split("_")[2]))
        fig, ax = plt.subplots(figsize=(8, 4))
        for model, grp in metrics_df.groupby("model"):
            ax.plot(range(1, len(step_cols) + 1), [grp[c].mean() for c in step_cols], marker="o", label=model)
        ax.set_xlabel("rollout step")
        ax.set_ylabel("relative L2")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(out / "rollout_curves" / "rollout_error_curve.png")
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4))
    grouped = metrics_df.groupby("model")[["one_step_rel_l2", "rollout_rel_l2"]].mean().sort_values("one_step_rel_l2")
    grouped.plot(kind="bar", ax=ax)
    ax.set_ylabel("relative L2")
    fig.tight_layout()
    fig.savefig(out / "rollout_curves" / "one_step_vs_rollout_by_model.png")
    plt.close(fig)

    if "mean_delay" in metrics_df.columns:
        fig, ax = plt.subplots(figsize=(7, 4))
        for model, grp in metrics_df.groupby("model"):
            g = grp.sort_values("mean_delay")
            ax.plot(g["mean_delay"], g["one_step_rel_l2"], marker="o", linestyle="none", label=model, alpha=0.75)
        ax.set_xlabel("delay / effective delay")
        ax.set_ylabel("one-step relative L2")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(out / "error_vs_delay" / "error_vs_delay.png")
        plt.close(fig)

    if "eval_nx" in metrics_df.columns:
        fig, ax = plt.subplots(figsize=(7, 4))
        for model, grp in metrics_df.groupby("model"):
            g = grp.groupby("eval_nx")["history_rel_l2"].mean().sort_index()
            ax.plot(g.index, g.values, marker="o", label=model)
        ax.set_xlabel("spatial resolution nx")
        ax.set_ylabel("history relative L2")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(out / "error_vs_resolution" / "error_vs_resolution.png")
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(metrics_df["inference_time_per_step"], metrics_df["one_step_rel_l2"], s=30)
    ax.set_xlabel("inference time per step (s)")
    ax.set_ylabel("one-step relative L2")
    fig.tight_layout()
    fig.savefig(out / "efficiency" / "time_vs_error.png")
    plt.close(fig)

    if {"speedup_vs_solver", "peak_memory_mb"}.issubset(metrics_df.columns):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        metrics_df.groupby("model")["speedup_vs_solver"].mean().sort_values().plot(kind="bar", ax=axes[0])
        axes[0].set_ylabel("speedup vs solver")
        metrics_df.groupby("model")["peak_memory_mb"].mean().sort_values().plot(kind="bar", ax=axes[1])
        axes[1].set_ylabel("peak memory (MB)")
        fig.tight_layout()
        fig.savefig(out / "efficiency" / "speedup_memory_by_model.png")
        plt.close(fig)

    osc_cols = [c for c in ["amplitude_error", "dominant_frequency_error", "phase_drift"] if c in metrics_df.columns]
    if osc_cols:
        osc_df = metrics_df[metrics_df[osc_cols].abs().sum(axis=1) > 0]
        if len(osc_df):
            fig, ax = plt.subplots(figsize=(8, 4))
            osc_df.groupby("model")[osc_cols].mean().plot(kind="bar", ax=ax)
            ax.set_ylabel("oscillatory metric")
            fig.tight_layout()
            fig.savefig(out / "prediction_examples" / "phase_amplitude_metrics.png")
            plt.close(fig)


def save_prediction_example(model, dataset, device, out_dir, benchmark: str, model_name: str):
    """Plot one autoregressive prediction trace against the reference trajectory."""
    import torch
    if len(dataset) == 0:
        return
    base_dataset = getattr(dataset, "dataset", dataset)
    sample = dataset[0]
    b = {k: v.unsqueeze(0).to(device) for k, v in sample.items() if hasattr(v, "to")}
    model.eval()
    preds = []
    with torch.no_grad():
        h = b["history"]
        targets = b.get("rollout_targets", b["target_history"].unsqueeze(1))
        kmax = int(targets.shape[1])
        # rollout_targets has shape [B,K,H,C,...]; decode as [B*K,H,C,...]
        # because ChannelNormalizer expects the channel axis at position 2 for
        # history tensors, then restore the rollout axis.
        bt, kt = targets.shape[:2]
        refs = base_dataset.normalizer.decode(targets.reshape(bt * kt, *targets.shape[2:]))
        refs = refs.reshape(bt, kt, *refs.shape[1:]).cpu().numpy()[0, :, -1]
        for _ in range(kmax):
            h = model(h, b["cond"], b["static"])
            preds.append(base_dataset.normalizer.decode(h).cpu().numpy()[0, -1])
    import numpy as np
    pred_obs = np.asarray(preds).reshape(kmax, -1).mean(axis=1)
    ref_obs = refs.reshape(kmax, -1).mean(axis=1)
    out = Path(out_dir) / "prediction_examples"
    out.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(range(1, kmax + 1), ref_obs, marker="o", label="reference")
    ax.plot(range(1, kmax + 1), pred_obs, marker="s", label="prediction")
    ax.set_xlabel("rollout step")
    ax.set_ylabel("spatial mean observable")
    ax.set_title(f"{benchmark}: {model_name}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / f"{benchmark}_{model_name}_prediction_example.png")
    plt.close(fig)
