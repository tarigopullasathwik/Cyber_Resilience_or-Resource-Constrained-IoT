"""
Figure generation for the experiment results (component 9 of the research plan).

Produces, under ``figures/``:

* ``architecture.png``            system architecture diagram
* ``closed_loop_flowchart.png``   closed-loop response flowchart
* ``score_timeline.png``          anomaly score vs adaptive threshold, per device
* ``threshold_adaptation.png``    threshold of each system over time
* ``confusion_matrices.png``      confusion matrices of the three systems
* ``system_comparison.png``       headline metric comparison
* ``latency_by_attack.png``       detection rate / latency per attack type
* ``actions_outcomes.png``        containment actions and verification outcomes
* ``resources.png``               simulated edge resources + centralised baseline
* ``severity_distribution.png``   severity decomposition
* ``learning_guard.png``          baseline-update modes (anti-poisoning evidence)
* ``robustness.png``              FPR on drifted devices, known vs held-out recall

Matplotlib runs with the ``Agg`` backend so the module works headlessly in
VS Code, in tests, and on a machine with no display.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import SYSTEM_LABELS, SYSTEM_PROPOSED, SYSTEM_STATIC, SYSTEMS  # noqa: E402

COLORS = {
    SYSTEM_STATIC: "#c0504d",
    "S2_adaptive_no_containment": "#4f81bd",
    SYSTEM_PROPOSED: "#4caf50",
}
DPI = 130


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# Static documentation diagrams
# --------------------------------------------------------------------------- #
def architecture_diagram(path: Path) -> Path:
    """Block diagram of the simulated edge architecture."""
    fig, ax = plt.subplots(figsize=(12.5, 8.2))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    def box(x, y, w, h, text, face="#eef3fb", edge="#2f4b7c", size=9, weight="normal"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.6,rounding_size=1.2",
                                    linewidth=1.3, facecolor=face, edgecolor=edge))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size,
                weight=weight, wrap=True)

    def arrow(x1, y1, x2, y2, color="#2f4b7c", style="-|>", text="", offset=(0, 1.4), ls="-"):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=14,
                                     linewidth=1.2, color=color, linestyle=ls))
        if text:
            ax.text((x1 + x2) / 2 + offset[0], (y1 + y2) / 2 + offset[1], text, fontsize=7.5,
                    color=color, ha="center")

    ax.text(50, 97, "Adaptive Edge Cyber-Resilience - system architecture",
            ha="center", fontsize=13, weight="bold")
    ax.text(50, 93.2, "all devices, traffic, attacks and containment actions are simulated in-process",
            ha="center", fontsize=8.5, style="italic", color="#666666")

    # Devices
    box(2, 68, 30, 20, "Heterogeneous IoT fleet (>= 20 devices)\n"
                       "cameras | thermostats | plugs | locks | meters\nwearables | PLCs | assistants | "
                       "lights | sensors\n"
                       "per-device normal profiles + legitimate drift\n"
                       "synthetic attacks: flood, port-scan, brute-force,\n"
                       "exfiltration, spoofing, stealth-exfiltration", face="#fdf3e3", edge="#b8860b", size=8.2)

    # Edge gateway
    ax.add_patch(FancyBboxPatch((35, 16), 44, 72, boxstyle="round,pad=1.0,rounding_size=1.6",
                                linewidth=1.8, facecolor="#f4f7f4", edgecolor="#2e7d32"))
    ax.text(57, 85.6, "EDGE GATEWAY (local, resource-constrained)", ha="center", fontsize=10,
            weight="bold", color="#2e7d32")
    box(37, 74, 40, 9, "B. edge monitoring + feature extraction\n"
                       "8 lightweight numeric features, raw telemetry never leaves the gateway",
        face="#ffffff", edge="#2e7d32", size=8)
    box(37, 62, 40, 9.5, "C. adaptive anomaly detector\n"
                         "robust z-score + IsolationForest (benign-only fit)\n"
                         "adaptive threshold, frozen while a device is suspicious",
        face="#ffffff", edge="#2e7d32", size=8)
    box(37, 51, 40, 8.5, "E. threat assessment\n"
                         "severity = f(anomaly magnitude, confidence, criticality, context)",
        face="#ffffff", edge="#2e7d32", size=8)
    box(37, 40, 40, 8.5, "F. containment policy + controller\n"
                         "OBSERVE -> RATE_LIMIT -> REAUTHENTICATE -> QUARANTINE\n"
                         "cooldown | criticality guard | resource guard | probes",
        face="#ffffff", edge="#2e7d32", size=8)
    box(23.4, 24, 27, 10.5, "G. response verification\nSUCCESS | FAILURE | INCONCLUSIVE\n"
                            "score + offending-indicator reduction",
        face="#ffffff", edge="#2e7d32", size=8)
    box(53, 24, 24, 10.5, "H. adaptive learning\nbaseline update + anti-poisoning\n"
                          "freeze | drift gates | containment block",
        face="#ffffff", edge="#2e7d32", size=8)
    box(37, 17, 40, 5.4, "simulated edge resource model: CPU / memory / uplink envelope "
                         "(labelled 'simulated')", face="#eef7ee", edge="#2e7d32", size=7.6)

    # Cloud comparison (dashed)
    box(85, 52, 13, 24, "centralised\nbaseline\n(simulated):\nraw telemetry\nupstream\n"
                        "-> higher\nlatency and\nbandwidth", face="#f7f7f7", edge="#999999", size=7.6)

    # Logs / dashboard
    box(2, 2, 96, 11, "logs: telemetry | decision log (score, threshold, severity, action) | "
                      "containment actions | verification outcomes | baseline updates | resources\n"
                      "-> metrics.py (accuracy, precision, recall, F1, FPR, latencies, containment "
                      "success, recovery time, resources) -> Streamlit dashboard + CSV/JSON export",
        face="#f0f0f0", edge="#555555", size=8)

    arrow(32, 80, 37, 78, text="telemetry (local only)")
    arrow(37, 70, 32, 70, color="#b8860b", text="containment (simulated)")
    arrow(57, 74, 57, 71.5)
    arrow(57, 62, 57, 59.5)
    arrow(57, 51, 57, 48.5)
    arrow(51, 40, 40, 34.5)
    arrow(66, 40, 66, 34.5)
    arrow(50.4, 29.3, 53, 29.3, text="verified benign -> adapt")
    arrow(53, 26, 45, 26, color="#2e7d32", style="<|-", text="baseline / threshold update")
    arrow(37, 60, 30, 60, color="#888888", style="<|-", text="")
    arrow(77, 66, 85, 66, color="#999999", ls="--", text="optional (not used)")
    ax.text(30, 63.5, "no raw telemetry to the cloud", fontsize=7, color="#777777", ha="center")
    ax.text(19.5, 22, "verification feeds back into\npolicy and learning", fontsize=7,
            color="#2e7d32", ha="center")
    return _save(fig, path)


def closed_loop_flowchart(path: Path) -> Path:
    """Flowchart of the closed-loop detection/response/recovery cycle."""
    fig, ax = plt.subplots(figsize=(12.0, 8.6))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    def box(x, y, w, h, text, face="#eef3fb", edge="#2f4b7c", size=8.4, bold=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.5,rounding_size=1.0",
                                    linewidth=1.3, facecolor=face, edgecolor=edge))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size,
                weight=("bold" if bold else "normal"))

    def decision(x, y, w, h, text, size=8.2):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.5,rounding_size=3.0",
                                    linewidth=1.5, facecolor="#fff8e1", edgecolor="#c79100"))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size)

    def arrow(x1, y1, x2, y2, text="", color="#2f4b7c", dx=0, dy=1.2, ls="-"):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=13,
                                     linewidth=1.1, color=color, linestyle=ls))
        if text:
            ax.text((x1 + x2) / 2 + dx, (y1 + y2) / 2 + dy, text, fontsize=7.4, color=color,
                    ha="center")

    ax.text(50, 97.5, "Closed-loop response flow (one simulation step per device)",
            ha="center", fontsize=13, weight="bold")
    box(3, 86, 27, 8, "1. observe device telemetry\n(allowance-normalised)", size=8.2)
    box(3, 74, 27, 8, "2. extract 8 edge features\n(local only)", size=8.2)
    box(3, 62, 27, 8, "3. fused anomaly score\n(statistical + IsolationForest)", size=8.2)
    decision(3, 50, 27, 8.5, "4. score > adaptive\nthreshold ?", size=8.2)
    box(3, 38, 27, 8, "5. benign observation:\nupdate baseline at low rate", size=8.2)
    decision(3, 26, 27, 8.5, "6. frozen? OR our own\ncontainment active?", size=8.2)
    box(3, 14, 27, 8, "7. hold baseline\n(poisoning guard, logged)", size=8.2)

    box(37, 86, 27, 8, "8. threat assessment\nseverity 0..100", size=8.2)
    box(37, 74, 27, 8, "9. policy chooses level\n+ safeguards", size=8.2)
    box(37, 62, 27, 8, "10. apply action (simulated)\nopen verification window", size=8.2)
    box(37, 50, 27, 8.5, "11. observe window:\nscore + offending indicators", size=8.2)
    decision(37, 37.5, 27, 9, "12. outcome?", size=8.2)
    box(37, 25, 27, 8, "13a. FAILURE -> escalate\n(<= max level)", size=8.2)
    box(37, 13, 27, 8, "13b. INCONCLUSIVE ->\nno escalation", size=8.2)

    box(70, 86, 27, 8, "14. SUCCESS ->\nrecovery_progress += 1", size=8.2)
    decision(70, 73, 27, 9, "15. enough benign\nprogress?", size=8.2)
    box(70, 60, 27, 8, "16. probe: step down one\nlevel, watch the window", size=8.2)
    decision(70, 47.5, 27, 9, "17. anomaly returns\nin probe?", size=8.2)
    box(70, 35, 27, 8, "18a. RE_APPLY previous\naction (logged)", size=8.2)
    box(70, 23, 27, 8, "18b. level 0 -> RESTORE\n+ recovery time", size=8.2)
    box(70, 10, 27, 8, "19. learn confirmed-benign\nbehaviour safely", size=8.2)

    arrow(16.5, 86, 16.5, 82)
    arrow(16.5, 74, 16.5, 70)
    arrow(16.5, 62, 16.5, 58.5)
    arrow(16.5, 50, 16.5, 46, text="no (benign)", dx=-8.5)
    arrow(16.5, 26, 16.5, 22, text="yes", dx=4.5, dy=-1.0)
    arrow(3, 54.2, -0.4, 54.2, text="", color="#2f4b7c")
    arrow(30, 54.2, 37, 88.5, text="yes (anomalous)", dx=6.0, dy=-3.0)
    arrow(50.5, 86, 50.5, 82)
    arrow(50.5, 74, 50.5, 70)
    arrow(50.5, 62, 50.5, 58.5)
    arrow(50.5, 50, 50.5, 46.5)
    arrow(50.5, 37.5, 50.5, 33, text="FAILURE", dx=7.5)
    arrow(42, 42, 42, 29, text="", color="#c0504d", ls="--")
    arrow(50.5, 25, 83.5, 90, color="#c0504d", text="escalate (step 10)", dx=-4, dy=2.5)
    arrow(64, 42, 70, 90, text="SUCCESS", dx=-3, dy=4.0, color="#2e7d32")
    arrow(83.5, 86, 83.5, 82)
    arrow(83.5, 73, 83.5, 68)
    arrow(83.5, 60, 83.5, 56.5, text="yes", dx=4.0)
    arrow(83.5, 47.5, 83.5, 43, text="no", dx=4.0)
    arrow(83.5, 47.5, 83.5, 31, text="yes", dx=4.0)
    arrow(70, 27, 64, 66, text="back to the policy", dx=-1, dy=4.0, color="#c0504d")
    arrow(30, 18, 70, 14, text="when the device stays benign, the baseline learns it",
          dx=0, dy=2.0, color="#2e7d32", ls="--")
    ax.text(50, 4.5, "Every step is logged: score, threshold, severity factors, action, "
                     "verification outcome, baseline update, recovery.", ha="center", fontsize=8,
            style="italic", color="#555555")
    return _save(fig, path)


# --------------------------------------------------------------------------- #
# Result-driven charts
# --------------------------------------------------------------------------- #
def _system_decision_log(result: Any, system: str) -> pd.DataFrame:
    return result.decision_logs.get(system, pd.DataFrame())


def score_timeline(result: Any, path: Path, max_devices: int = 4) -> Path:
    decisions = _system_decision_log(result, SYSTEM_PROPOSED)
    if decisions.empty:
        return path
    attacked = (decisions[decisions["ground_truth"] == 1]
                .groupby("device_id")["step"].size().sort_values(ascending=False))
    devices = list(attacked.index[:max_devices])
    if not devices:
        return path
    fig, axes = plt.subplots(len(devices), 1, figsize=(11, 2.4 * len(devices)), sharex=True)
    if len(devices) == 1:
        axes = [axes]
    actions = result.systems[SYSTEM_PROPOSED].actions
    for ax, device in zip(axes, devices):
        frame = decisions[decisions["device_id"] == device].sort_values("step")
        ax.plot(frame["step"], frame["anomaly_score"], linewidth=1.1, color="#1f3b73",
                label="anomaly score")
        ax.plot(frame["step"], frame["threshold"], linewidth=1.2, color="#e08a1e",
                linestyle="--", label="adaptive threshold")
        attacks = frame[frame["ground_truth"] == 1]
        for _, row in attacks.iterrows():
            ax.axvspan(row["step"] - 0.5, row["step"] + 0.5, color="#d9534f", alpha=0.16)
        if not actions.empty:
            device_actions = actions[(actions["device_id"] == device) & (actions["level"] >= 1)]
            for _, row in device_actions.iterrows():
                ax.axvline(row["step"], color="#2e7d32", alpha=0.45, linewidth=0.9)
        ax.set_ylabel("score")
        attack_type = attacks["attack_type"].iloc[0] if not attacks.empty else "none"
        crit = attacks["ground_truth"].iloc[0] if not attacks.empty else 0
        ax.set_title(f"{device} - label: {attack_type} (red bands = attack steps, "
                     "green lines = containment actions)", fontsize=8.5)
        ax.grid(alpha=0.25)
    axes[0].legend(fontsize=7.5, loc="upper left")
    axes[-1].set_xlabel("simulation step")
    fig.suptitle("Anomaly score vs adaptive threshold (proposed system S3)", fontsize=11)
    return _save(fig, path)


def threshold_adaptation(result: Any, path: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for system in SYSTEMS:
        decisions = _system_decision_log(result, system)
        if decisions.empty:
            continue
        grouped = decisions.groupby("step")["threshold"].first()
        axes[0].plot(grouped.index, grouped.values, linewidth=1.3,
                     color=COLORS.get(system, "#333333"), label=SYSTEM_LABELS.get(system, system))
    axes[0].set_title("Adaptive threshold over time (S1 stays frozen)", fontsize=9.5)
    axes[0].set_xlabel("simulation step")
    axes[0].set_ylabel("fused anomaly score threshold")
    axes[0].legend(fontsize=7.5)
    axes[0].grid(alpha=0.25)

    decisions = _system_decision_log(result, SYSTEM_PROPOSED)
    if not decisions.empty:
        benign = decisions[decisions["ground_truth"] == 0]["anomaly_score"].dropna()
        attack = decisions[decisions["ground_truth"] == 1]["anomaly_score"].dropna()
        axes[1].hist(benign, bins=40, alpha=0.75, label=f"benign (n={benign.size})", color="#4f81bd")
        axes[1].hist(attack, bins=40, alpha=0.75, label=f"attack (n={attack.size})", color="#c0504d")
        threshold_value = float(decisions["threshold"].median())
        axes[1].axvline(threshold_value, color="#e08a1e", linestyle="--",
                        label=f"threshold (median {threshold_value:.2f})")
    axes[1].set_title("Score separation, proposed system S3", fontsize=9.5)
    axes[1].set_xlabel("fused anomaly score")
    axes[1].set_ylabel("device-steps")
    axes[1].legend(fontsize=7.5)
    axes[1].grid(alpha=0.25)
    return _save(fig, path)


def confusion_matrices(result: Any, path: Path) -> Path:
    fig, axes = plt.subplots(1, len(SYSTEMS), figsize=(4.2 * len(SYSTEMS), 3.9))
    if len(SYSTEMS) == 1:
        axes = [axes]
    for ax, system in zip(axes, SYSTEMS):
        metrics = result.systems.get(system)
        if metrics is None:
            continue
        summary = metrics.summary
        matrix = np.array([[summary["tn"], summary["fp"]], [summary["fn"], summary["tp"]]], dtype=float)
        image = ax.imshow(matrix, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{int(matrix[i, j])}", ha="center", va="center", fontsize=11,
                        color="black")
        ax.set_xticks([0, 1], ["pred benign", "pred anomaly"])
        ax.set_yticks([0, 1], ["true benign", "true attack"])
        ax.set_title(SYSTEM_LABELS.get(system, system), fontsize=9)
        ax.set_xlabel(f"FPR={summary['false_positive_rate']:.3f}  "
                      f"recall={summary['recall']:.3f}", fontsize=8)
    fig.suptitle("Confusion matrices (device-steps, post-calibration window)", fontsize=11)
    return _save(fig, path)


def system_comparison(result: Any, path: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    metric_keys = ["accuracy", "precision", "recall", "f1", "false_positive_rate"]
    labels = [SYSTEM_LABELS.get(s, s).split(" (")[0] for s in SYSTEMS]
    x = np.arange(len(metric_keys))
    width = 0.26
    for i, system in enumerate(SYSTEMS):
        metrics = result.systems.get(system)
        if metrics is None:
            continue
        values = [float(metrics.summary.get(k) or 0.0) for k in metric_keys]
        bars = axes[0].bar(x + i * width, values, width, label=labels[i],
                           color=COLORS.get(system, "#333333"))
        axes[0].bar_label(bars, fmt="%.2f", fontsize=6.5, padding=1)
    axes[0].set_xticks(x + width, metric_keys, fontsize=8)
    axes[0].set_ylim(0, 1.15)
    axes[0].set_title("Detection quality (device-steps)", fontsize=9.5)
    axes[0].legend(fontsize=7.5)
    axes[0].grid(alpha=0.25, axis="y")

    second_keys = ["detection_latency_mean_steps", "response_latency_mean_steps",
                   "containment_success_rate", "recovery_time_mean_steps"]
    x2 = np.arange(len(second_keys))
    for i, system in enumerate(SYSTEMS):
        metrics = result.systems.get(system)
        if metrics is None:
            continue
        values = []
        for key in second_keys:
            value = metrics.summary.get(key)
            values.append(float(value) if value is not None and np.isfinite(value) else 0.0)
        bars = axes[1].bar(x2 + i * width, values, width, label=labels[i],
                           color=COLORS.get(system, "#333333"))
        axes[1].bar_label(bars, fmt="%.1f", fontsize=6.5, padding=1)
    axes[1].set_xticks(x2 + width, ["detect lat.\n(steps)", "response lat.\n(steps)",
                                    "containment\nsuccess", "recovery time\n(steps)"], fontsize=7.5)
    axes[1].set_title("Latency, containment and recovery", fontsize=9.5)
    axes[1].legend(fontsize=7.5)
    axes[1].grid(alpha=0.25, axis="y")
    fig.suptitle(f"System comparison - scenario {result.scenario}, seed {result.seed}", fontsize=11)
    return _save(fig, path)


def latency_by_attack(result: Any, path: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    for system in SYSTEMS:
        metrics = result.systems.get(system)
        if metrics is None or metrics.per_attack_type.empty:
            continue
        frame = metrics.per_attack_type
        axes[0].plot(frame["attack_type"], frame["detection_rate"], marker="o",
                     color=COLORS.get(system, "#333"), label=SYSTEM_LABELS.get(system, system))
        axes[1].plot(frame["attack_type"], frame["detection_latency_mean_steps"], marker="o",
                     color=COLORS.get(system, "#333"), label=SYSTEM_LABELS.get(system, system))
    axes[0].set_title("Detection rate per attack type (episodes)", fontsize=9.5)
    axes[0].set_ylim(-0.05, 1.05)
    axes[1].set_title("Mean detection latency per attack type (steps)", fontsize=9.5)
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.tick_params(axis="x", labelrotation=25, labelsize=8)
        ax.legend(fontsize=7.5)
    return _save(fig, path)


def actions_outcomes(result: Any, path: Path) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    metrics = result.systems.get(SYSTEM_PROPOSED)
    actions = metrics.actions if metrics is not None else pd.DataFrame()
    if not actions.empty:
        counts = actions[actions["level"] >= 1]["action"].value_counts()
        axes[0].bar(counts.index, counts.values, color="#2e7d32")
        for i, value in enumerate(counts.values):
            axes[0].text(i, value, str(int(value)), ha="center", va="bottom", fontsize=8)
    axes[0].set_title("Containment actions applied (S3)", fontsize=9.5)
    axes[0].grid(alpha=0.25, axis="y")
    axes[0].tick_params(axis="x", labelrotation=20, labelsize=8)

    verification = metrics.verification if metrics is not None else pd.DataFrame()
    if not verification.empty:
        counts = verification["outcome"].value_counts()
        axes[1].bar(counts.index, counts.values,
                    color=["#2e7d32" if k == "SUCCESS" else "#c0504d" if k == "FAILURE" else "#999999"
                           for k in counts.index])
        for i, value in enumerate(counts.values):
            axes[1].text(i, value, str(int(value)), ha="center", va="bottom", fontsize=8)
    axes[1].set_title("Response verification outcomes (S3)", fontsize=9.5)
    axes[1].grid(alpha=0.25, axis="y")

    labels, values = [], []
    for system in SYSTEMS:
        summary = result.systems[system].summary if system in result.systems else {}
        labels.append(SYSTEM_LABELS.get(system, system).split(" (")[0])
        values.append(float(summary.get("unnecessary_quarantines") or 0.0))
    axes[2].bar(labels, values, color="#8e44ad")
    for i, value in enumerate(values):
        axes[2].text(i, value, str(int(value)), ha="center", va="bottom", fontsize=8)
    axes[2].set_title("Unnecessary quarantines (ground truth benign)", fontsize=9.5)
    axes[2].grid(alpha=0.25, axis="y")
    axes[2].tick_params(axis="x", labelrotation=15, labelsize=7.5)
    return _save(fig, path)


def resources_figure(result: Any, path: Path) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(13, 7.0))
    for system in SYSTEMS:
        resources = result.resource_logs.get(system, pd.DataFrame())
        if resources.empty:
            continue
        color = COLORS.get(system, "#333333")
        axes[0][0].plot(resources["step"], resources["cpu_utilisation"], linewidth=1.0,
                        color=color, label=SYSTEM_LABELS.get(system, system))
        axes[0][1].plot(resources["step"], resources["memory_used_mb"], linewidth=1.0, color=color)
    axes[0][0].set_title("Simulated edge CPU utilisation", fontsize=9.5)
    axes[0][0].axhline(1.0, color="#c0504d", linestyle="--", linewidth=0.9, label="capacity")
    axes[0][1].set_title("Simulated edge memory footprint (MB)", fontsize=9.5)
    axes[0][0].legend(fontsize=7)
    for row in axes:
        for ax in row:
            ax.grid(alpha=0.25)

    resources = result.resource_logs.get(SYSTEM_PROPOSED, pd.DataFrame())
    if not resources.empty:
        axes[1][0].plot(resources["step"], resources["uplink_kbps"], linewidth=1.0,
                        color="#2e7d32", label="edge summary uplink (simulated)")
        axes[1][0].plot(resources["step"], resources["cloud_uplink_kbps"], linewidth=1.0,
                        color="#c0504d", label="centralised raw-telemetry uplink (simulated)")
        axes[1][0].set_yscale("log")
        axes[1][0].set_title("Simulated uplink bandwidth per step (log scale)", fontsize=9.5)
        axes[1][0].legend(fontsize=7)
        axes[1][1].plot(resources["step"], resources["edge_latency_ms"], linewidth=1.0,
                        color="#2e7d32", label="edge processing latency (simulated)")
        axes[1][1].plot(resources["step"], resources["cloud_latency_ms"], linewidth=1.0,
                        color="#c0504d", label="centralised processing latency (simulated)")
        axes[1][1].set_title("Simulated processing latency (ms)", fontsize=9.5)
        axes[1][1].legend(fontsize=7)
    for ax in axes[1]:
        ax.set_xlabel("simulation step")
        ax.grid(alpha=0.25)
    fig.suptitle("Resource behaviour - 'simulated' values come from the edge resource model, "
                 "not from host measurements", fontsize=10)
    return _save(fig, path)


def severity_distribution(result: Any, path: Path) -> Path:
    decisions = _system_decision_log(result, SYSTEM_PROPOSED)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.3))
    if decisions.empty:
        return _save(fig, path)
    anomalous = decisions[decisions["is_anomalous"] == 1]
    if not anomalous.empty:
        axes[0].hist(anomalous["severity"], bins=30, color="#b06f00", alpha=0.85)
        axes[0].axvline(25, color="#4f81bd", linestyle="--", label="OBSERVE boundary")
        axes[0].axvline(50, color="#e08a1e", linestyle="--", label="RATE_LIMIT boundary")
        axes[0].axvline(75, color="#c0504d", linestyle="--", label="REAUTH boundary")
        axes[0].legend(fontsize=7.5)
    axes[0].set_title("Severity of anomalous device-steps (with action bands)", fontsize=9.5)
    axes[0].set_xlabel("severity (0-100)")
    axes[0].set_ylabel("device-steps")
    axes[0].grid(alpha=0.25)

    scatter = axes[1].scatter(decisions["anomaly_score"], decisions["severity"], s=4,
                              c=decisions["criticality_factor"], cmap="viridis", alpha=0.6)
    axes[1].set_xlabel("anomaly score")
    axes[1].set_ylabel("severity")
    axes[1].set_title("Severity vs anomaly score (colour = criticality factor)", fontsize=9.5)
    axes[1].grid(alpha=0.25)
    fig.colorbar(scatter, ax=axes[1], label="criticality factor")
    return _save(fig, path)


def learning_guard(result: Any, path: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.3))
    modes = ["clean", "drift", "drift_release", "blocked_anomalous", "blocked_frozen",
             "blocked_containment"]
    x = np.arange(len(modes))
    width = 0.26
    for i, system in enumerate(SYSTEMS):
        frame = result.learning_logs.get(system, pd.DataFrame())
        if frame.empty:
            continue
        counts = frame["mode"].value_counts()
        values = [int(counts.get(mode, 0)) for mode in modes]
        axes[0].bar(x + i * width, values, width, label=SYSTEM_LABELS.get(system, system),
                    color=COLORS.get(system, "#333"))
    axes[0].set_xticks(x + width, modes, fontsize=7.5, rotation=25)
    axes[0].set_yscale("symlog")
    axes[0].set_title("Baseline-update modes (anti-poisoning evidence)", fontsize=9.5)
    axes[0].legend(fontsize=7)
    axes[0].grid(alpha=0.25, axis="y")

    decisions = _system_decision_log(result, SYSTEM_PROPOSED)
    if not decisions.empty:
        shifted = decisions[decisions["benign_shift"] == True]  # noqa: E712
        if not shifted.empty:
            for device_id, frame in shifted.groupby("device_id"):
                frame = frame.sort_values("step")
                axes[1].plot(frame["step"], frame["anomaly_score"], linewidth=1.0, label=device_id)
            step = float(shifted["step"].min())
            axes[1].axvline(step, color="#c0504d", linestyle="--", linewidth=1.0,
                            label="legitimate behaviour shift")
            axes[1].legend(fontsize=6.5, ncol=3)
        axes[1].set_title("Legitimate drift: score of shifted devices must fall back", fontsize=9.5)
        axes[1].set_xlabel("simulation step")
        axes[1].set_ylabel("anomaly score")
        axes[1].grid(alpha=0.25)
    return _save(fig, path)


def robustness_figure(result: Any, path: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.3))
    labels = [SYSTEM_LABELS.get(s, s).split(" (")[0] for s in SYSTEMS]
    x = np.arange(len(SYSTEMS))
    width = 0.36
    stable = [float(result.systems[s].summary.get("benign_stable_fpr") or 0.0) for s in SYSTEMS]
    shifted = [float(result.systems[s].summary.get("benign_shift_fpr") or 0.0) for s in SYSTEMS]
    axes[0].bar(x - width / 2, stable, width, label="stable benign devices", color="#4f81bd")
    axes[0].bar(x + width / 2, shifted, width, label="legitimately drifted devices", color="#c0504d")
    axes[0].set_xticks(x, labels, fontsize=8)
    axes[0].set_title("False-positive rate: stable vs drifted benign traffic", fontsize=9.5)
    axes[0].legend(fontsize=7.5)
    axes[0].grid(alpha=0.25, axis="y")

    known = [float(result.systems[s].summary.get("known_attack_pointwise_recall") or 0.0)
             for s in SYSTEMS]
    held = [float(result.systems[s].summary.get("held_out_pointwise_recall") or 0.0)
            for s in SYSTEMS]
    axes[1].bar(x - width / 2, known, width, label="development attack patterns", color="#2e7d32")
    axes[1].bar(x + width / 2, held, width, label="held-out (unseen) patterns", color="#8e44ad")
    axes[1].set_xticks(x, labels, fontsize=8)
    axes[1].set_ylim(0, 1.1)
    axes[1].set_title("Recall on development vs held-out attack patterns", fontsize=9.5)
    axes[1].legend(fontsize=7.5)
    axes[1].grid(alpha=0.25, axis="y")
    return _save(fig, path)


def device_history(result: Any, path: Path, n_devices: int = 3) -> Path:
    decisions = _system_decision_log(result, SYSTEM_PROPOSED)
    if decisions.empty:
        return path
    attacked = (decisions[decisions["ground_truth"] == 1]
                .groupby("device_id")["step"].size().sort_values(ascending=False))
    devices = list(attacked.index[:n_devices])
    if not devices:
        return path
    fig, axes = plt.subplots(len(devices), 1, figsize=(11, 2.5 * len(devices)), sharex=True)
    if len(devices) == 1:
        axes = [axes]
    for ax, device in zip(axes, devices):
        frame = decisions[decisions["device_id"] == device].sort_values("step")
        ax.plot(frame["step"], frame["observed_packet_rate"], linewidth=1.0, color="#1f3b73",
                label="observed packet rate (after allowance)")
        ax.set_ylabel("packets/s")
        twin = ax.twinx()
        twin.plot(frame["step"], frame["allowance"], linewidth=1.0, color="#8e44ad",
                  linestyle=":", label="containment allowance")
        twin.set_ylim(-0.05, 1.05)
        twin.set_ylabel("allowance")
        ax.set_title(f"{device}: observed behaviour and containment history", fontsize=8.5)
        ax.grid(alpha=0.25)
    axes[0].legend(fontsize=7, loc="upper right")
    axes[-1].set_xlabel("simulation step")
    fig.suptitle("Device behavioural history (proposed system S3)", fontsize=11)
    return _save(fig, path)


def make_all_figures(result: Any, figures_dir: Path, docs_dir: Optional[Path] = None) -> Dict[str, Path]:
    """
    Generate every figure for one experiment result and return their paths.

    Result-dependent figures go into ``figures_dir`` (one directory per scenario,
    so scenarios cannot overwrite each other). The two scenario-independent
    documentation diagrams are also written to ``docs_dir`` (``figures/``) so the
    README can link them with stable paths.
    """
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    written: Dict[str, Path] = {}

    def _run(name: str, function, *args) -> None:
        path = figures_dir / name
        try:
            written[name] = function(*args, path)
        except Exception as exc:  # pragma: no cover - figures must never break a run
            print(f"[figures] skipped {name}: {exc}")

    if docs_dir is not None:
        docs_dir = Path(docs_dir)
        for name, function in (("architecture.png", architecture_diagram),
                               ("closed_loop_flowchart.png", closed_loop_flowchart)):
            try:
                written[name] = function(docs_dir / name)
            except Exception as exc:  # pragma: no cover
                print(f"[figures] skipped {name}: {exc}")

    _run("score_timeline.png", score_timeline, result)
    _run("threshold_adaptation.png", threshold_adaptation, result)
    _run("confusion_matrices.png", confusion_matrices, result)
    _run("system_comparison.png", system_comparison, result)
    _run("latency_by_attack.png", latency_by_attack, result)
    _run("actions_outcomes.png", actions_outcomes, result)
    _run("resources.png", resources_figure, result)
    _run("severity_distribution.png", severity_distribution, result)
    _run("learning_guard.png", learning_guard, result)
    _run("robustness.png", robustness_figure, result)
    _run("device_history.png", device_history, result)
    return written
