"""
Evaluation metrics (component 5 of the research plan).

Every number produced here is computed from the logs of an experiment that has
actually run: no metric is hard-coded or estimated. Ground-truth labels are used
**only** in this module, never in detection, assessment or containment decisions.

Metric definitions (all stated explicitly because the same word often means
different things in different papers):

Point-wise (per device-step, inside the post-calibration evaluation window)
    TP  predicted anomalous AND label anomalous
    FP  predicted anomalous AND label benign        -> drives the false-positive rate
    TN  predicted benign    AND label benign
    FN  predicted benign    AND label anomalous     -> missed detections
    accuracy   (TP+TN)/(TP+FP+TN+FN)
    precision  TP/(TP+FP)
    recall     TP/(TP+FN)          (also called detection rate)
    f1         2*precision*recall/(precision+recall)
    fpr        FP/(FP+TN)          (false-positive rate over benign device-steps)

Episode-wise (one row per injected attack, device and attack interval)
    detected            at least one anomalous decision inside
                        [start, end + latency_grace_steps]
    detection_latency   steps from attack start to the first anomalous decision
    response_latency    steps from that first detection to the first containment
                        action of level >= 1 on the same device
    missed_attack       an episode that was never detected

Containment
    containment_success_rate  SUCCESS/(SUCCESS+FAILURE) verification windows;
                              INCONCLUSIVE windows are reported separately and
                              excluded from the rate (they carry too little evidence)
    recovery_time       steps from the first containment action to the full restore
    unnecessary_quarantine  a QUARANTINE action applied on a device-step whose
                        ground-truth label is benign

Resources
    simulated_*   outputs of the edge resource model (``src/feature_extractor.py``):
                  clearly *not* measurements of the host machine
    measured_*    actual host measurements (wall-clock timing and a tracemalloc
                  peak) taken while the online loop ran

Zero denominators never raise: they return 0.0 (or NaN where a rate is
undefined) and the supporting counts (``n_*``) are always reported alongside so a
degenerate case is visible instead of hidden.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import SYSTEM_LABELS, SYSTEMS  # noqa: E402

#: Extra steps after an attack ends in which a first detection still counts, so a
#: one-step-late decision is not scored as a miss. Documented choice.
LATENCY_GRACE_STEPS: int = 2


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Division that never raises and never returns inf/nan for a zero denominator."""
    try:
        if denominator is None or float(denominator) == 0.0:
            return float(default)
        value = float(numerator) / float(denominator)
        if not np.isfinite(value):
            return float(default)
        return value
    except (TypeError, ValueError, ZeroDivisionError):
        return float(default)


# --------------------------------------------------------------------------- #
# Point-wise classification
# --------------------------------------------------------------------------- #
def confusion_counts(y_true: Sequence[int], y_pred: Sequence[int]) -> Dict[str, int]:
    y_true_arr = np.asarray(y_true, dtype=int)
    y_pred_arr = np.asarray(y_pred, dtype=int)
    if y_true_arr.size == 0:
        return {"tp": 0, "fp": 0, "tn": 0, "fn": 0, "n": 0}
    tp = int(np.sum((y_true_arr == 1) & (y_pred_arr == 1)))
    fp = int(np.sum((y_true_arr == 0) & (y_pred_arr == 1)))
    tn = int(np.sum((y_true_arr == 0) & (y_pred_arr == 0)))
    fn = int(np.sum((y_true_arr == 1) & (y_pred_arr == 0)))
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn, "n": int(y_true_arr.size)}


def classification_metrics(y_true: Sequence[int], y_pred: Sequence[int]) -> Dict[str, float]:
    """Accuracy, precision, recall, F1 and false-positive rate with safe division."""
    counts = confusion_counts(y_true, y_pred)
    tp, fp, tn, fn, n = counts["tp"], counts["fp"], counts["tn"], counts["fn"], counts["n"]
    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    return {
        "n_observations": n,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "accuracy": safe_divide(tp + tn, n),
        "precision": precision,
        "recall": recall,
        "f1": safe_divide(2 * precision * recall, precision + recall),
        "false_positive_rate": safe_divide(fp, fp + tn),
        "specificity": safe_divide(tn, tn + fp),
    }


# --------------------------------------------------------------------------- #
# Attack episodes and latency
# --------------------------------------------------------------------------- #
def attack_episodes(telemetry: pd.DataFrame) -> pd.DataFrame:
    """One row per injected attack (device + contiguous labelled interval)."""
    attacked = telemetry[telemetry["ground_truth"] == 1]
    if attacked.empty:
        return pd.DataFrame(columns=["device_id", "attack_id", "attack_type", "held_out",
                                     "start_step", "end_step", "n_steps"])
    rows = (
        attacked.groupby(["device_id", "attack_id", "attack_type", "attack_held_out"], sort=True)
        .agg(start_step=("step", "min"), end_step=("step", "max"), n_steps=("step", "size"))
        .reset_index()
        .rename(columns={"attack_held_out": "held_out"})
    )
    return rows[["device_id", "attack_id", "attack_type", "held_out", "start_step", "end_step", "n_steps"]]


def detection_outcomes(
    episodes: pd.DataFrame,
    decisions: pd.DataFrame,
    grace_steps: int = LATENCY_GRACE_STEPS,
) -> pd.DataFrame:
    """
    Episode-level detection outcome and detection latency.

    A detection counts when the device is flagged anomalous at any step inside
    ``[start_step, end_step + grace_steps]``.
    """
    columns = ["device_id", "attack_id", "attack_type", "held_out", "start_step", "end_step",
               "detected", "detect_step", "detection_latency_steps"]
    if episodes.empty:
        return pd.DataFrame(columns=columns)
    flagged = decisions[decisions["is_anomalous"] == 1]
    rows: List[Dict[str, Any]] = []
    for _, episode in episodes.iterrows():
        window = flagged[
            (flagged["device_id"] == episode["device_id"])
            & (flagged["step"] >= episode["start_step"])
            & (flagged["step"] <= episode["end_step"] + grace_steps)
        ]
        if window.empty:
            rows.append({**episode.to_dict(), "detected": 0, "detect_step": np.nan,
                         "detection_latency_steps": np.nan})
        else:
            first = int(window["step"].min())
            rows.append({**episode.to_dict(), "detected": 1, "detect_step": first,
                         "detection_latency_steps": int(first - int(episode["start_step"]))})
    return pd.DataFrame(rows, columns=columns)


def response_latency(
    episode_outcomes: pd.DataFrame,
    actions: pd.DataFrame,
) -> pd.DataFrame:
    """
    Steps between first detection and the first mitigation action (level >= 1).

    OBSERVE-only decisions are *not* counted as a response: the metric measures
    actuation, and any deviation from that definition is stated in the README.
    """
    columns = list(episode_outcomes.columns) + ["response_step", "response_latency_steps"]
    if episode_outcomes.empty or actions.empty:
        frame = episode_outcomes.copy()
        frame["response_step"] = np.nan
        frame["response_latency_steps"] = np.nan
        return frame
    mitigations = actions[actions["level"] >= 1]
    rows: List[Dict[str, Any]] = []
    for _, episode in episode_outcomes.iterrows():
        row = episode.to_dict()
        if not int(episode.get("detected", 0)):
            row.update({"response_step": np.nan, "response_latency_steps": np.nan})
            rows.append(row)
            continue
        candidates = mitigations[
            (mitigations["device_id"] == episode["device_id"])
            & (mitigations["step"] >= int(episode["detect_step"]))
        ]
        if candidates.empty:
            row.update({"response_step": np.nan, "response_latency_steps": np.nan})
        else:
            first = int(candidates["step"].min())
            row.update({"response_step": first,
                        "response_latency_steps": int(first - int(episode["detect_step"]))})
        rows.append(row)
    return pd.DataFrame(rows, columns=columns)


def latency_summary(frame: pd.DataFrame, column: str) -> Dict[str, float]:
    if frame.empty or column not in frame.columns:
        return {"mean": float("nan"), "median": float("nan"), "max": float("nan"), "n": 0}
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    if values.empty:
        return {"mean": float("nan"), "median": float("nan"), "max": float("nan"), "n": 0}
    return {"mean": float(values.mean()), "median": float(values.median()),
            "max": float(values.max()), "n": int(values.size)}


# --------------------------------------------------------------------------- #
# Containment metrics
# --------------------------------------------------------------------------- #
def containment_success_rate(verification: pd.DataFrame) -> Dict[str, float]:
    """SUCCESS/(SUCCESS+FAILURE) over completed verification windows."""
    if verification.empty:
        return {"success": 0, "failure": 0, "inconclusive": 0, "rate": float("nan"), "n": 0}
    counts = verification["outcome"].value_counts().to_dict()
    success = int(counts.get("SUCCESS", 0))
    failure = int(counts.get("FAILURE", 0))
    inconclusive = int(counts.get("INCONCLUSIVE", 0))
    return {
        "success": success, "failure": failure, "inconclusive": inconclusive,
        "rate": safe_divide(success, success + failure, default=float("nan")),
        "n": success + failure + inconclusive,
    }


def recovery_summary(recovery: pd.DataFrame) -> Dict[str, float]:
    if recovery.empty or "recovery_time_steps" not in recovery.columns:
        return {"n_recoveries": 0, "mean": float("nan"), "median": float("nan"), "max": float("nan")}
    values = pd.to_numeric(recovery["recovery_time_steps"], errors="coerce").dropna()
    if values.empty:
        return {"n_recoveries": 0, "mean": float("nan"), "median": float("nan"), "max": float("nan")}
    return {"n_recoveries": int(values.size), "mean": float(values.mean()),
            "median": float(values.median()), "max": float(values.max())}


def unnecessary_quarantines(actions: pd.DataFrame, telemetry: pd.DataFrame) -> Dict[str, float]:
    """
    QUARANTINE actions applied to device-steps whose ground truth is benign.

    Ground truth is used here for *evaluation only*; the policy itself never sees it.
    """
    if actions.empty:
        return {"quarantine_actions": 0, "unnecessary": 0, "rate": float("nan")}
    quarantine = actions[(actions["action"] == "QUARANTINE") & (actions["level"] >= 1)]
    if quarantine.empty:
        return {"quarantine_actions": 0, "unnecessary": 0, "rate": float("nan")}
    labels = telemetry[["step", "device_id", "ground_truth"]]
    merged = quarantine.merge(labels, on=["step", "device_id"], how="left")
    total = int(merged.shape[0])
    unnecessary = int((merged["ground_truth"] == 0).sum())
    return {"quarantine_actions": total, "unnecessary": unnecessary,
            "rate": safe_divide(unnecessary, total, default=float("nan"))}


def drift_adaptation_time(
    decisions: pd.DataFrame,
    telemetry: pd.DataFrame,
    benign_run: int = 5,
) -> Dict[str, Any]:
    """
    How long a legitimately drifted device stays flagged after its behaviour change.

    For every device that undergoes a benign behavioural shift, the clock starts at
    the first shifted step and stops at the first run of ``benign_run`` consecutive
    non-anomalous decisions. Devices that never settle again are reported as
    ``not_adapted`` (they are exactly the ones a static threshold can never accept).
    """
    empty = {"n_shifted_devices": 0, "n_shifted_devices_adapted": 0,
             "drift_adaptation_median_steps": float("nan"),
             "drift_adaptation_mean_steps": float("nan")}
    if "benign_shift" not in telemetry.columns or decisions.empty:
        return empty
    shifted = (telemetry[telemetry["benign_shift"] == True]  # noqa: E712
               .groupby("device_id")["step"].min())
    if shifted.empty:
        return empty
    times: List[int] = []
    for device_id, start_step in shifted.items():
        frame = decisions[(decisions["device_id"] == device_id)
                          & (decisions["step"] >= int(start_step))].sort_values("step")
        if frame.empty:
            continue
        flags = frame["is_anomalous"].to_numpy(dtype=int)
        steps = frame["step"].to_numpy(dtype=int)
        streak = 0
        for i, flag in enumerate(flags):
            streak = 0 if flag else streak + 1
            if streak >= benign_run:
                times.append(int(steps[i - benign_run + 1] - int(start_step)))
                break
    return {
        "n_shifted_devices": int(shifted.shape[0]),
        "n_shifted_devices_adapted": len(times),
        "drift_adaptation_median_steps": float(np.median(times)) if times else float("nan"),
        "drift_adaptation_mean_steps": float(np.mean(times)) if times else float("nan"),
    }


def missed_attacks(episode_outcomes: pd.DataFrame) -> Dict[str, Any]:
    if episode_outcomes.empty:
        return {"n_episodes": 0, "missed": 0, "missed_rate": float("nan"), "missed_types": []}
    missed = episode_outcomes[episode_outcomes["detected"] == 0]
    return {
        "n_episodes": int(episode_outcomes.shape[0]),
        "missed": int(missed.shape[0]),
        "missed_rate": safe_divide(missed.shape[0], episode_outcomes.shape[0], default=float("nan")),
        "missed_types": sorted(missed["attack_type"].unique().tolist()) if not missed.empty else [],
    }


# --------------------------------------------------------------------------- #
# Resource metrics
# --------------------------------------------------------------------------- #
def resource_summary(resources: pd.DataFrame) -> Dict[str, float]:
    """Simulated resource-model outputs (aggregated per step)."""
    if resources.empty:
        return {}
    out = {
        "simulated_cpu_mean": float(resources["cpu_used"].mean()),
        "simulated_cpu_max": float(resources["cpu_used"].max()),
        "simulated_cpu_utilisation_mean": float(resources["cpu_utilisation"].mean()),
        "simulated_memory_mean_mb": float(resources["memory_used_mb"].mean()),
        "simulated_memory_max_mb": float(resources["memory_used_mb"].max()),
        "simulated_edge_uplink_mean_kbps": float(resources["uplink_kbps"].mean()),
        "simulated_edge_latency_mean_ms": float(resources["edge_latency_ms"].mean()),
        "simulated_cloud_uplink_mean_kbps": float(resources["cloud_uplink_kbps"].mean()),
        "simulated_cloud_latency_mean_ms": float(resources["cloud_latency_ms"].mean()),
        "simulated_steps_cpu_saturated": int(resources["cpu_saturated"].sum()),
        "simulated_bandwidth_saturated_steps": int(resources["bandwidth_saturated"].sum()),
    }
    out["simulated_bandwidth_overhead_ratio"] = safe_divide(
        out["simulated_cloud_uplink_mean_kbps"], out["simulated_edge_uplink_mean_kbps"],
        default=float("nan"),
    )
    return out


def centralization_comparison(resources: pd.DataFrame) -> Dict[str, float]:
    """Edge-resident processing vs shipping raw telemetry to a central service."""
    if resources.empty:
        return {}
    edge_latency = float(resources["edge_latency_ms"].mean())
    cloud_latency = float(resources["cloud_latency_ms"].mean())
    edge_uplink = float(resources["uplink_kbps"].mean())
    cloud_uplink = float(resources["cloud_uplink_kbps"].mean())
    return {
        "edge_latency_mean_ms": edge_latency,
        "central_latency_mean_ms": cloud_latency,
        "latency_reduction_ms": cloud_latency - edge_latency,
        "latency_reduction_ratio": safe_divide(cloud_latency - edge_latency, cloud_latency,
                                               default=float("nan")),
        "edge_uplink_mean_kbps": edge_uplink,
        "central_uplink_mean_kbps": cloud_uplink,
        "uplink_reduction_ratio": safe_divide(cloud_uplink - edge_uplink, cloud_uplink,
                                              default=float("nan")),
    }


# --------------------------------------------------------------------------- #
# Bundle
# --------------------------------------------------------------------------- #
@dataclass
class SystemMetrics:
    """All metrics, tables and measured host numbers for one experiment system."""

    system: str
    label: str
    summary: Dict[str, Any]
    confusion: pd.DataFrame
    per_attack_type: pd.DataFrame
    episodes: pd.DataFrame
    verification: pd.DataFrame
    actions: pd.DataFrame
    resources: pd.DataFrame
    host_measurements: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"system": self.system, "label": self.label, "summary": self.summary,
                "host_measurements": self.host_measurements}


def summarize_system(
    system: str,
    decisions: pd.DataFrame,
    telemetry: pd.DataFrame,
    actions: pd.DataFrame,
    verification: pd.DataFrame,
    recovery: pd.DataFrame,
    resources: pd.DataFrame,
    host_measurements: Optional[Dict[str, Any]] = None,
    learning_blocked: Optional[Dict[str, int]] = None,
) -> SystemMetrics:
    """Compute the complete metric set for one system's logs."""
    # Steps where no decision could be made (a fully isolated device emits no
    # telemetry) are excluded from the point-wise metrics and reported separately,
    # because scoring them as "benign" would fabricate false negatives.
    scored_mask = decisions["scored"] == 1 if "scored" in decisions.columns \
        else pd.Series(True, index=decisions.index)
    n_not_scored = int((~scored_mask).sum())
    eval_decisions = decisions[scored_mask]
    pointwise = classification_metrics(eval_decisions["ground_truth"], eval_decisions["is_anomalous"])
    episodes = attack_episodes(telemetry)
    outcomes = detection_outcomes(episodes, eval_decisions)
    outcomes = response_latency(outcomes, actions)

    # Held-out (unknown-attack proxy) vs known-attack comparison, point-wise.
    held_mask_decisions = eval_decisions
    held_out_pointwise = classification_metrics(
        held_mask_decisions[held_mask_decisions["attack_held_out"] == True]["ground_truth"],  # noqa: E712
        held_mask_decisions[held_mask_decisions["attack_held_out"] == True]["is_anomalous"],  # noqa: E712
    )
    known_attacks = held_mask_decisions[
        (held_mask_decisions["ground_truth"] == 1) & (held_mask_decisions["attack_held_out"] == False)  # noqa: E712
    ]
    held_attacks = held_mask_decisions[held_mask_decisions["attack_held_out"] == True]  # noqa: E712

    # Benign device-steps split into stable devices and legitimately shifted ones.
    benign = eval_decisions[eval_decisions["ground_truth"] == 0]
    benign_shift = benign[benign["benign_shift"] == True]      # noqa: E712
    benign_stable = benign[benign["benign_shift"] == False]    # noqa: E712

    per_attack_rows: List[Dict[str, Any]] = []
    for attack_type, group in outcomes.groupby("attack_type", sort=True):
        per_attack_rows.append({
            "attack_type": attack_type,
            "held_out": bool(group["held_out"].iloc[0]),
            "episodes": int(group.shape[0]),
            "detected": int(group["detected"].sum()),
            "detection_rate": safe_divide(group["detected"].sum(), group.shape[0]),
            "detection_latency_mean_steps": float(
                pd.to_numeric(group["detection_latency_steps"], errors="coerce").mean()),
            "response_latency_mean_steps": float(
                pd.to_numeric(group["response_latency_steps"], errors="coerce").mean()),
        })
    per_attack_type = pd.DataFrame(per_attack_rows)

    drift = drift_adaptation_time(eval_decisions, telemetry)
    containment = containment_success_rate(verification)
    recovery_stats = recovery_summary(recovery)
    quarantine = unnecessary_quarantines(actions, telemetry)
    missed = missed_attacks(outcomes)
    resource_stats = resource_summary(resources)
    central = centralization_comparison(resources)
    detection_lat = latency_summary(outcomes, "detection_latency_steps")
    response_lat = latency_summary(outcomes, "response_latency_steps")

    summary: Dict[str, Any] = {
        "system": system, "label": SYSTEM_LABELS.get(system, system),
        **pointwise,
        "held_out_pointwise_recall": safe_divide(
            held_attacks[held_attacks["ground_truth"] == 1]["is_anomalous"].sum(),
            (held_attacks["ground_truth"] == 1).sum(), default=float("nan")),
        "known_attack_pointwise_recall": safe_divide(
            known_attacks["is_anomalous"].sum(), known_attacks.shape[0], default=float("nan")),
        "benign_stable_fpr": safe_divide(benign_stable["is_anomalous"].sum(), benign_stable.shape[0],
                                        default=float("nan")),
        "benign_shift_fpr": safe_divide(benign_shift["is_anomalous"].sum(), benign_shift.shape[0],
                                       default=float("nan")),
        "n_steps_not_scored": n_not_scored,
        "n_scored_observations": int(eval_decisions.shape[0]),
        **drift,
        "n_episodes": missed["n_episodes"],
        "n_missed_attacks": missed["missed"],
        "missed_attack_rate": missed["missed_rate"],
        "missed_attack_types": ",".join(missed["missed_types"]),
        "detection_latency_mean_steps": detection_lat["mean"],
        "detection_latency_median_steps": detection_lat["median"],
        "detection_latency_max_steps": detection_lat["max"],
        "detection_latency_mean_seconds": detection_lat["mean"] * float(
            decisions.attrs.get("step_seconds", 5.0) or 5.0),
        "response_latency_mean_steps": response_lat["mean"],
        "response_latency_median_steps": response_lat["median"],
        "response_latency_mean_seconds": response_lat["mean"] * float(
            decisions.attrs.get("step_seconds", 5.0) or 5.0),
        "containment_success": containment["success"],
        "containment_failure": containment["failure"],
        "containment_inconclusive": containment["inconclusive"],
        "containment_success_rate": containment["rate"],
        "recovery_n": recovery_stats["n_recoveries"],
        "recovery_time_mean_steps": recovery_stats["mean"],
        "recovery_time_median_steps": recovery_stats["median"],
        "recovery_time_max_steps": recovery_stats["max"],
        "quarantine_actions": quarantine["quarantine_actions"],
        "unnecessary_quarantines": quarantine["unnecessary"],
        "unnecessary_quarantine_rate": quarantine["rate"],
        "n_containment_actions": int((actions["level"] >= 1).sum()) if not actions.empty else 0,
        **{k: v for k, v in resource_stats.items()},
        **{f"centralization_{k}": v for k, v in central.items()},
        **(learning_blocked or {}),
    }
    confusion = pd.DataFrame([
        {"outcome": "TP (attack detected)", "count": pointwise["tp"]},
        {"outcome": "FP (false alarm)", "count": pointwise["fp"]},
        {"outcome": "TN (benign)", "count": pointwise["tn"]},
        {"outcome": "FN (missed attack)", "count": pointwise["fn"]},
    ])
    return SystemMetrics(
        system=system, label=SYSTEM_LABELS.get(system, system), summary=summary,
        confusion=confusion, per_attack_type=per_attack_type, episodes=outcomes,
        verification=verification, actions=actions, resources=resources,
        host_measurements=host_measurements or {},
    )


def comparison_table(metrics_by_system: Dict[str, SystemMetrics]) -> pd.DataFrame:
    """Side-by-side metric table for all systems (the main results table)."""
    keys = [
        "accuracy", "precision", "recall", "f1", "false_positive_rate",
        "benign_stable_fpr", "benign_shift_fpr", "n_steps_not_scored",
        "drift_adaptation_median_steps", "n_shifted_devices_adapted", "n_shifted_devices",
        "known_attack_pointwise_recall", "held_out_pointwise_recall",
        "detection_latency_mean_steps", "response_latency_mean_steps",
        "n_episodes", "n_missed_attacks", "missed_attack_rate",
        "containment_success_rate", "containment_success", "containment_failure",
        "containment_inconclusive", "n_containment_actions", "recovery_n",
        "recovery_time_mean_steps", "quarantine_actions", "unnecessary_quarantines",
        "simulated_cpu_utilisation_mean", "simulated_memory_mean_mb",
        "simulated_steps_cpu_saturated", "simulated_bandwidth_saturated_steps",
        "simulated_edge_uplink_mean_kbps", "simulated_bandwidth_overhead_ratio",
        "measured_wall_clock_seconds", "measured_per_step_ms", "measured_peak_alloc_mb",
    ]
    rows: List[Dict[str, Any]] = []
    for key in keys:
        row: Dict[str, Any] = {"metric": key}
        for system in SYSTEMS:
            metrics = metrics_by_system.get(system)
            if metrics is None:
                row[SYSTEM_LABELS.get(system, system)] = None
                continue
            value = metrics.summary.get(key)
            if value is None:
                value = metrics.host_measurements.get(key) if metrics.host_measurements else None
            row[SYSTEM_LABELS.get(system, system)] = value
        rows.append(row)
    return pd.DataFrame(rows)
