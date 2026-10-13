"""
Tests for the metric calculations: exact values on hand-built inputs, safe
handling of zero denominators, latency accounting, containment outcomes,
unnecessary quarantines, resource summaries and the comparison table.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from config import SYSTEM_ADAPTIVE, SYSTEM_PROPOSED, SYSTEM_STATIC
from src.metrics import (
    attack_episodes,
    centralization_comparison,
    classification_metrics,
    comparison_table,
    confusion_counts,
    containment_success_rate,
    detection_outcomes,
    drift_adaptation_time,
    latency_summary,
    missed_attacks,
    recovery_summary,
    resource_summary,
    response_latency,
    safe_divide,
    summarize_system,
    unnecessary_quarantines,
)


def test_safe_divide_never_raises_or_returns_inf():
    assert safe_divide(1, 0, default=0.0) == 0.0
    assert safe_divide(0, 0, default=float("nan")) != safe_divide(0, 0, default=float("nan")) \
        or np.isnan(safe_divide(0, 0, default=float("nan")))
    assert safe_divide(3, 4) == pytest.approx(0.75)
    assert safe_divide(None, 4, default=-1.0) == -1.0


def test_confusion_counts_are_exact():
    y_true = [1, 1, 0, 0, 1, 0]
    y_pred = [1, 0, 1, 0, 1, 0]
    counts = confusion_counts(y_true, y_pred)
    assert counts == {"tp": 2, "fp": 1, "tn": 2, "fn": 1, "n": 6}


def test_classification_metrics_match_hand_computed_values():
    y_true = [1, 1, 0, 0, 1, 0]
    y_pred = [1, 0, 1, 0, 1, 0]
    metrics = classification_metrics(y_true, y_pred)
    assert metrics["accuracy"] == pytest.approx(4 / 6)
    assert metrics["precision"] == pytest.approx(2 / 3)
    assert metrics["recall"] == pytest.approx(2 / 3)
    assert metrics["f1"] == pytest.approx(2 / 3)
    assert metrics["false_positive_rate"] == pytest.approx(1 / 3)


def test_metrics_with_a_degenerate_prediction_are_zero_not_nan():
    all_benign = classification_metrics([0, 0, 0], [0, 0, 0])
    assert all_benign["precision"] == 0.0
    assert all_benign["recall"] == 0.0
    assert all_benign["f1"] == 0.0
    assert all_benign["false_positive_rate"] == 0.0
    assert all_benign["accuracy"] == 1.0

    all_positive = classification_metrics([1, 1, 1], [1, 1, 1])
    assert all_positive["precision"] == 1.0
    assert all_positive["false_positive_rate"] == 0.0

    empty = classification_metrics([], [])
    assert empty["n_observations"] == 0
    assert empty["accuracy"] == 0.0


def _decisions_frame(rows):
    """rows: iterable of (step, device_id, is_anomalous); everything else is defaulted."""
    rows = [tuple(row) + (1,) * (4 - len(row)) for row in rows]
    frame = pd.DataFrame(rows, columns=["step", "device_id", "is_anomalous", "scored"])
    frame["anomaly_score"] = 0.9
    frame["threshold"] = 0.5
    frame["severity"] = 50.0
    frame["ground_truth"] = 0
    frame["attack_held_out"] = False
    frame["benign_shift"] = False
    frame["attack_type"] = ""
    frame["criticality_factor"] = 0.5
    return frame


def test_detection_latency_and_missed_attacks():
    telemetry = pd.DataFrame([
        {"step": s, "device_id": "d1", "ground_truth": 1 if 10 <= s < 15 else 0,
         "attack_type": "flood" if 10 <= s < 15 else "", "attack_id": "flood-00" if 10 <= s < 15 else "",
         "attack_held_out": False, "benign_shift": False}
        for s in range(20)
    ])
    decisions = _decisions_frame([
        (10, "d1", 0), (11, "d1", 1), (12, "d1", 1), (13, "d1", 0), (14, "d1", 0),
        (15, "d1", 0), (16, "d1", 0),
    ])
    episodes = attack_episodes(telemetry)
    assert episodes.shape[0] == 1
    outcomes = detection_outcomes(episodes, decisions)
    assert outcomes["detected"].iloc[0] == 1
    assert outcomes["detection_latency_steps"].iloc[0] == 1
    assert latency_summary(outcomes, "detection_latency_steps")["mean"] == 1.0

    # A never-detected episode is reported as a miss, not as a zero latency.
    miss_decisions = _decisions_frame([(s, "d1", 0) for s in range(20)])
    misses = detection_outcomes(episodes, miss_decisions)
    assert misses["detected"].iloc[0] == 0
    assert np.isnan(misses["detection_latency_steps"].iloc[0])
    assert missed_attacks(misses)["missed"] == 1
    assert missed_attacks(misses)["missed_types"] == ["flood"]


def test_detection_counts_inside_the_grace_window_but_not_beyond():
    telemetry = pd.DataFrame([
        {"step": s, "device_id": "d1", "ground_truth": 1 if s == 10 else 0,
         "attack_type": "flood" if s == 10 else "", "attack_id": "flood-00" if s == 10 else "",
         "attack_held_out": False, "benign_shift": False}
        for s in range(20)
    ])
    episodes = attack_episodes(telemetry)
    late_ok = _decisions_frame([(12, "d1", 1)])
    assert detection_outcomes(episodes, late_ok, grace_steps=2)["detected"].iloc[0] == 1
    too_late = _decisions_frame([(14, "d1", 1)])
    assert detection_outcomes(episodes, too_late, grace_steps=2)["detected"].iloc[0] == 0


def test_response_latency_ignores_observe_only_events():
    outcomes = pd.DataFrame([{
        "device_id": "d1", "attack_id": "a", "attack_type": "flood", "held_out": False,
        "start_step": 10, "end_step": 12, "n_steps": 3, "detected": 1, "detect_step": 10,
        "detection_latency_steps": 0,
    }])
    actions = pd.DataFrame([
        {"step": 10, "device_id": "d1", "action": "OBSERVE", "level": 0, "kind": "OBSERVE"},
        {"step": 13, "device_id": "d1", "action": "RATE_LIMIT", "level": 1, "kind": "APPLY"},
    ])
    frame = response_latency(outcomes, actions)
    assert frame["response_step"].iloc[0] == 13
    assert frame["response_latency_steps"].iloc[0] == 3


def test_containment_success_rate_excludes_inconclusive():
    verification = pd.DataFrame({"outcome": ["SUCCESS", "SUCCESS", "FAILURE", "INCONCLUSIVE"]})
    stats = containment_success_rate(verification)
    assert stats["success"] == 2 and stats["failure"] == 1 and stats["inconclusive"] == 1
    assert stats["rate"] == pytest.approx(2 / 3)
    assert containment_success_rate(pd.DataFrame({"outcome": []}))["rate"] != \
        containment_success_rate(pd.DataFrame({"outcome": []}))["rate"]  # NaN


def test_unnecessary_quarantines_use_ground_truth_only_for_evaluation():
    actions = pd.DataFrame([
        {"step": 5, "device_id": "d1", "action": "QUARANTINE", "level": 3},
        {"step": 6, "device_id": "d2", "action": "QUARANTINE", "level": 3},
        {"step": 7, "device_id": "d2", "action": "RATE_LIMIT", "level": 1},
    ])
    telemetry = pd.DataFrame([
        {"step": 5, "device_id": "d1", "ground_truth": 1},
        {"step": 6, "device_id": "d2", "ground_truth": 0},
        {"step": 7, "device_id": "d2", "ground_truth": 0},
    ])
    stats = unnecessary_quarantines(actions, telemetry)
    assert stats["quarantine_actions"] == 2
    assert stats["unnecessary"] == 1
    assert stats["rate"] == pytest.approx(0.5)


def test_recovery_summary_statistics():
    recovery = pd.DataFrame({"recovery_time_steps": [4, 8, 12]})
    stats = recovery_summary(recovery)
    assert stats["n_recoveries"] == 3
    assert stats["mean"] == pytest.approx(8.0)
    assert stats["median"] == pytest.approx(8.0)
    assert stats["max"] == 12
    assert recovery_summary(pd.DataFrame())["n_recoveries"] == 0


def test_drift_adaptation_time_measures_return_to_normal():
    telemetry = pd.DataFrame([
        {"step": s, "device_id": "d1", "ground_truth": 0, "attack_type": "", "attack_id": "",
         "attack_held_out": False, "benign_shift": s >= 10}
        for s in range(30)
    ])
    decisions = _decisions_frame([(s, "d1", 1 if 10 <= s < 16 else 0) for s in range(30)])
    stats = drift_adaptation_time(decisions, telemetry, benign_run=3)
    assert stats["n_shifted_devices"] == 1
    assert stats["n_shifted_devices_adapted"] == 1
    assert stats["drift_adaptation_median_steps"] == pytest.approx(6.0)


def test_resource_and_centralization_summaries():
    resources = pd.DataFrame({
        "step": [0, 1], "cpu_used": [0.3, 0.5], "cpu_utilisation": [0.3, 0.5],
        "memory_used_mb": [76.8, 76.8], "memory_utilisation": [0.3, 0.3],
        "uplink_kbps": [0.9, 0.9], "edge_latency_ms": [20.0, 20.0],
        "cloud_uplink_kbps": [16.0, 16.0], "cloud_latency_ms": [187.0, 187.0],
        "cpu_saturated": [False, False], "bandwidth_saturated": [False, False],
    })
    stats = resource_summary(resources)
    assert stats["simulated_cpu_utilisation_mean"] == pytest.approx(0.4)
    assert stats["simulated_bandwidth_overhead_ratio"] == pytest.approx(16.0 / 0.9)
    central = centralization_comparison(resources)
    assert central["latency_reduction_ratio"] == pytest.approx((187 - 20) / 187)
    assert central["edge_uplink_mean_kbps"] < central["central_uplink_mean_kbps"]


def test_summarize_system_and_comparison_table_build():
    telemetry = pd.DataFrame([
        {"step": s, "device_id": "d1", "ground_truth": 1 if 10 <= s < 12 else 0,
         "attack_type": "flood" if 10 <= s < 12 else "", "attack_id": "flood-00" if 10 <= s < 12 else "",
         "attack_held_out": False, "benign_shift": False}
        for s in range(20)
    ])
    decisions = _decisions_frame([(s, "d1", 1 if 10 <= s < 12 else 0) for s in range(20)])
    decisions["ground_truth"] = [1 if 10 <= s < 12 else 0 for s in range(20)]
    decisions.loc[19, "scored"] = 0          # one fully isolated observation
    decisions.loc[19, "is_anomalous"] = 0
    actions = pd.DataFrame(columns=["step", "device_id", "kind", "action", "level",
                                    "previous_level", "severity", "score", "threshold", "reason"])
    verification = pd.DataFrame(columns=["request_id", "device_id", "action", "level_index",
                                         "outcome", "start_step", "end_step", "steps_observed",
                                         "score_before", "score_after", "score_reduction",
                                         "indicator_reduction", "reason"])
    recovery = pd.DataFrame(columns=["step", "device_id", "previous_action", "recovery_time_steps"])
    resources = pd.DataFrame({
        "step": [0], "cpu_used": [0.3], "cpu_utilisation": [0.3], "memory_used_mb": [76.8],
        "memory_utilisation": [0.3], "uplink_kbps": [0.9], "edge_latency_ms": [20.0],
        "cloud_uplink_kbps": [16.0], "cloud_latency_ms": [187.0], "cpu_saturated": [False],
        "bandwidth_saturated": [False],
    })
    metrics = summarize_system(
        system=SYSTEM_STATIC, decisions=decisions, telemetry=telemetry, actions=actions,
        verification=verification, recovery=recovery, resources=resources,
        host_measurements={"measured_per_step_ms": 12.0},
    )
    assert metrics.summary["recall"] == 1.0
    assert metrics.summary["n_steps_not_scored"] == 1
    assert metrics.confusion.set_index("outcome")["count"].to_dict()["TP (attack detected)"] == 2

    table = comparison_table({SYSTEM_STATIC: metrics})
    assert set(table.columns) == {"metric", SYSTEM_STATIC} or table.shape[0] > 5
    assert "accuracy" in table["metric"].tolist()
