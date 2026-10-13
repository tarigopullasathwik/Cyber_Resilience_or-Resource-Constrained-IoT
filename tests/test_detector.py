"""
Tests for the anomaly detector: benign-only calibration, detection of injected
anomalous scenarios (including held-out patterns), static vs adaptive threshold
behaviour, freeze logic, bounded scores and degenerate-input robustness.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from config import FEATURE_NAMES
from src.anomaly_detector import AnomalyDetector
from src.metrics import classification_metrics, detection_outcomes, attack_episodes


def _run_online(calibrated, adaptive: bool = True):
    """Score every post-calibration step and return the decision frame."""
    config = calibrated["config"]
    telemetry = calibrated["telemetry"]
    manager, extractor = calibrated["manager"], calibrated["extractor"]
    detector = AnomalyDetector(config.detector, seed=config.seed, adaptive=adaptive)
    calibration = telemetry[telemetry["step"] < config.calibration_steps]
    vectors = [extractor.extract(row) for _, row in calibration.iterrows()]
    detector.calibrate(
        pd.DataFrame(np.vstack([v.values for v in vectors]), columns=list(FEATURE_NAMES)),
        [v.device_id for v in vectors],
    )
    online = telemetry[telemetry["step"] >= config.calibration_steps]
    rows = []
    for step, group in online.groupby("step"):
        ids = [str(r["device_id"]) for _, r in group.iterrows()]
        X = np.vstack([extractor.extract(r, step=int(step)).values for _, r in group.iterrows()])
        for result in detector.process_batch(int(step), ids, X):
            rows.append({**result.to_dict(), "device_id": result.device_id,
                         "ground_truth": int(
                             group[group["device_id"] == result.device_id]["ground_truth"].iloc[0])})
    return detector, pd.DataFrame(rows)


def test_calibration_requires_feature_columns_only(calibrated):
    detector = AnomalyDetector(calibrated["config"].detector, seed=1, adaptive=True)
    frame = pd.DataFrame(calibrated["X_calibration"], columns=list(FEATURE_NAMES))
    # Labels must not be needed (and are not accepted as inputs by design).
    frame_with_labels = frame.copy()
    frame_with_labels["ground_truth"] = 0
    threshold = detector.calibrate(frame_with_labels, calibrated["calibration_ids"])
    assert 0.0 < threshold < 1.0
    assert set(detector.thresholds) == set(calibrated["calibration_ids"])

    broken = frame.drop(columns=["pkt_rate_ratio"])
    with pytest.raises(ValueError):
        AnomalyDetector(calibrated["config"].detector, seed=1).calibrate(broken, calibrated["calibration_ids"])


def test_detector_detects_injected_attacks_and_keeps_false_positives_low(calibrated):
    detector, decisions = _run_online(calibrated, adaptive=True)
    pointwise = classification_metrics(decisions["ground_truth"], decisions["is_anomalous"])
    assert pointwise["recall"] >= 0.80, pointwise
    assert pointwise["false_positive_rate"] <= 0.15, pointwise

    episodes = attack_episodes(calibrated["telemetry"])
    outcomes = detection_outcomes(episodes, decisions)
    assert outcomes["detected"].mean() >= 0.8, outcomes[outcomes["detected"] == 0]


def test_held_out_attack_patterns_are_also_detected(calibrated):
    """The withheld (unknown-attack proxy) patterns must not be invisible."""
    _, decisions = _run_online(calibrated, adaptive=True)
    held_out = decisions[decisions["ground_truth"] == 1]
    episodes = attack_episodes(calibrated["telemetry"])
    held_episodes = episodes[episodes["held_out"] == True]  # noqa: E712
    if held_episodes.empty:
        pytest.skip("this seed produced no held-out attack episodes")
    outcomes = detection_outcomes(held_episodes, decisions)
    assert outcomes["detected"].mean() >= 0.5, outcomes
    assert not held_out.empty


def test_static_threshold_never_adapts(calibrated):
    config = calibrated["config"]
    detector = AnomalyDetector(config.detector, seed=config.seed, adaptive=False)
    telemetry = calibrated["telemetry"]
    extractor = calibrated["extractor"]
    calibration = telemetry[telemetry["step"] < config.calibration_steps]
    vectors = [extractor.extract(row) for _, row in calibration.iterrows()]
    detector.calibrate(
        pd.DataFrame(np.vstack([v.values for v in vectors]), columns=list(FEATURE_NAMES)),
        [v.device_id for v in vectors],
    )
    frozen_thresholds = dict(detector.thresholds)
    for step, group in telemetry[telemetry["step"] >= config.calibration_steps].groupby("step"):
        ids = [str(r["device_id"]) for _, r in group.iterrows()]
        X = np.vstack([extractor.extract(r, step=int(step)).values for _, r in group.iterrows()])
        detector.process_batch(int(step), ids, X)
    assert detector.thresholds == frozen_thresholds
    assert len(detector.threshold_history) == 1
    assert detector.frozen_devices == set(), "a static detector has no freeze state"


def _fresh_calibrated_detector(calibrated, adaptive: bool = True):
    config = calibrated["config"]
    detector = AnomalyDetector(config.detector, seed=config.seed, adaptive=adaptive)
    detector.calibrate(
        pd.DataFrame(calibrated["X_calibration"], columns=list(FEATURE_NAMES)),
        calibrated["calibration_ids"],
    )
    return detector


def test_adaptive_threshold_tracks_the_recent_benign_level(calibrated):
    """
    The adaptive threshold must follow the device's recent benign score level
    (its 0.99 quantile) instead of staying at the calibration value, and it must
    respect the configured floor/ceiling.
    """
    detector = _fresh_calibrated_detector(calibrated, adaptive=True)
    device = calibrated["calibration_ids"][0]
    before = detector.thresholds[device]
    window = detector._benign_windows[device]

    window.clear()
    for _ in range(int(calibrated["config"].detector.threshold_min_samples) + 5):
        window.append(0.9)  # a device whose benign level moved upwards
    detector._adapt_threshold(device, step=1)
    assert detector.thresholds[device] > before

    window.clear()
    for _ in range(int(calibrated["config"].detector.threshold_min_samples) + 5):
        window.append(5.0)  # absurd level: must be clipped, never exceed the ceiling
    for step in range(2, 120):
        detector._adapt_threshold(device, step=step)
    assert detector.thresholds[device] <= calibrated["config"].detector.threshold_max
    assert detector.thresholds[device] >= calibrated["config"].detector.threshold_min
    assert len(detector.threshold_history) > 2


def test_adaptive_threshold_ignores_suspicious_devices(calibrated):
    """A frozen (suspicious) device may not influence the adaptive threshold."""
    detector = _fresh_calibrated_detector(calibrated, adaptive=True)
    device = calibrated["calibration_ids"][0]
    index = calibrated["calibration_ids"].index(device)
    attack = calibrated["X_calibration"][index].copy()
    attack[0] *= 9.0
    attack[1] *= 9.0
    before = dict(detector.thresholds)
    window = detector._benign_windows[device]
    window.clear()
    for step in range(1, 60):
        detector.process(step, device, attack)
    assert device in detector.frozen_devices
    assert len(window) == 0, "a frozen device must not feed the benign window"
    assert detector.thresholds == before


def test_freeze_on_anomaly_and_release_after_patience(calibrated):
    config = calibrated["config"]
    detector = AnomalyDetector(config.detector, seed=config.seed, adaptive=True)
    manager, extractor = calibrated["manager"], calibrated["extractor"]
    telemetry = calibrated["telemetry"]
    calibration = telemetry[telemetry["step"] < config.calibration_steps]
    vectors = [extractor.extract(row) for _, row in calibration.iterrows()]
    detector.calibrate(
        pd.DataFrame(np.vstack([v.values for v in vectors]), columns=list(FEATURE_NAMES)),
        [v.device_id for v in vectors],
    )
    device = calibrated["calibration_ids"][0]
    normal = calibrated["X_calibration"][calibrated["calibration_ids"].index(device)].copy()
    attack = normal.copy()
    attack[0] *= 9.0      # flooding
    attack[1] *= 9.0
    attack[2] = 12.0      # brute-force style auth failures
    result = detector.process(0, device, attack)
    assert result.is_anomalous is True
    assert device in detector.frozen_devices

    # A frozen device cannot unfreeze on a single benign step ...
    detector.process(1, device, normal)
    assert device in detector.frozen_devices
    # ... but does after unfreeze_patience consecutive benign steps.
    for step in range(2, 2 + int(config.detector.unfreeze_patience)):
        detector.process(step, device, normal)
    assert device not in detector.frozen_devices


def test_scores_and_contributions_are_bounded(calibrated):
    _, decisions = _run_online(calibrated, adaptive=True)
    assert decisions["anomaly_score"].between(0.0, 1.0).all()
    assert decisions["stat_score"].between(0.0, 1.0).all()
    assert decisions["if_score"].between(0.0, 1.0).all()
    assert decisions["threshold"].between(0.0, 1.0).all()
    assert decisions["anomaly_score"].notna().all()


def test_degenerate_feature_column_does_not_break_scoring(calibrated):
    """A constant calibration column must give a finite, bounded score."""
    config = calibrated["config"]
    detector = AnomalyDetector(config.detector, seed=config.seed, adaptive=True)
    X = calibrated["X_calibration"].copy()
    X[:, 3] = 0.0  # zero-variance feature column
    detector.calibrate(pd.DataFrame(X, columns=list(FEATURE_NAMES)), calibrated["calibration_ids"])
    probe = X[0].copy()
    probe[3] = 5.0
    result = detector.process(1, calibrated["calibration_ids"][0], probe)
    assert np.isfinite(result.score)
    assert 0.0 <= result.score <= 1.0
    assert result.contributions["conn_count_z"] > 0.5


def test_unseen_device_falls_back_to_fleet_statistics(calibrated):
    """
    A device with no calibration history must be handled with the fleet-level
    fallback statistics (never a crash, never a KeyError), and it uses the global
    threshold instead of a per-device one.
    """
    detector = _fresh_calibrated_detector(calibrated, adaptive=True)
    result = detector.process(999, "dev_unknown", np.ones(len(FEATURE_NAMES)))
    assert np.isfinite(result.score)
    assert 0.0 <= result.score <= 1.0
    assert result.threshold == pytest.approx(detector.threshold)
    assert detector.threshold == pytest.approx(float(np.median(list(detector.thresholds.values()))))


def test_top_contributions_identify_the_driving_feature(calibrated):
    detector = calibrated["detector"]
    device = calibrated["calibration_ids"][0]
    index = calibrated["calibration_ids"].index(device)
    flooding = calibrated["X_calibration"][index].copy()
    flooding[0] *= 10.0
    result = detector.process(0, device, flooding)
    top = detector.top_contributions(result, k=1)
    assert top[0][0] == "pkt_rate_ratio"
