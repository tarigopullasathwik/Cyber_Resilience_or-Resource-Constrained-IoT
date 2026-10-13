"""
Reproducibility tests: identical seeds must give identical telemetry, identical
detector behaviour, and identical experiment metrics (the paper-style claim that
every number can be regenerated from the recorded seed).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from config import FEATURE_NAMES, SYSTEM_PROPOSED, ExperimentConfig
from src.anomaly_detector import AnomalyDetector
from src.experiment_runner import ExperimentRunner
from src.traffic_simulator import TrafficSimulator


def _small_config(seed: int) -> ExperimentConfig:
    return ExperimentConfig(seed=seed, n_devices=10, n_steps=150, calibration_steps=50,
                            warmup_steps=55)


def test_telemetry_is_identical_for_the_same_seed():
    telemetry_a, attacks_a = TrafficSimulator(_small_config(2026)).generate()
    telemetry_b, attacks_b = TrafficSimulator(_small_config(2026)).generate()
    pd.testing.assert_frame_equal(telemetry_a, telemetry_b)
    pd.testing.assert_frame_equal(attacks_a, attacks_b)


def test_detector_is_deterministic_for_the_same_seed():
    telemetry, _ = TrafficSimulator(_small_config(2026)).generate()
    config = _small_config(2026)
    from src.adaptive_learning import AdaptiveBaselineManager
    from src.feature_extractor import FeatureExtractor
    from src.traffic_simulator import TrafficSimulator as Sim

    simulator = Sim(config)
    manager = AdaptiveBaselineManager(config.learning)
    manager.fit(telemetry, config.calibration_steps)
    extractor = FeatureExtractor(
        manager,
        {d.device_id: d.allowed_protocols for d in simulator.devices},
        {d.device_id: d.normal_protocol for d in simulator.devices},
    )
    calibration = telemetry[telemetry["step"] < config.calibration_steps]
    vectors = [extractor.extract(row) for _, row in calibration.iterrows()]
    X = pd.DataFrame(np.vstack([v.values for v in vectors]), columns=list(FEATURE_NAMES))
    ids = [v.device_id for v in vectors]

    scores = []
    for _ in range(2):
        detector = AnomalyDetector(config.detector, seed=config.seed, adaptive=True)
        detector.calibrate(X, ids)
        online = telemetry[telemetry["step"] >= config.calibration_steps].head(200)
        values = []
        for step, group in online.groupby("step"):
            group_ids = [str(r["device_id"]) for _, r in group.iterrows()]
            batch = np.vstack([extractor.extract(r, step=int(step)).values
                               for _, r in group.iterrows()])
            values.extend(r.score for r in detector.process_batch(int(step), group_ids, batch))
        scores.append(values)
    assert scores[0] == scores[1]
    assert len(scores[0]) > 0


def test_full_experiment_is_reproducible_for_the_same_seed():
    first = ExperimentRunner(_small_config(2026), scenario="test").run(systems=[SYSTEM_PROPOSED])
    second = ExperimentRunner(_small_config(2026), scenario="test").run(systems=[SYSTEM_PROPOSED])
    pd.testing.assert_frame_equal(first.decision_logs[SYSTEM_PROPOSED],
                                  second.decision_logs[SYSTEM_PROPOSED])
    summary_a = first.systems[SYSTEM_PROPOSED].summary
    summary_b = second.systems[SYSTEM_PROPOSED].summary
    for key in ("accuracy", "precision", "recall", "f1", "false_positive_rate",
                "n_missed_attacks", "containment_success", "unnecessary_quarantines"):
        assert summary_a[key] == summary_b[key], key
    assert len(first.systems[SYSTEM_PROPOSED].actions) == \
        len(second.systems[SYSTEM_PROPOSED].actions)


def test_different_seeds_produce_different_experiments():
    a = ExperimentRunner(_small_config(2026), scenario="test").run(systems=[SYSTEM_PROPOSED])
    b = ExperimentRunner(_small_config(2027), scenario="test").run(systems=[SYSTEM_PROPOSED])
    assert not np.allclose(a.decision_logs[SYSTEM_PROPOSED]["anomaly_score"].to_numpy(),
                           b.decision_logs[SYSTEM_PROPOSED]["anomaly_score"].to_numpy())


def test_experiment_records_the_seed_and_configuration():
    result = ExperimentRunner(_small_config(2026), scenario="test").run(systems=[SYSTEM_PROPOSED])
    assert result.seed == 2026
    assert result.config.seed == 2026
    assert result.metadata["scenario"] == "test"
    assert "generated_utc" in result.metadata
    # The reported summary must be enough to rerun the experiment.
    assert result.config.to_dict()["n_devices"] == 10


def test_ground_truth_labels_are_never_detector_inputs():
    """
    Adding ground-truth label columns to the calibration data must change nothing:
    the detector is fitted on features only.
    """
    from src.adaptive_learning import AdaptiveBaselineManager
    from src.feature_extractor import FeatureExtractor

    config = _small_config(2026)
    simulator = TrafficSimulator(config)
    telemetry, _ = simulator.generate()
    manager = AdaptiveBaselineManager(config.learning)
    manager.fit(telemetry, config.calibration_steps)
    extractor = FeatureExtractor(
        manager,
        {d.device_id: d.allowed_protocols for d in simulator.devices},
        {d.device_id: d.normal_protocol for d in simulator.devices},
    )
    calibration = telemetry[telemetry["step"] < config.calibration_steps]
    vectors = [extractor.extract(row) for _, row in calibration.iterrows()]
    X = pd.DataFrame(np.vstack([v.values for v in vectors]), columns=list(FEATURE_NAMES))
    ids = [v.device_id for v in vectors]

    plain = AnomalyDetector(config.detector, seed=config.seed, adaptive=True)
    plain.calibrate(X, ids)

    with_labels = X.copy()
    for index, row in calibration.reset_index(drop=True).iterrows():
        with_labels.loc[index, "ground_truth"] = row["ground_truth"]
        with_labels.loc[index, "attack_type"] = row["attack_type"]
    labelled = AnomalyDetector(config.detector, seed=config.seed, adaptive=True)
    labelled.calibrate(with_labels, ids)

    assert plain.thresholds == labelled.thresholds
    probe = X.iloc[0].to_numpy()
    assert plain.process(1, ids[0], probe).score == labelled.process(1, ids[0], probe).score
    # And no label-like column may sneak into the feature vector itself.
    for name in FEATURE_NAMES:
        assert "ground" not in name and "attack" not in name and "label" not in name
