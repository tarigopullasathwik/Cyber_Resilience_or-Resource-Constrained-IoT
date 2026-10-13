"""
Tests for the adaptive baseline manager and its anti-poisoning safeguards:
drift gates, suspicion freeze, containment block, bounded updates, and learning
of a genuine legitimate behaviour change.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from config import FEATURE_NAMES, ExperimentConfig
from src.adaptive_learning import AdaptiveBaselineManager
from src.feature_extractor import FeatureExtractor


@pytest.fixture(scope="module")
def learning_world():
    config = ExperimentConfig(seed=777, n_devices=12, n_steps=220, calibration_steps=60,
                              warmup_steps=65)
    from src.traffic_simulator import TrafficSimulator

    simulator = TrafficSimulator(config)
    telemetry, attacks = simulator.generate()
    manager = AdaptiveBaselineManager(config.learning)
    manager.fit(telemetry, config.calibration_steps)
    extractor = FeatureExtractor(
        manager,
        {d.device_id: d.allowed_protocols for d in simulator.devices},
        {d.device_id: d.normal_protocol for d in simulator.devices},
    )
    return config, simulator, telemetry, attacks, manager, extractor


def _features(extractor, row) -> dict:
    return extractor.extract(row).as_dict()


def test_baseline_fit_matches_device_normal_behaviour(learning_world):
    config, simulator, telemetry, _, manager, _ = learning_world
    snapshot = manager.snapshot()
    assert snapshot.shape[0] == len(simulator.devices)
    for device in simulator.devices[:5]:
        row = snapshot[snapshot["device_id"] == device.device_id].iloc[0]
        assert abs(row["loc_packet_rate"] - device.base_pkt_rate) < 0.35 * device.base_pkt_rate
        assert row["scale_packet_rate"] > 0.0
        assert row["frozen"] == 0


def test_drift_gates_reject_every_attack_observation(learning_world):
    """
    No injected attack may satisfy the drift gates: otherwise an attack could be
    silently learned into the baseline (baseline poisoning).
    """
    _, _, telemetry, attacks, _, extractor = learning_world
    leaked = []
    checked = 0
    for _, attack in attacks.iterrows():
        rows = telemetry[(telemetry["device_id"] == attack["device_id"])
                         & (telemetry["step"] >= attack["start_step"])
                         & (telemetry["step"] <= attack["end_step"])]
        for _, row in rows.iterrows():
            checked += 1
            if extractor.baselines is not None and \
                    extractor.baselines.drift_evidence(_features(extractor, row)):
                leaked.append((attack["attack_type"], row["step"], row["device_id"]))
    assert checked > 0
    assert not leaked, f"drift gates accepted attack observations: {leaked[:5]}"


def test_drift_gates_accept_most_legitimate_shift_observations(learning_world):
    _, simulator, telemetry, _, _, extractor = learning_world
    shifted = telemetry[(telemetry["benign_shift"] == True)  # noqa: E712
                        & (telemetry["ground_truth"] == 0)]
    if shifted.empty:
        pytest.skip("no benign shift in this scenario")
    accepted = sum(
        1 for _, row in shifted.iterrows()
        if extractor.baselines.drift_evidence(_features(extractor, row))
    )
    assert accepted / shifted.shape[0] >= 0.5


def test_anomalous_observation_never_moves_the_baseline(learning_world):
    config, simulator, telemetry, attacks, manager, extractor = learning_world
    attack = attacks.iloc[0]
    device = str(attack["device_id"])
    row = telemetry[(telemetry["device_id"] == device)
                    & (telemetry["step"] == attack["start_step"])].iloc[0]
    before = dict(manager.baseline_of(device).loc)
    event = manager.step(
        step=int(row["step"]), device_id=device,
        observation={k: float(row[k]) for k in before},
        score=0.95, threshold=0.5, is_anomalous=True,
        features=_features(extractor, row),
    )
    assert event is not None
    assert event.mode in {"blocked_anomalous", "blocked_frozen"}
    assert manager.baseline_of(device).frozen is True
    assert manager.baseline_of(device).loc == before
    assert manager.blocked_updates_anomalous >= 1


def test_observed_attack_signature_is_recorded_for_audit(learning_world):
    _, _, telemetry, attacks, manager, extractor = learning_world
    attack = attacks[attacks["attack_type"] == "brute_force"]
    if attack.empty:
        pytest.skip("no brute-force attack in this scenario")
    attack = attack.iloc[0]
    row = telemetry[(telemetry["device_id"] == attack["device_id"])
                    & (telemetry["step"] == attack["start_step"])].iloc[0]
    event = manager.step(step=1, device_id=str(attack["device_id"]),
                         observation={k: float(row[k]) for k in
                                      ("packet_rate", "bytes_per_sec", "conn_count",
                                       "dest_diversity", "cpu_usage", "memory_usage")},
                         score=0.8, threshold=0.5, is_anomalous=True,
                         features=_features(extractor, row))
    assert "failed_auth" in event.context


def test_frozen_baseline_refuses_updates_until_patience(learning_world):
    config, _, _, _, manager, _ = learning_world
    device = "dev_00"
    manager.freeze(device, 0)
    assert manager.is_frozen(device)
    before = dict(manager.baseline_of(device).loc)
    observation = dict(before)
    for step in range(1, int(config.learning.freeze_patience)):
        event = manager.step(step=step, device_id=device, observation=observation,
                             score=0.0, threshold=0.5, is_anomalous=False)
        assert event.mode == "blocked_frozen"
    assert manager.baseline_of(device).loc == before
    released = manager.note_benign(device)
    assert released is True
    assert manager.is_frozen(device) is False


def test_learning_blocked_when_heavy_containment_distorts_telemetry(learning_world):
    config, _, _, _, manager, _ = learning_world
    device = "dev_01"
    loc_before = dict(manager.baseline_of(device).loc)
    event = manager.step(step=1, device_id=device, observation=loc_before, score=0.0,
                         threshold=0.5, is_anomalous=False, traffic_allowance=0.05)
    assert event.mode == "blocked_containment"
    assert manager.baseline_of(device).loc == loc_before
    # A mild throttle still allows the (gate-protected) drift pathway.
    event2 = manager.step(step=2, device_id=device, observation=loc_before, score=0.0,
                          threshold=0.5, is_anomalous=False, traffic_allowance=0.35)
    assert event2.mode in {"clean", "drift"}


def test_single_update_is_bounded(learning_world):
    config, _, _, _, manager, _ = learning_world
    device = "dev_02"
    baseline = manager.baseline_of(device)
    start = baseline.loc["packet_rate"]
    huge = {key: baseline.loc[key] * 10.0 for key in baseline.loc}
    manager.step(step=1, device_id=device, observation=huge, score=0.0, threshold=1.0,
                 is_anomalous=False, features={"pkt_rate_ratio": 10.0, "bytes_rate_ratio": 10.0,
                                               "failed_auth_rate": 0.0, "protocol_shift": 0.0,
                                               "conn_count_z": 0.0, "dest_diversity_z": 0.0,
                                               "cpu_delta": 0.0, "mem_delta": 0.0})
    moved = manager.baseline_of(device).loc["packet_rate"]
    limit = config.learning.max_single_update_fraction * start
    assert abs(moved - start) <= limit + 1e-9


def test_legitimate_volume_shift_is_eventually_learned(learning_world):
    """
    A device that legitimately moves its volume by ~35% must be absorbed by the
    adaptive baseline instead of being flagged forever.
    """
    config, _, _, _, manager, _ = learning_world
    device = "dev_03"
    baseline = manager.baseline_of(device)
    start = baseline.loc["packet_rate"]
    shifted = dict(baseline.loc)
    shifted["packet_rate"] = start * 1.35
    shifted["bytes_per_sec"] = baseline.loc["bytes_per_sec"] * 1.35
    features = {"pkt_rate_ratio": 1.35, "bytes_rate_ratio": 1.35, "failed_auth_rate": 0.0,
                "protocol_shift": 0.0, "conn_count_z": 0.2, "dest_diversity_z": 0.0,
                "cpu_delta": 0.01, "mem_delta": 0.01}
    modes = []
    for step in range(300):
        event = manager.step(step=step, device_id=device, observation=shifted,
                             score=0.2, threshold=0.5, is_anomalous=False, features=features)
        modes.append(event.mode)
    assert any(mode in {"drift", "clean"} for mode in modes)
    assert manager.baseline_of(device).loc["packet_rate"] > start * 1.15


def test_drift_evidence_requires_features(learning_world):
    _, _, _, _, manager, _ = learning_world
    assert manager.drift_evidence(None) is False
    assert manager.drift_evidence({}) is False
    benign = {"pkt_rate_ratio": 1.1, "bytes_rate_ratio": 1.1, "failed_auth_rate": 0.0,
              "protocol_shift": 0.0, "conn_count_z": 0.5, "dest_diversity_z": 0.5,
              "cpu_delta": 0.01, "mem_delta": 0.01}
    assert manager.drift_evidence(benign) is True
    for key, value in (("pkt_rate_ratio", 8.0), ("failed_auth_rate", 5.0),
                       ("protocol_shift", 1.0), ("dest_diversity_z", 9.0),
                       ("cpu_delta", 0.9)):
        hostile = dict(benign)
        hostile[key] = value
        assert manager.drift_evidence(hostile) is False, key
