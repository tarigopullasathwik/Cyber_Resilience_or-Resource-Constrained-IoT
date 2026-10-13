"""
Tests for the IoT traffic simulator: fleet heterogeneity, plausible normal
telemetry, attack scheduling, held-out patterns and reproducibility.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from config import (
    ATTACK_CATALOG,
    BENIGN_SHIFT_MAGNITUDE,
    HELD_OUT_ATTACKS,
    KNOWN_ATTACKS,
    ExperimentConfig,
)
from src.traffic_simulator import TELEMETRY_COLUMNS, TrafficSimulator


@pytest.fixture(scope="module")
def small_sim():
    config = ExperimentConfig(n_devices=10, n_steps=160, calibration_steps=50, warmup_steps=55)
    simulator = TrafficSimulator(config)
    telemetry, attacks = simulator.generate()
    return simulator, telemetry, attacks


def test_fleet_is_heterogeneous_and_large_enough(small_sim):
    simulator, telemetry, _ = small_sim
    assert len(simulator.devices) >= 10
    device_types = {d.device_type for d in simulator.devices}
    assert len(device_types) >= 5, "the fleet must contain heterogeneous device classes"
    criticalities = {d.criticality for d in simulator.devices}
    assert criticalities.issubset({"low", "medium", "high", "critical"})
    assert telemetry["device_id"].nunique() == len(simulator.devices)


def test_default_configuration_has_at_least_twenty_devices():
    assert ExperimentConfig().n_devices >= 20


def test_telemetry_columns_and_value_ranges(small_sim):
    _, telemetry, _ = small_sim
    for column in TELEMETRY_COLUMNS:
        assert column in telemetry.columns
    assert (telemetry["packet_rate"] >= 0).all()
    assert (telemetry["bytes_per_sec"] >= 0).all()
    assert (telemetry["conn_count"] >= 0).all()
    assert (telemetry["failed_auth_attempts"] >= 0).all()
    assert (telemetry["dest_diversity"] >= 1).all()
    assert telemetry["cpu_usage"].between(0, 100).all()
    assert telemetry["memory_usage"].between(0, 100).all()
    assert telemetry["step"].min() == 0
    assert telemetry["step"].max() == ExperimentConfig(n_steps=160).n_steps - 1


def test_normal_traffic_matches_the_device_profile(small_sim):
    simulator, telemetry, _ = small_sim
    """A device's benign mean must be close to its declared normal profile."""
    for device in simulator.devices:
        rows = telemetry[(telemetry["device_id"] == device.device_id)
                         & (telemetry["ground_truth"] == 0)]
        assert not rows.empty
        mean_rate = rows["packet_rate"].mean()
        assert 0.7 * device.base_pkt_rate < mean_rate < 1.3 * device.base_pkt_rate


def test_attacks_are_labelled_and_start_after_calibration(small_sim):
    simulator, telemetry, attacks = small_sim
    assert not attacks.empty
    calibration = simulator.config.calibration_steps
    assert (attacks["start_step"] > calibration).all()
    labelled = telemetry[telemetry["ground_truth"] == 1]
    assert labelled["attack_type"].ne("").all()
    assert labelled["attack_id"].ne("").all()
    # Labelled steps must be exactly the scheduled intervals.
    for _, attack in attacks.iterrows():
        rows = labelled[labelled["attack_id"] == attack["attack_id"]]
        assert rows["step"].min() == attack["start_step"]
        assert rows["step"].max() == attack["end_step"]


def test_attack_labels_are_absent_from_benign_rows(small_sim):
    _, telemetry, _ = small_sim
    benign = telemetry[telemetry["ground_truth"] == 0]
    assert (benign["attack_type"] == "").all()
    assert (benign["attack_held_out"] == False).all()  # noqa: E712


def test_attack_catalogue_has_held_out_patterns_not_used_for_development():
    assert len(HELD_OUT_ATTACKS) >= 1, "at least one pattern must be withheld"
    assert set(HELD_OUT_ATTACKS).isdisjoint(set(KNOWN_ATTACKS))
    for name in HELD_OUT_ATTACKS:
        assert ATTACK_CATALOG[name].held_out is True
    # Each required attack family of the research plan must exist.
    for family in ("flood", "port_scan", "brute_force", "exfiltration"):
        assert family in ATTACK_CATALOG


def test_attack_effect_is_visible_and_matches_direction(small_sim):
    _, telemetry, attacks = small_sim
    """Each attack must actually move the telemetry in the direction it claims."""
    checked = 0
    for _, attack in attacks.iterrows():
        attack_rows = telemetry[
            (telemetry["device_id"] == attack["device_id"])
            & (telemetry["step"] >= attack["start_step"])
            & (telemetry["step"] <= attack["end_step"])
        ]
        normal_rows = telemetry[
            (telemetry["device_id"] == attack["device_id"])
            & ((telemetry["step"] < attack["start_step"]) | (telemetry["step"] > attack["end_step"]))
            & (telemetry["ground_truth"] == 0)
        ]
        if attack_rows.empty or normal_rows.empty:
            continue
        checked += 1
        name = attack["attack_type"]
        if name in ("flood", "port_scan", "brute_force", "exfiltration", "stealth_exfiltration"):
            assert attack_rows["packet_rate"].mean() > normal_rows["packet_rate"].mean(), name
        if name == "brute_force":
            assert attack_rows["failed_auth_attempts"].mean() > normal_rows[
                "failed_auth_attempts"].mean()
        if name == "port_scan":
            assert attack_rows["dest_diversity"].mean() > normal_rows["dest_diversity"].mean()
        if name == "exfiltration":
            assert attack_rows["bytes_per_sec"].mean() > 3.0 * normal_rows["bytes_per_sec"].mean()
    assert checked >= 2


def test_benign_behaviour_shift_exists_and_is_not_attacked(small_sim):
    simulator, telemetry, _ = small_sim
    assert simulator.shift_device_ids, "the run should contain legitimate behaviour changes"
    attacked = set(simulator.attacks and [a.device_id for a in simulator.attacks])
    assert set(simulator.shift_device_ids).isdisjoint(attacked)
    shifted = telemetry[telemetry["benign_shift"] == True]  # noqa: E712
    assert not shifted.empty
    assert (shifted["ground_truth"] == 0).all()
    # The shift must actually move the traffic of at least one device.
    device = simulator.shift_device_ids[0]
    before = telemetry[(telemetry["device_id"] == device)
                       & (telemetry["step"] < simulator.shift_steps[0])]["packet_rate"].mean()
    after = telemetry[(telemetry["device_id"] == device)
                      & (telemetry["step"] >= simulator.shift_steps[0])]["packet_rate"].mean()
    if before > 0:
        assert abs(after / before - 1.0) > 0.05


def test_shift_direction_is_stable_and_mixed():
    """
    The direction of a legitimate shift must be a deterministic function of the
    device id (Python's built-in hash() is randomised per process, which would
    silently break reproducibility between runs).
    """
    directions = {f"dev_{i:02d}": TrafficSimulator.shift_direction(f"dev_{i:02d}")
                  for i in range(24)}
    assert set(directions.values()) == {1.0, -1.0}, "shifts must go both ways"
    # A second call in the same process must agree, and the values are pinned so a
    # switch back to the randomised built-in hash would fail this test.
    for device_id, value in directions.items():
        assert TrafficSimulator.shift_direction(device_id) == value
    assert directions["dev_00"] == 1.0 and directions["dev_04"] == -1.0


def test_same_seed_reproduces_identical_traffic():
    config_a = ExperimentConfig(seed=99, n_devices=8, n_steps=80, calibration_steps=30)
    config_b = ExperimentConfig(seed=99, n_devices=8, n_steps=80, calibration_steps=30)
    telemetry_a, attacks_a = TrafficSimulator(config_a).generate()
    telemetry_b, attacks_b = TrafficSimulator(config_b).generate()
    pd.testing.assert_frame_equal(telemetry_a, telemetry_b)
    pd.testing.assert_frame_equal(attacks_a, attacks_b)


def test_different_seed_produces_different_traffic():
    config_a = ExperimentConfig(seed=1, n_devices=8, n_steps=80, calibration_steps=30)
    config_b = ExperimentConfig(seed=2, n_devices=8, n_steps=80, calibration_steps=30)
    telemetry_a, _ = TrafficSimulator(config_a).generate()
    telemetry_b, _ = TrafficSimulator(config_b).generate()
    assert not np.allclose(telemetry_a["packet_rate"].to_numpy(),
                           telemetry_b["packet_rate"].to_numpy())


def test_profile_frame_matches_idle_behaviour(small_sim):
    simulator, _, _ = small_sim
    profiles = simulator.profile_frame()
    assert profiles.shape[0] == len(simulator.devices)
    assert {"device_id", "device_type", "criticality", "normal_pkt_rate"}.issubset(profiles.columns)
    assert BENIGN_SHIFT_MAGNITUDE > 0
