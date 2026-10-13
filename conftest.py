"""
pytest configuration and shared fixtures.

Makes ``config`` and ``src`` importable regardless of the directory pytest is
invoked from, and builds the small, fixed-seed simulation world the tests share
(session-scoped so the fleet is generated once instead of once per test).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import FEATURE_NAMES, ExperimentConfig  # noqa: E402
from src.adaptive_learning import AdaptiveBaselineManager  # noqa: E402
from src.anomaly_detector import AnomalyDetector  # noqa: E402
from src.feature_extractor import FeatureExtractor  # noqa: E402
from src.traffic_simulator import TrafficSimulator  # noqa: E402


@pytest.fixture(scope="session")
def quick_config() -> ExperimentConfig:
    """Small but complete configuration: keeps the whole suite fast."""
    return ExperimentConfig(
        seed=4242, n_devices=12, n_steps=200, calibration_steps=60, warmup_steps=65,
    )


@pytest.fixture(scope="session")
def world(quick_config):
    """(simulator, telemetry, attacks) for the shared scenario."""
    simulator = TrafficSimulator(quick_config)
    telemetry, attacks = simulator.generate()
    return simulator, telemetry, attacks


@pytest.fixture(scope="session")
def calibrated(world):
    """
    A calibrated (manager, extractor, detector) triple plus the calibration matrices.

    Returns a dict so tests can pick what they need without unpacking positions.
    """
    simulator, telemetry, _ = world
    config = simulator.config
    manager = AdaptiveBaselineManager(config.learning)
    manager.fit(telemetry, config.calibration_steps)
    extractor = FeatureExtractor(
        manager,
        {d.device_id: d.allowed_protocols for d in simulator.devices},
        {d.device_id: d.normal_protocol for d in simulator.devices},
    )
    calibration = telemetry[telemetry["step"] < config.calibration_steps]
    vectors = [extractor.extract(row) for _, row in calibration.iterrows()]
    X = np.vstack([v.values for v in vectors])
    device_ids = [v.device_id for v in vectors]
    detector = AnomalyDetector(config.detector, seed=config.seed, adaptive=True)
    detector.calibrate(pd.DataFrame(X, columns=list(FEATURE_NAMES)), device_ids)
    return {
        "config": config, "simulator": simulator, "telemetry": telemetry,
        "manager": manager, "extractor": extractor, "detector": detector,
        "X_calibration": X, "calibration_ids": device_ids,
    }


def make_detector(config, adaptive: bool = True) -> AnomalyDetector:
    return AnomalyDetector(config.detector, seed=config.seed, adaptive=adaptive)
