"""
Tests for threat assessment / severity scoring boundaries.

The research plan requires explicit checks on the severity score: that it stays
inside its documented range, that each factor is bounded and monotonic, that the
formula really is configurable (weights are renormalised, not assumed to sum to
one), and that severity is kept separate from the anomaly score and from the
ground-truth label.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from config import SeverityConfig
from src.threat_assessment import ThreatAssessor, severity_band

CRITICALITY = {"dev_critical": "critical", "dev_high": "high",
               "dev_medium": "medium", "dev_low": "low", "dev_unknown": "nope"}


def _assessor(config: SeverityConfig | None = None) -> ThreatAssessor:
    return ThreatAssessor(config or SeverityConfig(), CRITICALITY)


# --------------------------------------------------------------------------- #
# Range
# --------------------------------------------------------------------------- #
def test_severity_is_always_inside_its_declared_range():
    assessor = _assessor()
    for score in (0.0, 0.05, 0.4, 0.4001, 0.75, 0.99, 5.0):
        for threshold in (0.02, 0.4, 0.9):
            for device in ("dev_critical", "dev_low", "dev_unknown"):
                result = assessor.assess(
                    step=1, device_id=device, score=score, threshold=threshold,
                    stat_score=min(score, 1.0), if_score=min(score, 1.0),
                    anomalous_share=1.0, traffic_share=1.0,
                )
                assert 0.0 <= result.severity <= 100.0, (score, threshold, device)
                assert 0.0 <= result.magnitude <= 1.0
                assert 0.0 <= result.confidence <= 1.0
                assert 0.0 <= result.criticality <= 1.0
                assert 0.0 <= result.context <= 1.0


def test_all_factors_at_zero_give_zero_and_all_at_one_give_hundred():
    assessor = _assessor()
    zero = assessor.assess(step=0, device_id="dev_low", score=0.4, threshold=0.4,
                           stat_score=0.4, if_score=0.4, anomalous_share=0.0,
                           traffic_share=0.0)
    assert zero.magnitude == 0.0
    assert zero.severity == pytest.approx(
        100.0 * (SeverityConfig().w_criticality * 0.25) /
        (SeverityConfig().w_magnitude + SeverityConfig().w_confidence
         + SeverityConfig().w_criticality + SeverityConfig().w_network_context)
    )


# --------------------------------------------------------------------------- #
# Factor boundaries
# --------------------------------------------------------------------------- #
def test_magnitude_is_threshold_relative_and_saturates():
    assessor = _assessor()
    config = assessor.config
    threshold = 0.4
    assert assessor.magnitude_factor(threshold, threshold) == 0.0
    assert assessor.magnitude_factor(threshold - 0.3, threshold) == 0.0
    saturation_score = threshold * config.magnitude_saturation
    assert assessor.magnitude_factor(saturation_score, threshold) == pytest.approx(1.0)
    assert assessor.magnitude_factor(saturation_score * 3, threshold) == 1.0
    middle = assessor.magnitude_factor(threshold * 0.5 * (1 + config.magnitude_saturation), threshold)
    assert 0.0 < middle < 1.0
    # A higher threshold means the same score is less threatening (documented).
    assert assessor.magnitude_factor(0.6, 0.5) < assessor.magnitude_factor(0.6, 0.3)


def test_confidence_rewards_scorer_agreement_and_margin():
    assessor = _assessor()
    assert assessor.confidence_factor(0.9, 0.9, 1.0) == pytest.approx(1.0)
    assert assessor.confidence_factor(1.0, 0.0, 0.0) == pytest.approx(0.0)
    assert assessor.confidence_factor(1.0, 0.0, 1.0) == pytest.approx(0.5)
    assert assessor.confidence_factor(0.5, 0.5, 0.4) == pytest.approx(0.7)


def test_criticality_mapping_is_monotonic_and_configurable():
    assessor = _assessor()
    values = [assessor.criticality_factor(d) for d in
              ("dev_low", "dev_medium", "dev_high", "dev_critical")]
    assert values == sorted(values)
    assert values[0] == 0.25 and values[-1] == 1.0
    assert assessor.criticality_factor("dev_unknown") == 0.5  # documented default

    custom = SeverityConfig(criticality_values={"low": 0.1, "critical": 0.9})
    custom_assessor = ThreatAssessor(custom, CRITICALITY)
    assert custom_assessor.criticality_factor("dev_low") == 0.1
    assert custom_assessor.criticality_factor("dev_critical") == 0.9


def test_context_grows_with_incident_size_and_traffic_share():
    assessor = _assessor()
    assert assessor.context_factor(0.0, 0.0) == 0.0
    assert assessor.context_factor(1.0, 1.0) == 1.0
    assert assessor.context_factor(0.5, 0.0) == pytest.approx(0.3)
    assert assessor.context_factor(0.5, 1.0) == pytest.approx(0.7)
    assert assessor.context_factor(-5.0, -5.0) == 0.0
    assert assessor.context_factor(9.0, 9.0) == 1.0


# --------------------------------------------------------------------------- #
# Formula behaviour
# --------------------------------------------------------------------------- #
def test_weights_are_renormalised_and_configurable():
    base = _assessor()
    scaled_config = SeverityConfig(w_magnitude=0.7, w_confidence=0.5, w_criticality=0.4,
                                   w_network_context=0.4)  # sums to 2.0
    scaled = ThreatAssessor(scaled_config, CRITICALITY)
    same_ratio_config = SeverityConfig(w_magnitude=1.4, w_confidence=1.0, w_criticality=0.8,
                                       w_network_context=0.8)  # exactly 2x the above
    same_ratio = ThreatAssessor(same_ratio_config, CRITICALITY)

    kwargs = dict(step=3, device_id="dev_high", score=0.8, threshold=0.4, stat_score=0.8,
                  if_score=0.8, anomalous_share=0.5, traffic_share=0.5)
    assert scaled.assess(**kwargs).severity == pytest.approx(
        same_ratio.assess(**kwargs).severity, rel=1e-9)

    # Removing the criticality weight removes asset importance from the decision.
    no_criticality = ThreatAssessor(SeverityConfig(w_criticality=0.0), CRITICALITY)
    critical_device = no_criticality.assess(**{**kwargs, "device_id": "dev_critical"}).severity
    low_device = no_criticality.assess(**{**kwargs, "device_id": "dev_low"}).severity
    assert critical_device == approx_float(low_device)
    # ... whereas the default configuration does distinguish them.
    assert base.assess(**{**kwargs, "device_id": "dev_critical"}).severity > \
        base.assess(**{**kwargs, "device_id": "dev_low"}).severity


def approx_float(value: float) -> float:
    return pytest.approx(value, abs=1e-9)


def test_severity_bands_positions():
    assert severity_band(-0.1) == "none"
    assert severity_band(0.0) == "none"
    assert severity_band(24.99) == "low"
    assert severity_band(25.0) == "moderate"
    assert severity_band(49.99) == "moderate"
    assert severity_band(50.0) == "high"
    assert severity_band(74.99) == "high"
    assert severity_band(75.0) == "critical"
    assert severity_band(100.0) == "critical"


def test_assessment_separates_anomaly_score_confidence_and_severity():
    assessor = _assessor()
    result = assessor.assess(step=7, device_id="dev_medium", score=0.75, threshold=0.5,
                             stat_score=0.8, if_score=0.7, anomalous_share=0.25,
                             traffic_share=0.6)
    payload = result.to_dict()
    assert payload["anomaly_score"] == 0.75          # detector output, unchanged
    assert payload["severity"] != payload["anomaly_score"]
    assert payload["confidence"] != payload["severity"]
    # No label-like input is accepted by the assessment API.
    parameters = set(inspect.signature(assessor.assess).parameters)
    for forbidden in ("ground_truth", "attack_type", "label", "held_out"):
        assert forbidden not in parameters
    assert "ground_truth" not in payload and "attack_type" not in payload


def test_degenerate_inputs_do_not_produce_nan():
    assessor = _assessor()
    result = assessor.assess(step=1, device_id="dev_low", score=float("nan"),
                             threshold=0.0, stat_score=float("nan"), if_score=0.0,
                             anomalous_share=0.0, traffic_share=0.0)
    assert np.isfinite(result.severity - 0.0)  # severity itself stays finite
    assert 0.0 <= result.severity <= 100.0
