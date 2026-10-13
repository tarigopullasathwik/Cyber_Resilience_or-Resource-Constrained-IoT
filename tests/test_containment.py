"""
Tests for the containment policy and closed-loop controller: severity bands,
criticality and resource safeguards, cooldowns, escalation on verification
failure, probes, recovery and the no-action-when-benign guarantee.
"""

from __future__ import annotations

import numpy as np
import pytest

from config import ContainmentConfig, VerificationConfig
from src.anomaly_detector import ScoreResult
from src.containment import ClosedLoopController, ContainmentPolicy
from src.response_verifier import ResponseVerifier


def make_result(step: int, device_id: str, score: float = 0.9, threshold: float = 0.4,
                anomalous: bool = True) -> ScoreResult:
    return ScoreResult(
        step=step, device_id=device_id, score=score, stat_score=score, if_score=score,
        threshold=threshold, is_anomalous=anomalous,
        contributions={"pkt_rate_ratio": 0.9, "bytes_rate_ratio": 0.8, "failed_auth_rate": 0.0,
                       "conn_count_z": 0.2, "dest_diversity_z": 0.1, "cpu_delta": 0.3,
                       "mem_delta": 0.1, "protocol_shift": 0.0},
    )


INDICATORS = {"pkt_rate_ratio": 8.0, "bytes_rate_ratio": 6.0}


# --------------------------------------------------------------------------- #
# Policy
# --------------------------------------------------------------------------- #
def test_severity_bands_map_to_the_expected_levels():
    policy = ContainmentPolicy()
    controller = ClosedLoopController(
        device_criticality={"dev_a": "medium", "dev_b": "medium"},
    )
    state = controller.state_of("dev_a")
    state.escalations = 2  # quarantine is allowed once lighter responses were tried
    state.level = 2
    assert policy.decide(10.0, state, 1.0, 0).action == "OBSERVE"
    assert policy.decide(35.0, state, 1.0, 0).action == "RATE_LIMIT"
    assert policy.decide(60.0, state, 1.0, 0).action == "REAUTHENTICATE"
    assert policy.decide(95.0, state, 1.0, 0).action == "QUARANTINE"


def test_critical_devices_are_not_quarantined_below_the_critical_threshold():
    controller = ClosedLoopController(device_criticality={"dev_plc": "critical"})
    state = controller.state_of("dev_plc")
    state.escalations = 3
    state.level = 2
    decision = controller.policy.decide(80.0, state, 1.0, 0)
    assert decision.action == "REAUTHENTICATE"
    assert "critical_asset_guard" in decision.notes
    assert controller.policy.decide(95.0, state, 1.0, 0).action == "QUARANTINE"


def test_graduated_response_requires_a_lighter_action_first():
    controller = ClosedLoopController(device_criticality={"dev_x": "high"})
    state = controller.state_of("dev_x")
    assert state.escalations == 0 and state.level == 0
    decision = controller.policy.decide(99.0, state, 1.0, 0)
    assert decision.action == "REAUTHENTICATE"
    assert "graduated_response" in decision.notes


def test_resource_guard_skips_actions_when_the_gateway_is_exhausted():
    controller = ClosedLoopController(device_criticality={"dev_x": "high"})
    state = controller.state_of("dev_x")
    state.escalations = 1
    decision = controller.policy.decide(95.0, state, cpu_free_units=0.0, step=0)
    assert decision.action == "OBSERVE"
    assert "resource_guard_observe" in decision.notes


# --------------------------------------------------------------------------- #
# Controller behaviour
# --------------------------------------------------------------------------- #
def _controller(criticality=None) -> ClosedLoopController:
    return ClosedLoopController(
        ContainmentConfig(), VerificationConfig(),
        ResponseVerifier(VerificationConfig()), criticality or {"dev_a": "medium"},
    )


def test_a_containment_action_is_applied_and_verified():
    controller = _controller()
    events = controller.process(10, "dev_a", make_result(10, "dev_a"), severity=60.0,
                                observed_indicators=INDICATORS, telemetry_available=True,
                                cpu_free_units=1.0)
    applied = [e for e in events if e.level >= 1]
    assert applied and applied[0].kind == "APPLY"
    assert controller.allowance("dev_a") < 1.0
    assert controller.state_of("dev_a").pending_request is not None
    # The allowed traffic allowance for REAUTHENTICATE is 0.10.
    assert controller.allowance("dev_a") == pytest.approx(0.10)


def test_cooldown_blocks_a_second_action_but_holds_the_first():
    controller = _controller()
    first = controller.process(10, "dev_a", make_result(10, "dev_a"), 49.0, INDICATORS, True, 1.0)
    assert [e.action for e in first if e.level >= 1] == ["RATE_LIMIT"]
    # Severity now asks for a stronger response, but the cooldown has not expired
    # and the severity jump is small -> the policy holds instead of escalating.
    events = controller.process(11, "dev_a", make_result(11, "dev_a"), 52.0, INDICATORS, True, 1.0)
    assert [e.kind for e in events] == ["HOLD"]
    assert controller.state_of("dev_a").level == 1


def test_large_severity_jump_overrides_the_cooldown():
    controller = _controller()
    controller.process(10, "dev_a", make_result(10, "dev_a"), 30.0, INDICATORS, True, 1.0)
    events = controller.process(11, "dev_a", make_result(11, "dev_a"), 95.0, INDICATORS, True, 1.0)
    assert any(e.kind == "ESCALATE" for e in events), events


def test_failed_verification_escalates_the_response():
    controller = _controller()
    controller.process(10, "dev_a", make_result(10, "dev_a"), 60.0, INDICATORS, True, 1.0)
    level_after_apply = controller.state_of("dev_a").level
    # Observation window (4 steps) with no reduction at all -> FAILURE -> escalate.
    outcome = None
    for step in range(11, 15):
        events = controller.process(step, "dev_a", make_result(step, "dev_a", score=0.95),
                                    60.0, INDICATORS, True, 1.0)
        if any(e.kind in {"ESCALATE", "MAX_LEVEL"} for e in events):
            outcome = events
    assert outcome is not None, "a failure must escalate or report the maximum level"
    assert controller.state_of("dev_a").level >= level_after_apply
    assert any(r.outcome == "FAILURE" for r in controller.verifier.results)


def test_successful_verification_leads_to_recovery():
    controller = _controller()
    controller.process(10, "dev_a", make_result(10, "dev_a"), 30.0, INDICATORS, True, 1.0)
    assert controller.state_of("dev_a").level == 1
    recovered = False
    for step in range(11, 60):
        benign = make_result(step, "dev_a", score=0.05, anomalous=False)
        events = controller.process(step, "dev_a", benign, 5.0, {"pkt_rate_ratio": 1.0},
                                    True, 1.0)
        if any(e.kind == "RESTORE" for e in events):
            recovered = True
            break
    assert recovered, "the device should be restored once containment is verified benign"
    assert controller.state_of("dev_a").level == 0
    assert not controller.recovery_frame().empty
    assert controller.recovery_frame()["recovery_time_steps"].iloc[0] > 0


def test_probe_is_reverted_when_the_anomaly_returns():
    controller = _controller()
    controller.process(5, "dev_a", make_result(5, "dev_a"), 60.0, INDICATORS, True, 1.0)
    state = controller.state_of("dev_a")
    # Manually put the controller in the "step down happened" state to isolate the probe.
    state.pre_probe_level = 2
    state.level = 1
    state.probing = True
    state.probe_until = 20
    events = controller.process(12, "dev_a", make_result(12, "dev_a"), 70.0, INDICATORS, True, 1.0)
    assert any(e.kind == "RE_APPLY" for e in events)
    assert controller.state_of("dev_a").level == 2
    assert controller.state_of("dev_a").probing is False


def test_a_benign_stream_never_triggers_containment():
    controller = _controller()
    for step in range(200):
        events = controller.process(step, "dev_a", make_result(step, "dev_a", score=0.1,
                                                               anomalous=False),
                                    severity=3.0, observed_indicators={}, telemetry_available=True,
                                    cpu_free_units=1.0)
        assert all(e.level == 0 for e in events), events
    assert controller.action_log_frame().empty or \
        (controller.action_log_frame()["level"] == 0).all()


def test_fully_isolated_device_still_verifies_its_quarantine():
    controller = _controller()
    controller.process(5, "dev_a", make_result(5, "dev_a"), 99.0, INDICATORS, True, 1.0)
    state = controller.state_of("dev_a")
    # Pretend the lighter responses were already tried and were not enough.
    state.escalations = 2
    state.level = 2
    state.cooldown_until = 0
    events = controller.process(6, "dev_a", make_result(6, "dev_a"), 99.0, INDICATORS, True, 1.0)
    assert any(e.action == "QUARANTINE" for e in events), events
    assert controller.allowance("dev_a") == 0.0
    # With no telemetry at all, the open quarantine window must close as a success.
    for step in range(7, 12):
        controller.process(step, "dev_a", None, 99.0, None, telemetry_available=False,
                           cpu_free_units=1.0)
    outcomes = [r.outcome for r in controller.verifier.results]
    assert "SUCCESS" in outcomes


def test_unseen_device_defaults_are_safe():
    controller = _controller()
    assert controller.allowance("never_seen") == 1.0
    assert controller.action_name("never_seen") == "OBSERVE"
    assert controller.under_containment("never_seen") is False
