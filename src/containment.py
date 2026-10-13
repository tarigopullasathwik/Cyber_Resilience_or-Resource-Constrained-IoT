"""
Autonomous containment policy and closed-loop controller (component F).

Response levels (simulated only - this code never touches a real network,
firewall or device):

===============  ===============================================================
Level            Meaning in the simulation
===============  ===============================================================
``OBSERVE``      Record the event and keep monitoring; nothing is changed.
``RATE_LIMIT``   Reduce the simulated traffic allowance to 35%.
``REAUTHENTICATE`` Issue a simulated authentication challenge; allowance 10%.
``QUARANTINE``   Isolate the device in the simulation; allowance 0, no telemetry.
===============  ===============================================================

Decision inputs (all label-free): severity from ``threat_assessment``, the device
criticality label, the simulated free CPU of the gateway, and the device's own
containment state (current level, cooldown, escalation count, open verification).

Safeguards against unnecessary or harmful isolation
---------------------------------------------------
1. ``critical_asset_guard``   critical devices are quarantined only above
   ``critical_quarantine_severity`` (85 by default).
2. ``graduated_response``     QUARANTINE requires at least one earlier action
   (a lighter response must have been tried) unless severity is already extreme.
3. ``resource_guard``         actions are downgraded or skipped when the simulated
   gateway lacks CPU headroom for the action cost.
4. ``cooldown``               at most one new action per ``cooldown_steps`` steps,
   except when severity jumps by ``escalation_severity_jump`` or a verification
   just failed.
5. ``probe_before_recovery`` restoration is *probed* one level at a time and
   reverted if the anomaly returns within the observation window.
6. ``learning_under_containment`` is disabled at all times, so a device that the
   system is itself throttling can never become its own baseline.

The controller is deterministic: identical inputs produce identical decisions.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import ContainmentConfig, VerificationConfig  # noqa: E402
from src.anomaly_detector import ScoreResult  # noqa: E402
from src.response_verifier import ResponseVerifier, VerificationResult  # noqa: E402

#: Action event kinds written to the containment log.
KIND_APPLY = "APPLY"
KIND_ESCALATE = "ESCALATE"
KIND_OBSERVE = "OBSERVE"
KIND_HOLD = "HOLD"
KIND_DE_ESCALATE = "DE_ESCALATE"
KIND_RESTORE = "RESTORE"
KIND_RECOVERY = "RECOVERY"
KIND_RE_APPLY = "RE_APPLY"
KIND_MAX_LEVEL = "MAX_LEVEL"


@dataclass
class Decision:
    """Outcome of the containment policy for one observation."""

    action: str
    level: int
    reason: str
    notes: List[str] = field(default_factory=list)


@dataclass
class DeviceContainmentState:
    """Per-device containment bookkeeping."""

    device_id: str
    criticality: str = "medium"
    level: int = 0
    level_since: int = 0
    last_action_step: int = -10 ** 6
    cooldown_until: int = -10 ** 6
    escalations: int = 0
    actions_applied: int = 0
    anomaly_streak: int = 0
    recovery_progress: int = 0
    pending_request: Optional[str] = None
    last_outcome: str = ""
    last_severity: float = 0.0
    probing: bool = False
    pre_probe_level: int = 0
    probe_until: int = -1
    first_action_step: int = -1
    last_restore_step: int = -1
    total_contained_steps: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "device_id": self.device_id, "criticality": self.criticality,
            "level": self.level, "action": None, "escalations": self.escalations,
            "actions_applied": self.actions_applied, "anomaly_streak": self.anomaly_streak,
            "recovery_progress": self.recovery_progress, "last_outcome": self.last_outcome,
            "probing": int(self.probing), "total_contained_steps": self.total_contained_steps,
            "first_action_step": self.first_action_step,
            "last_restore_step": self.last_restore_step,
        }


@dataclass
class ActionEvent:
    """One logged containment decision (including holds and recoveries)."""

    step: int
    device_id: str
    kind: str
    action: str
    level: int
    previous_level: int
    severity: float
    score: float
    threshold: float
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step, "device_id": self.device_id, "kind": self.kind,
            "action": self.action, "level": self.level, "previous_level": self.previous_level,
            "severity": round(self.severity, 4),
            "score": round(self.score, 6) if np.isfinite(self.score) else None,
            "threshold": round(self.threshold, 6) if np.isfinite(self.threshold) else None,
            "reason": self.reason,
        }


class ContainmentPolicy:
    """Maps severity + context onto a response level, with safeguards."""

    def __init__(self, config: Optional[ContainmentConfig] = None):
        self.config = config or ContainmentConfig()

    def band_level(self, severity: float) -> Tuple[int, str]:
        cfg = self.config
        # A non-finite severity (corrupt input) must never escalate: every
        # comparison against NaN is False, which would otherwise select the most
        # severe band. Invalid evidence means "keep observing".
        if not np.isfinite(float(severity)):
            return 0, "severity_not_finite"
        if severity <= cfg.observe_max:
            return 0, f"severity<={cfg.observe_max}"
        if severity <= cfg.rate_limit_max:
            return 1, f"severity<={cfg.rate_limit_max}"
        if severity <= cfg.reauth_max:
            return 2, f"severity<={cfg.reauth_max}"
        return 3, f"severity>{cfg.reauth_max}"

    def decide(
        self,
        severity: float,
        state: DeviceContainmentState,
        cpu_free_units: float,
        step: int,
    ) -> Decision:
        """Return the level the policy *wants*, before cooldown handling."""
        cfg = self.config
        level, reason = self.band_level(float(severity))
        notes: List[str] = []

        if level >= 3 and state.criticality == "critical" and \
                float(severity) < cfg.critical_quarantine_severity:
            level = 2
            notes.append("critical_asset_guard")

        if cfg.require_graduated_escalation and level >= 3 and \
                state.escalations == 0 and state.level < 2:
            level = 2
            notes.append("graduated_response")

        free = float(max(cpu_free_units, 0.0))
        if free < cfg.min_cpu_headroom_for_action:
            if level > 0:
                notes.append("resource_guard_observe")
            level = 0
        else:
            while level > 0 and cfg.action_cost[cfg.levels[level]] > free:
                level -= 1
                notes.append("resource_guard_downgrade")

        return Decision(action=cfg.levels[level], level=level, reason=reason, notes=notes)


class ClosedLoopController:
    """
    Owns the containment state of every device and drives the closed loop:
    detect -> assess -> act -> verify -> escalate/recover -> learn.
    """

    def __init__(
        self,
        containment_config: Optional[ContainmentConfig] = None,
        verification_config: Optional[VerificationConfig] = None,
        verifier: Optional[ResponseVerifier] = None,
        device_criticality: Optional[Dict[str, str]] = None,
    ):
        self.config = containment_config or ContainmentConfig()
        self.verification_config = verification_config or VerificationConfig()
        self.verifier = verifier or ResponseVerifier(self.verification_config)
        self.policy = ContainmentPolicy(self.config)
        self.criticality = dict(device_criticality or {})
        self.states: Dict[str, DeviceContainmentState] = {}
        self.action_log: List[ActionEvent] = []
        self.recovery_log: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------ #
    # State helpers
    # ------------------------------------------------------------------ #
    def state_of(self, device_id: str) -> DeviceContainmentState:
        state = self.states.get(device_id)
        if state is None:
            state = DeviceContainmentState(
                device_id=device_id, criticality=self.criticality.get(device_id, "medium")
            )
            self.states[device_id] = state
        return state

    def allowance(self, device_id: str) -> float:
        """Simulated traffic allowance for a device under its current action."""
        state = self.states.get(device_id)
        if state is None:
            return 1.0
        return float(self.config.action_traffic_allowance[self.config.levels[state.level]])

    def action_name(self, device_id: str) -> str:
        state = self.state_of(device_id)
        return self.config.levels[state.level]

    def under_containment(self, device_id: str) -> bool:
        return self.state_of(device_id).level >= 1

    # ------------------------------------------------------------------ #
    # Main entry point
    # ------------------------------------------------------------------ #
    def process(
        self,
        step: int,
        device_id: str,
        score_result: Optional[ScoreResult],
        severity: float,
        observed_indicators: Optional[Dict[str, float]],
        telemetry_available: bool,
        cpu_free_units: float,
    ) -> List[ActionEvent]:
        """
        Process one device for one step and return the action events it produced.

        ``score_result`` is ``None`` when the device is fully isolated and the
        gateway therefore receives no telemetry (quarantine), which is handled as
        "no evidence of ongoing attack, but the isolation itself is being
        verified".
        """
        state = self.state_of(device_id)
        events: List[ActionEvent] = []
        score = float(score_result.score) if score_result is not None else float("nan")
        threshold = float(score_result.threshold) if score_result is not None else float("nan")
        is_anomalous = bool(score_result.is_anomalous) if score_result is not None else False
        # Severity of the *previous* step is what an "escalation jump" is measured
        # against, so it must be captured before this step overwrites it.
        previous_severity = float(state.last_severity)
        if score_result is not None:
            state.last_severity = float(severity)

        # (1) advance any open verification window ----------------------- #
        closed = self._advance_verification(step, device_id, score_result, observed_indicators)

        # (2) a failed verification is strong evidence -> escalate ------- #
        handled = False
        if closed is not None and closed.outcome == self.verification_config.outcome_failure:
            escalation = self._escalate_after_failure(
                step, device_id, closed, cpu_free_units, observed_indicators
            )
            events.extend(escalation)
            handled = bool(escalation)

        # (3) observation handling --------------------------------------- #
        if not handled:
            if is_anomalous:
                state.anomaly_streak += 1
                state.recovery_progress = 0
                if state.probing:
                    events.append(self._re_apply(step, device_id, score, threshold, severity,
                                                 "probe_revealed_persistent_anomaly", cpu_free_units,
                                                 observed_indicators))
                else:
                    events.extend(self._handle_anomaly(step, device_id, score, threshold, severity,
                                                       cpu_free_units, observed_indicators,
                                                       previous_severity=previous_severity))
            else:
                state.anomaly_streak = 0
                events.extend(self._handle_benign_or_isolated(step, device_id, score, threshold,
                                                              severity, telemetry_available,
                                                              cpu_free_units))

        for event in events:
            self.action_log.append(event)
        if state.level >= 1:
            state.total_contained_steps += 1
        return events

    # ------------------------------------------------------------------ #
    # Verification plumbing
    # ------------------------------------------------------------------ #
    def _advance_verification(
        self,
        step: int,
        device_id: str,
        score_result: Optional[ScoreResult],
        observed_indicators: Optional[Dict[str, float]],
    ) -> Optional[VerificationResult]:
        state = self.state_of(device_id)
        if state.pending_request is None:
            return None
        if score_result is None:
            # Fully isolated: the offending flows are definitionally suppressed.
            score = 0.0
            indicators = {k: 0.0 for k in (observed_indicators or {})}
        else:
            score = float(score_result.score)
            indicators = dict(observed_indicators or {})
        result = self.verifier.observe(step, device_id, score, indicators)
        if result is not None:
            state.pending_request = None
            state.last_outcome = result.outcome
            if result.outcome == self.verification_config.outcome_failure:
                state.recovery_progress = 0
        return result

    def _open_verification(
        self,
        step: int,
        device_id: str,
        action: str,
        level: int,
        score: float,
        observed_indicators: Optional[Dict[str, float]],
        allowance: float,
    ) -> None:
        state = self.state_of(device_id)
        if state.pending_request is not None:
            self.verifier.close_as_inconclusive(device_id, step, reason="superseded_by_new_action")
        state.pending_request = self.verifier.request(
            step=step, device_id=device_id, action=action, level_index=level,
            score_before=float(score) if np.isfinite(score) else 0.0,
            indicators=dict(observed_indicators or {}), allowance=float(allowance),
        )

    # ------------------------------------------------------------------ #
    # Anomalous path
    # ------------------------------------------------------------------ #
    def _handle_anomaly(
        self,
        step: int,
        device_id: str,
        score: float,
        threshold: float,
        severity: float,
        cpu_free_units: float,
        indicators: Optional[Dict[str, float]] = None,
        forced_reason: str = "",
        previous_severity: Optional[float] = None,
    ) -> List[ActionEvent]:
        state = self.state_of(device_id)
        decision = self.policy.decide(severity, state, cpu_free_units, step)
        desired = decision.level
        target_action = self.config.levels[desired]

        reason = decision.reason if not decision.notes else \
            f"{decision.reason}|notes:{','.join(decision.notes)}"
        if desired <= state.level:
            # Nothing new to do: the current action already covers this severity.
            if desired == 0:
                if step - state.last_action_step >= self.config.cooldown_steps:
                    state.last_action_step = step
                    return [self._event(step, device_id, KIND_OBSERVE, target_action, 0, 0,
                                        severity, score, threshold, reason)]
                return []
            # Sustained anomalous traffic while a response is already in place:
            # escalate one level (documented behaviour, not a cooldown violation).
            if state.anomaly_streak >= int(self.config.sustained_anomaly_steps) \
                    and state.level < len(self.config.levels) - 1:
                return self._escalate(step, device_id, score, threshold, severity, cpu_free_units,
                                      reason="sustained_anomaly_after_effective_response",
                                      indicators=indicators)
            return []

        # cooldown guard -------------------------------------------------- #
        within_cooldown = step < state.cooldown_until
        reference = float(state.last_severity) if previous_severity is None else previous_severity
        severity_jump = (float(severity) - reference) >= self.config.escalation_severity_jump
        if within_cooldown and not forced_reason and not severity_jump:
            return [self._event(step, device_id, KIND_HOLD, self.config.levels[state.level],
                                state.level, state.level, severity, score, threshold,
                                f"cooldown_guard(desired={target_action},reason={decision.reason})")]

        return self._escalate(step, device_id, score, threshold, severity, cpu_free_units,
                              reason=forced_reason or reason,
                              notes=decision.notes, target_level=desired,
                              indicators=indicators)

    def _escalate_after_failure(
        self,
        step: int,
        device_id: str,
        verification: VerificationResult,
        cpu_free_units: float,
        indicators: Optional[Dict[str, float]] = None,
    ) -> List[ActionEvent]:
        """A verification failed -> stronger containment is justified."""
        state = self.state_of(device_id)
        if state.escalations >= int(self.verification_config.max_escalations) \
                or state.level >= len(self.config.levels) - 1:
            return [self._event(step, device_id, KIND_MAX_LEVEL, self.config.levels[state.level],
                                state.level, state.level, state.last_severity, float("nan"),
                                float("nan"), "verification_failed_at_max_level")]
        return self._escalate(
            step, device_id, float("nan"), float("nan"), state.last_severity, cpu_free_units,
            reason="verification_failure",
            target_level=min(state.level + 1, len(self.config.levels) - 1),
            indicators=indicators,
        )

    def _escalate(
        self,
        step: int,
        device_id: str,
        score: float,
        threshold: float,
        severity: float,
        cpu_free_units: float,
        reason: str,
        notes: Optional[Sequence[str]] = None,
        target_level: Optional[int] = None,
        indicators: Optional[Dict[str, float]] = None,
    ) -> List[ActionEvent]:
        state = self.state_of(device_id)
        desired = state.level + 1 if target_level is None else int(target_level)
        desired = int(np.clip(desired, 1, len(self.config.levels) - 1))
        # Resource guard applies to enforced escalations as well.
        while desired > 1 and self.config.action_cost[self.config.levels[desired]] > max(cpu_free_units, 0.0):
            desired -= 1
        previous = state.level
        state.level = desired
        state.level_since = step
        state.last_action_step = step
        state.cooldown_until = step + int(self.config.cooldown_steps)
        state.escalations += 1
        state.actions_applied += 1
        state.probing = False
        if state.first_action_step < 0:
            state.first_action_step = step
        note_text = ",".join(list(notes or [])) or "-"
        kind = KIND_ESCALATE if previous > 0 else KIND_APPLY
        self._open_verification(
            step, device_id, self.config.levels[desired], desired, score,
            indicators, self.config.action_traffic_allowance[self.config.levels[desired]],
        )
        return [self._event(step, device_id, kind, self.config.levels[desired], desired, previous,
                            severity, score, threshold, f"{reason}|notes:{note_text}")]

    # ------------------------------------------------------------------ #
    # Benign / isolated path (recovery, probes, restores)
    # ------------------------------------------------------------------ #
    def _handle_benign_or_isolated(
        self,
        step: int,
        device_id: str,
        score: float,
        threshold: float,
        severity: float,
        telemetry_available: bool,
        cpu_free_units: float,
    ) -> List[ActionEvent]:
        state = self.state_of(device_id)
        events: List[ActionEvent] = []
        if state.probing:
            if step >= state.probe_until:
                state.probing = False
                state.recovery_progress = 0
            return events

        if state.level == 0:
            return events

        # Evidence supports recovery: no anomaly, and either no open window or a
        # successful verification of the current action.
        verification_ok = state.pending_request is None and \
            state.last_outcome == self.verification_config.outcome_success
        evidence = (not telemetry_available) or verification_ok
        if not evidence:
            return events

        state.recovery_progress += 1
        if state.recovery_progress < int(self.config.recovery_steps):
            return events

        if state.level == 1:
            return [self._restore(step, device_id, score, threshold, severity)]
        return [self._de_escalate(step, device_id, score, threshold, severity, cpu_free_units)]

    def _de_escalate(
        self,
        step: int,
        device_id: str,
        score: float,
        threshold: float,
        severity: float,
        cpu_free_units: float,
    ) -> ActionEvent:
        """Step down one level as a *probe*; reverted if the anomaly returns."""
        state = self.state_of(device_id)
        previous = state.level
        state.pre_probe_level = previous
        state.level = max(0, previous - 1)
        state.probing = True
        state.probe_until = step + int(self.verification_config.observation_window)
        state.recovery_progress = 0
        state.last_action_step = step
        state.cooldown_until = step + int(self.config.cooldown_steps)
        return self._event(step, device_id, KIND_DE_ESCALATE, self.config.levels[state.level],
                           state.level, previous, severity, score, threshold,
                           "probe_before_full_restore")

    def _restore(
        self, step: int, device_id: str, score: float, threshold: float, severity: float
    ) -> ActionEvent:
        state = self.state_of(device_id)
        previous = state.level
        state.level = 0
        state.probing = False
        state.recovery_progress = 0
        state.anomaly_streak = 0
        state.last_action_step = step
        state.cooldown_until = step + int(self.config.cooldown_steps)
        recovery_time = step - state.first_action_step if state.first_action_step >= 0 else -1
        state.last_restore_step = step
        state.first_action_step = -1
        state.last_outcome = ""
        self.recovery_log.append(
            {"step": int(step), "device_id": device_id, "previous_action":
                self.config.levels[previous], "recovery_time_steps": int(recovery_time)}
        )
        return self._event(step, device_id, KIND_RESTORE, self.config.levels[0], 0, previous,
                           severity, score, threshold, f"full_restore_after_recovery({recovery_time} steps)")

    def _re_apply(
        self,
        step: int,
        device_id: str,
        score: float,
        threshold: float,
        severity: float,
        reason: str,
        cpu_free_units: float,
        indicators: Optional[Dict[str, float]] = None,
    ) -> ActionEvent:
        """A probe failed: put the previous action back immediately."""
        state = self.state_of(device_id)
        previous = state.level
        target = max(1, state.pre_probe_level)
        while target > 1 and self.config.action_cost[self.config.levels[target]] > max(cpu_free_units, 0.0):
            target -= 1
        state.level = target
        state.probing = False
        state.recovery_progress = 0
        state.last_action_step = step
        state.cooldown_until = step + int(self.config.cooldown_steps)
        state.actions_applied += 1
        if state.first_action_step < 0:
            state.first_action_step = step
        self._open_verification(
            step, device_id, self.config.levels[target], target, score,
            indicators, self.config.action_traffic_allowance[self.config.levels[target]],
        )
        return self._event(step, device_id, KIND_RE_APPLY, self.config.levels[target], target,
                           previous, severity, score, threshold, reason)

    # ------------------------------------------------------------------ #
    # Small helpers
    # ------------------------------------------------------------------ #
    def _event(
        self, step: int, device_id: str, kind: str, action: str, level: int, previous_level: int,
        severity: float, score: float, threshold: float, reason: str,
    ) -> ActionEvent:
        return ActionEvent(
            step=int(step), device_id=device_id, kind=kind, action=action, level=int(level),
            previous_level=int(previous_level),
            severity=float(severity) if np.isfinite(severity) else 0.0,
            score=float(score), threshold=float(threshold), reason=reason,
        )

    # ------------------------------------------------------------------ #
    # Introspection (dashboard / metrics)
    # ------------------------------------------------------------------ #
    def action_log_frame(self) -> pd.DataFrame:
        if not self.action_log:
            return pd.DataFrame(columns=["step", "device_id", "kind", "action", "level",
                                         "previous_level", "severity", "score", "threshold", "reason"])
        return pd.DataFrame([e.to_dict() for e in self.action_log])

    def state_frame(self) -> pd.DataFrame:
        rows = []
        for state in self.states.values():
            row = state.to_dict()
            row["action"] = self.config.levels[state.level]
            row["allowance"] = self.config.action_traffic_allowance[self.config.levels[state.level]]
            rows.append(row)
        return pd.DataFrame(rows)

    def recovery_frame(self) -> pd.DataFrame:
        if not self.recovery_log:
            return pd.DataFrame(columns=["step", "device_id", "previous_action", "recovery_time_steps"])
        return pd.DataFrame(self.recovery_log)

    def counters(self) -> Dict[str, Any]:
        frame = self.action_log_frame()
        counts: Dict[str, int] = {}
        if not frame.empty:
            counts = frame[frame["level"] >= 1]["action"].value_counts().to_dict()
        return {
            "actions_by_type": counts,
            "n_action_events": int(len(frame)),
            "n_recoveries": len(self.recovery_log),
            "n_verifications": len(self.verifier.results),
            "verification_outcomes": self.verifier.outcome_counts(),
        }


if __name__ == "__main__":  # pragma: no cover - manual smoke test
    from src.anomaly_detector import ScoreResult

    controller = ClosedLoopController(device_criticality={"dev_00": "high"})
    result = ScoreResult(
        step=10, device_id="dev_00", score=0.9, stat_score=0.92, if_score=0.86,
        threshold=0.5, is_anomalous=True,
        contributions={"pkt_rate_ratio": 0.95, "bytes_rate_ratio": 0.9},
    )
    indicators = {"pkt_rate_ratio": 8.0, "bytes_rate_ratio": 6.0}
    for event in controller.process(10, "dev_00", result, severity=60.0,
                                    observed_indicators=indicators, telemetry_available=True,
                                    cpu_free_units=1.0):
        print("applied:", event.to_dict())
    print("allowance now:", controller.allowance("dev_00"))
    for step in range(11, 26):
        benign = ScoreResult(step=step, device_id="dev_00", score=0.05, stat_score=0.05,
                             if_score=0.05, threshold=0.5, is_anomalous=False,
                             contributions={})
        for event in controller.process(step, "dev_00", benign, severity=2.0,
                                        observed_indicators={"pkt_rate_ratio": 1.0},
                                        telemetry_available=True, cpu_free_units=1.0):
            print("recovery:", event.kind, event.action, event.reason)
    print("counters:", controller.counters())
