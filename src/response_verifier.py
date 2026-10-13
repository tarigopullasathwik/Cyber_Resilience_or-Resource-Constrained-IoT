"""
Response verification and recovery accounting (component G of the architecture).

After every containment action the gateway opens an observation window and asks a
single question: *did the simulated anomalous behaviour actually decrease?*
Because containment in this simulation works by reducing the device's traffic
allowance, the score alone cannot answer that question -- a suppressed device can
look "less normal" than a healthy one. The verifier therefore judges on two
independent signals:

1. the fused anomaly score (absolute level and relative reduction), and
2. the *offending indicators*: the raw magnitudes of the features that triggered
   the alarm, measured before and after the action. This is what makes suppressed
   attack traffic (rather than suppressed telemetry) count as a success.

Outcome rules (documented, deterministic, and free of ground-truth labels):

``INCONCLUSIVE``  fewer than ``observation_window`` post-action steps were observed
(e.g. the run ended). Verification cannot be claimed on too little data.
``SUCCESS``       any of: score after <= ``success_absolute_score``;
mean reduction of the offending indicators >= ``success_score_reduction``;
or relative score reduction >= ``success_score_reduction``.
``FAILURE``       none of the above held over a complete observation window.

The verifier also records *why* it decided, and separates "the offending flows
went away because we throttled the device" (containment effect) from "the device
behaved normally again" (behaviour recovery), which the README discusses as a
limitation of containment-effect evidence.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import VerificationConfig  # noqa: E402


@dataclass
class VerificationRequest:
    """One open post-containment measurement."""

    request_id: str
    device_id: str
    action: str
    level_index: int
    start_step: int
    deadline_step: int
    score_before: float
    indicators_before: Dict[str, float]
    allowance: float
    scores: List[float] = field(default_factory=list)
    indicators_seen: List[Dict[str, float]] = field(default_factory=list)

    def steps_observed(self) -> int:
        return len(self.scores)


@dataclass
class VerificationResult:
    """Outcome of one verification window."""

    request_id: str
    device_id: str
    action: str
    level_index: int
    outcome: str
    start_step: int
    end_step: int
    steps_observed: int
    score_before: float
    score_after: float
    score_reduction: float
    indicator_reduction: float
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id, "device_id": self.device_id, "action": self.action,
            "level_index": self.level_index, "outcome": self.outcome,
            "start_step": self.start_step, "end_step": self.end_step,
            "steps_observed": self.steps_observed,
            "score_before": round(self.score_before, 6),
            "score_after": round(self.score_after, 6),
            "score_reduction": round(self.score_reduction, 6),
            "indicator_reduction": round(self.indicator_reduction, 6),
            "reason": self.reason,
        }


class ResponseVerifier:
    """Collects post-action evidence and classifies containment effectiveness."""

    def __init__(self, config: Optional[VerificationConfig] = None):
        self.config = config or VerificationConfig()
        self.open_requests: Dict[str, VerificationRequest] = {}
        self.results: List[VerificationResult] = []
        self._counter: int = 0

    # ------------------------------------------------------------------ #
    # Request lifecycle
    # ------------------------------------------------------------------ #
    def request(
        self,
        step: int,
        device_id: str,
        action: str,
        level_index: int,
        score_before: float,
        indicators: Dict[str, float],
        allowance: float,
    ) -> str:
        """Open a verification window for a containment action."""
        self._counter += 1
        request_id = f"v{self._counter:05d}"
        window = max(1, int(self.config.observation_window))
        self.open_requests[request_id] = VerificationRequest(
            request_id=request_id, device_id=device_id, action=action, level_index=level_index,
            start_step=int(step), deadline_step=int(step) + window,
            score_before=float(score_before),
            indicators_before=dict(indicators or {}), allowance=float(allowance),
        )
        return request_id

    def observe(
        self,
        step: int,
        device_id: str,
        score: float,
        indicators: Optional[Dict[str, float]] = None,
    ) -> Optional[VerificationResult]:
        """
        Feed one post-action observation to the open window of ``device_id``.

        Returns a :class:`VerificationResult` when the window closes, else None.
        """
        request_id = self.open_for(device_id)
        if request_id is None:
            return None
        request = self.open_requests[request_id]
        request.scores.append(float(score))
        if indicators:
            request.indicators_seen.append(dict(indicators))
        # The window covers exactly ``observation_window`` post-action steps
        # (start_step+1 ... start_step+window), so it closes on the last of them.
        if int(step) >= request.deadline_step:
            return self._close(request, end_step=int(step))
        return None

    def close_as_inconclusive(self, device_id: str, step: int, reason: str = "superseded") -> Optional[VerificationResult]:
        """Abandon an open window (a new action superseded it, or the run ended)."""
        request_id = self.open_for(device_id)
        if request_id is None:
            return None
        request = self.open_requests[request_id]
        return self._close(request, end_step=int(step), forced="INCONCLUSIVE", forced_reason=reason)

    def close_all_inconclusive(self, step: int) -> List[VerificationResult]:
        out: List[VerificationResult] = []
        for device_id in list({r.device_id for r in self.open_requests.values()}):
            result = self.close_as_inconclusive(device_id, step, reason="run_ended")
            if result is not None:
                out.append(result)
        return out

    # ------------------------------------------------------------------ #
    # Decision
    # ------------------------------------------------------------------ #
    def _close(
        self,
        request: VerificationRequest,
        end_step: int,
        forced: Optional[str] = None,
        forced_reason: str = "",
    ) -> VerificationResult:
        cfg = self.config
        steps_observed = request.steps_observed()
        score_after = float(np.mean(request.scores)) if request.scores else float("nan")
        score_reduction = (
            float((request.score_before - score_after) / max(abs(request.score_before), 1e-9))
            if request.scores else float("nan")
        )
        indicator_reduction = self._indicator_reduction(request)

        if forced is not None:
            outcome, reason = forced, forced_reason or "forced"
        elif steps_observed < int(cfg.observation_window):
            outcome = cfg.outcome_inconclusive
            reason = f"insufficient_evidence({steps_observed}/{cfg.observation_window} steps)"
        elif score_after <= cfg.success_absolute_score:
            outcome = cfg.outcome_success
            reason = f"score_below_absolute({score_after:.3f}<={cfg.success_absolute_score})"
        elif indicator_reduction >= cfg.success_score_reduction:
            outcome = cfg.outcome_success
            reason = f"offending_indicators_reduced({indicator_reduction:.2f})"
        elif score_reduction >= cfg.success_score_reduction:
            outcome = cfg.outcome_success
            reason = f"score_reduced({score_reduction:.2f})"
        else:
            outcome = cfg.outcome_failure
            reason = (f"no_effect(score_delta={score_reduction:.2f},"
                      f"indicator_delta={indicator_reduction:.2f})")

        result = VerificationResult(
            request_id=request.request_id, device_id=request.device_id, action=request.action,
            level_index=request.level_index, outcome=outcome, start_step=request.start_step,
            end_step=int(end_step), steps_observed=steps_observed,
            score_before=request.score_before, score_after=score_after,
            score_reduction=score_reduction, indicator_reduction=indicator_reduction,
            reason=reason,
        )
        self.results.append(result)
        self.open_requests.pop(request.request_id, None)
        return result

    @staticmethod
    def _indicator_reduction(request: VerificationRequest) -> float:
        """
        Mean relative reduction of the offending indicators that triggered the alarm.

        Indicators with a non-positive baseline value are skipped (no meaningful
        relative change can be computed). Zero denominators are handled explicitly
        and a reduced-but-not-vanished signal still counts proportionally.
        """
        if not request.indicators_before or not request.indicators_seen:
            return 0.0
        after_mean: Dict[str, float] = {}
        for name in request.indicators_before:
            values = [obs.get(name, 0.0) for obs in request.indicators_seen if name in obs]
            after_mean[name] = float(np.mean(values)) if values else 0.0
        reductions: List[float] = []
        for name, before in request.indicators_before.items():
            if before <= 1e-9:
                continue
            after = after_mean.get(name, before)
            reductions.append(float(np.clip((before - after) / before, -1.0, 1.0)))
        if not reductions:
            return 0.0
        return float(np.clip(np.mean(reductions), -1.0, 1.0))

    # ------------------------------------------------------------------ #
    # Introspection
    # ------------------------------------------------------------------ #
    def open_for(self, device_id: str) -> Optional[str]:
        for request_id, request in self.open_requests.items():
            if request.device_id == device_id:
                return request_id
        return None

    def results_frame(self) -> pd.DataFrame:
        if not self.results:
            return pd.DataFrame(columns=[
                "request_id", "device_id", "action", "level_index", "outcome", "start_step",
                "end_step", "steps_observed", "score_before", "score_after", "score_reduction",
                "indicator_reduction", "reason",
            ])
        return pd.DataFrame([r.to_dict() for r in self.results])

    def outcome_counts(self) -> Dict[str, int]:
        counts = {self.config.outcome_success: 0, self.config.outcome_failure: 0,
                  self.config.outcome_inconclusive: 0}
        for result in self.results:
            counts[result.outcome] = counts.get(result.outcome, 0) + 1
        return counts

    def status(self) -> Dict[str, Any]:
        return {"verifications": len(self.results), "outcome_counts": self.outcome_counts(),
                "open_windows": len(self.open_requests)}


if __name__ == "__main__":  # pragma: no cover - manual smoke test
    verifier = ResponseVerifier()
    rid = verifier.request(10, "dev_00", "RATE_LIMIT", 1, 0.8, {"pkt_rate_ratio": 7.0}, 0.35)
    print("opened", rid, "window", verifier.config.observation_window, "steps")
    for step in range(11, 11 + verifier.config.observation_window):
        result = verifier.observe(step, "dev_00", 0.4, {"pkt_rate_ratio": 2.4})
        if result:
            print(result.to_dict())
    print("counts:", verifier.outcome_counts())
