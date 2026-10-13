"""
Threat assessment (component E of the architecture).

Four *separate* concepts are kept apart on purpose, because conflating them is the
most common source of bogus results in this kind of prototype:

* **anomaly score** - the detector output in [0, 1] (unsupervised, no labels),
* **estimated confidence** - how much the detector's two scorers agree and how far
  the score sits beyond the threshold (still label-free),
* **severity** - a 0..100 operational risk score that additionally uses asset
  criticality and network context (used by the containment policy),
* **ground-truth label** - used *only* in ``metrics.py`` for offline evaluation.

Severity formula
----------------
``severity = 100 * (w_mag*magnitude + w_conf*confidence + w_crit*criticality
                    + w_ctx*context) / (w_mag + w_conf + w_crit + w_ctx)``

with every factor normalised to [0, 1]:

* ``magnitude``: how far the fused score exceeds the adaptive threshold,
  saturating at ``SeverityConfig.magnitude_saturation`` x threshold. This means
  severity is threshold-relative, so a device-specific baseline that is naturally
  noisy cannot inflate risk.
* ``confidence``: ``0.5 * agreement + 0.5 * magnitude`` where
  ``agreement = 1 - |stat_score - if_score|`` (the transparent scorer and the
  IsolationForest agree) -> 1 means both scorers tell the same story.
* ``criticality``: asset importance mapped through
  ``SeverityConfig.criticality_values`` (low .25, medium .5, high .75, critical 1.0).
  Configurable per device by an operator; **not** a detector input.
* ``context``: blast-radius proxy = ``0.6 * anomalous_share + 0.4 * traffic_share``,
  where ``anomalous_share`` is the fraction of the fleet currently anomalous and
  ``traffic_share`` is this device's share of the fleet's bytes per second.
  Rationale: an event inside a spreading incident is worse than an isolated one,
  and a high-volume device has a larger potential impact.

All weights live in ``config.SeverityConfig`` so the scoring is fully configurable;
weights are renormalised, so they need not sum to 1.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import SeverityConfig  # noqa: E402

EPS: float = 1e-9


def _finite(value: Any, default: float = 0.0) -> float:
    """
    Coerce an input to a finite float.

    Defensive on purpose: if a broken observation produced a NaN/Inf score, every
    comparison against it is False, which in the containment policy would fall
    through to the *most severe* branch. Sanitising here keeps a corrupt input from
    silently escalating to quarantine.
    """
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(default)
    return number if np.isfinite(number) else float(default)


@dataclass
class ThreatAssessment:
    """Severity plus the exact factor decomposition that produced it."""

    step: int
    device_id: str
    anomaly_score: float
    threshold: float
    magnitude: float
    confidence: float
    criticality: float
    context: float
    severity: float
    criticality_label: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step, "device_id": self.device_id,
            "anomaly_score": round(self.anomaly_score, 6), "threshold": round(self.threshold, 6),
            "magnitude": round(self.magnitude, 6), "confidence": round(self.confidence, 6),
            "criticality": round(self.criticality, 6), "context": round(self.context, 6),
            "severity": round(self.severity, 4), "criticality_label": self.criticality_label,
        }


def severity_band(severity: float, config: Optional[SeverityConfig] = None) -> str:
    """Human-readable band used by the dashboard (purely presentational)."""
    if severity <= 0.0:
        return "none"
    if severity < 25.0:
        return "low"
    if severity < 50.0:
        return "moderate"
    if severity < 75.0:
        return "high"
    return "critical"


class ThreatAssessor:
    """
    Turns detector output plus asset/context information into a severity score.

    Parameters
    ----------
    config:
        :class:`config.SeverityConfig`.
    device_criticality:
        Mapping ``device_id -> criticality label`` ("low" ... "critical").
    """

    def __init__(
        self,
        config: Optional[SeverityConfig] = None,
        device_criticality: Optional[Dict[str, str]] = None,
    ):
        self.config = config or SeverityConfig()
        self.device_criticality = dict(device_criticality or {})

    # ------------------------------------------------------------------ #
    # Factors
    # ------------------------------------------------------------------ #
    def magnitude_factor(self, score: float, threshold: float) -> float:
        """How far the score exceeds the threshold, saturating at the config ratio."""
        threshold = max(_finite(threshold, 0.0), EPS)
        excess = max(0.0, _finite(score, 0.0) - threshold)
        span = max(threshold * (self.config.magnitude_saturation - 1.0), EPS)
        return float(np.clip(excess / span, 0.0, 1.0))

    def confidence_factor(self, stat_score: float, if_score: float, magnitude: float) -> float:
        """Scorer agreement (0..1) blended with the threshold margin."""
        agreement = float(np.clip(
            1.0 - abs(_finite(stat_score, 0.0) - _finite(if_score, 0.0)), 0.0, 1.0))
        return float(np.clip(0.5 * agreement + 0.5 * _finite(magnitude, 0.0), 0.0, 1.0))

    def criticality_factor(self, device_id: str) -> float:
        label = self.device_criticality.get(device_id, "medium")
        return float(self.config.criticality_values.get(label, 0.5))

    def context_factor(self, anomalous_share: float, traffic_share: float) -> float:
        return float(np.clip(0.6 * _finite(anomalous_share, 0.0)
                             + 0.4 * _finite(traffic_share, 0.0), 0.0, 1.0))

    # ------------------------------------------------------------------ #
    # Assessment
    # ------------------------------------------------------------------ #
    def assess(
        self,
        step: int,
        device_id: str,
        score: float,
        threshold: float,
        stat_score: float,
        if_score: float,
        anomalous_share: float = 0.0,
        traffic_share: float = 0.0,
    ) -> ThreatAssessment:
        cfg = self.config
        magnitude = self.magnitude_factor(score, threshold)
        confidence = self.confidence_factor(stat_score, if_score, magnitude)
        criticality = self.criticality_factor(device_id)
        context = self.context_factor(anomalous_share, traffic_share)
        total_weight = max(cfg.w_magnitude + cfg.w_confidence + cfg.w_criticality
                           + cfg.w_network_context, EPS)
        severity = 100.0 * (
            cfg.w_magnitude * magnitude + cfg.w_confidence * confidence
            + cfg.w_criticality * criticality + cfg.w_network_context * context
        ) / total_weight
        # The reported severity is always finite inside [severity_min, severity_max],
        # even if the incoming score was corrupt (see _finite).
        severity = float(np.clip(_finite(severity, cfg.severity_min),
                                 cfg.severity_min, cfg.severity_max))
        return ThreatAssessment(
            step=int(step), device_id=device_id, anomaly_score=float(score),
            threshold=float(threshold), magnitude=magnitude, confidence=confidence,
            criticality=criticality, context=context, severity=severity,
            criticality_label=self.device_criticality.get(device_id, "medium"),
        )

    def assess_frame(self, rows: Iterable[ThreatAssessment]) -> pd.DataFrame:
        return pd.DataFrame([a.to_dict() for a in rows])


if __name__ == "__main__":  # pragma: no cover - manual smoke test
    assessor = ThreatAssessor(device_criticality={"dev_00": "critical", "dev_01": "low"})
    for dev, score in (("dev_00", 0.45), ("dev_00", 0.95), ("dev_01", 0.95), ("dev_01", 0.30)):
        a = assessor.assess(step=10, device_id=dev, score=score, threshold=0.40,
                            stat_score=min(1.0, score + 0.05), if_score=score,
                            anomalous_share=0.2, traffic_share=0.4)
        print(a.to_dict(), severity_band(a.severity))
