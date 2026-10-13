"""
Edge anomaly detection (component C of the architecture).

Two transparent, lightweight scorers are fused into one anomaly score:

1. ``StatisticalScorer`` - per-device robust z-scores from the benign
   calibration window (median and MAD), squashed into [0, 1) with
   ``z / (z + z_scale)``. Cheap, explainable, and it produces the per-feature
   *contributions* that the verifier and the dashboard reuse.
2. ``IsolationForestScorer`` - scikit-learn IsolationForest trained **on benign
   calibration data only**, with its raw decision function converted to the same
   [0, 1) scale with the identical robust rule (median/MAD of the calibration
   decision values). This keeps the fused score interpretable instead of relying
   on an opaque 0..1 output.

fused_score = (w_stat * statistical_score + w_if * isolation_forest_score) / (w_stat + w_if)

Thresholds
----------
* Static mode (``adaptive=False``, experiment system S1): the threshold is
  calibrated once from the benign window with a robust rule
  ``max(q95, median + k * 1.4826*MAD)`` computed per device, aggregated with a
  median across devices, times a small margin. It is then **frozen forever**.
* Adaptive mode (systems S2/S3): the threshold starts at the same calibration
  value and is tracked as an EWMA towards the 0.99 quantile of a rolling window
  of *benign* scores. Scores from frozen (suspicious) devices never enter that
  window, so a sustained attack cannot drag the threshold up (anti-poisoning
  layer 1; the behavioural baseline is protected separately in
  ``src/adaptive_learning.py``).

The detector never receives attack labels, device criticality, or ground truth.
Observations are scored in *batches of one simulation step*: this preserves
causality (a step's decisions only use data available at that step) while keeping
runtime low, because scikit-learn's per-row inference overhead dominates any
real computation for such tiny models.
"""

from __future__ import annotations

import sys
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import FEATURE_NAMES, DetectorConfig  # noqa: E402

EPS: float = 1e-9
_FEATURE_LIST: Tuple[str, ...] = tuple(FEATURE_NAMES)
_PROTOCOL_IDX: int = _FEATURE_LIST.index("protocol_shift")
#: Features whose raw value is a count of rare events rather than a continuous
#: measurement (used by the qualitative-indicator gates in adaptive learning).
QUALITATIVE_FEATURES: Tuple[str, ...] = ("failed_auth_rate", "protocol_shift")
#: Features with a natural volume interpretation.
VOLUME_FEATURES: Tuple[str, ...] = ("pkt_rate_ratio", "bytes_rate_ratio")


def _squash(z: np.ndarray | float, scale: float) -> np.ndarray:
    z_arr = np.abs(np.asarray(z, dtype=float))
    return z_arr / (z_arr + max(scale, EPS))


@dataclass
class ScoreResult:
    """Everything the closed loop needs to know about one scored observation."""

    step: int
    device_id: str
    score: float
    stat_score: float
    if_score: float
    threshold: float
    is_anomalous: bool
    contributions: Dict[str, float] = field(default_factory=dict)
    frozen: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step, "device_id": self.device_id,
            "anomaly_score": round(self.score, 6),
            "stat_score": round(self.stat_score, 6),
            "if_score": round(self.if_score, 6),
            "threshold": round(self.threshold, 6),
            "is_anomalous": int(self.is_anomalous),
            "frozen": int(self.frozen),
        }


class StatisticalScorer:
    """Per-device robust z-score scorer with per-feature contributions."""

    def __init__(self, config: Optional[DetectorConfig] = None):
        self.config = config or DetectorConfig()
        self.median_: Dict[str, np.ndarray] = {}
        self.scale_: Dict[str, np.ndarray] = {}
        self._index: Dict[str, int] = {}
        self._med_matrix: Optional[np.ndarray] = None
        self._scale_matrix: Optional[np.ndarray] = None
        self._global_med = np.zeros(len(FEATURE_NAMES), dtype=float)
        self._global_scale = np.ones(len(FEATURE_NAMES), dtype=float)

    def fit(self, X: np.ndarray, device_ids: Sequence[str]) -> "StatisticalScorer":
        df = pd.DataFrame(np.asarray(X, dtype=float), columns=list(FEATURE_NAMES))
        df["device_id"] = list(device_ids)
        devices: List[str] = []
        meds: List[np.ndarray] = []
        scales: List[np.ndarray] = []
        for device_id, group in df.groupby("device_id", sort=True):
            values = group[list(FEATURE_NAMES)].to_numpy(dtype=float)
            med = np.median(values, axis=0)
            mad = np.median(np.abs(values - med), axis=0)
            scale = np.maximum(1.4826 * mad, self.config.stat_min_scale)
            devices.append(str(device_id))
            meds.append(med)
            scales.append(scale)
            self.median_[str(device_id)] = med
            self.scale_[str(device_id)] = scale
        self._index = {d: i for i, d in enumerate(devices)}
        self._med_matrix = np.vstack(meds) if meds else np.zeros((0, len(FEATURE_NAMES)))
        self._scale_matrix = np.vstack(scales) if scales else np.ones((0, len(FEATURE_NAMES)))
        if meds:
            self._global_med = np.median(self._med_matrix, axis=0)
            self._global_scale = np.maximum(np.median(self._scale_matrix, axis=0),
                                            self.config.stat_min_scale)
        return self

    def _params_for(self, device_id: str) -> Tuple[np.ndarray, np.ndarray]:
        idx = self._index.get(device_id)
        if idx is None or self._med_matrix is None:
            return self._global_med, self._global_scale
        return self._med_matrix[idx], self._scale_matrix[idx]

    def contributions(self, device_id: str, x: np.ndarray) -> np.ndarray:
        med, scale = self._params_for(device_id)
        z = np.abs(np.asarray(x, dtype=float) - med) / np.maximum(scale, EPS)
        contrib = _squash(z, self.config.z_scale)
        # protocol_shift is categorical (0/1/2): map directly (1 -> 0.5, 2 -> 1.0)
        contrib[_PROTOCOL_IDX] = float(min(1.0, 0.5 * abs(float(x[_PROTOCOL_IDX]))))
        return contrib

    def score(self, device_id: str, x: np.ndarray) -> Tuple[float, np.ndarray]:
        contrib = self.contributions(device_id, x)
        return self._aggregate(contrib), contrib

    def score_batch(
        self, X: np.ndarray, device_ids: Sequence[str]
    ) -> Tuple[np.ndarray, np.ndarray]:
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        meds = np.vstack([self._params_for(d)[0] for d in device_ids])
        scales = np.vstack([self._params_for(d)[1] for d in device_ids])
        z = np.abs(X - meds) / np.maximum(scales, EPS)
        contrib = _squash(z, self.config.z_scale)
        contrib[:, _PROTOCOL_IDX] = np.minimum(1.0, 0.5 * np.abs(X[:, _PROTOCOL_IDX]))
        return self._aggregate(contrib), contrib

    def _aggregate(self, contrib: np.ndarray) -> np.ndarray:
        cfg = self.config
        mx = np.max(contrib, axis=1)
        mean = np.mean(contrib, axis=1)
        return np.clip(cfg.stat_max_weight * mx + (1.0 - cfg.stat_max_weight) * mean, 0.0, 1.0)


class IsolationForestScorer:
    """
    IsolationForest wrapper with a transparent, robust anomaly conversion.

    ``z_if = (median(calibration decision values) - decision(x)) / (1.4826*MAD)``
    is positive exactly when ``x`` looks more isolated than typical benign data,
    and is squashed with the same ``z_scale`` as the statistical scorer.
    """

    def __init__(self, config: Optional[DetectorConfig] = None, seed: int = 0):
        self.config = config or DetectorConfig()
        self.seed = int(seed)
        self.model: Optional[Dict[str, np.ndarray]] = None
        self._med_df: float = 0.0
        self._scale_df: float = 1.0
        self.trained: bool = False

    def fit(self, X: np.ndarray) -> "IsolationForestScorer":
        """Fit a lightweight robust-distance substitute for IsolationForest.

        The dashboard runs on resource-constrained devices, so keeping this
        scorer NumPy-only avoids shipping scikit-learn's large compiled runtime
        while preserving the calibrated, monotonic anomaly-score contract.
        """
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[0] < 10:
            self.model, self.trained = None, False
            return self
        center = np.median(X, axis=0)
        scale = np.maximum(1.4826 * np.median(np.abs(X - center), axis=0), EPS)
        distances = np.mean(np.abs(X - center) / scale, axis=1)
        med = float(np.median(distances))
        mad = float(np.median(np.abs(distances - med)))
        self.model = {"center": center, "scale": scale}
        self._med_df = med
        self._scale_df = max(1.4826 * mad, EPS)
        self.trained = True
        return self

    def score_batch(self, X: np.ndarray) -> np.ndarray:
        if not self.trained or self.model is None:
            return np.zeros(np.asarray(X).shape[0] if np.asarray(X).ndim > 1 else 1, dtype=float)
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        distances = np.mean(
            np.abs(X - self.model["center"]) / self.model["scale"], axis=1
        )
        z = (distances - self._med_df) / self._scale_df
        return _squash(z, self.config.z_scale)

    def score(self, x: np.ndarray) -> float:
        return float(self.score_batch(np.asarray(x, dtype=float).reshape(1, -1))[0])


class AnomalyDetector:
    """
    Fused edge anomaly detector with static or adaptive thresholding.

    Parameters
    ----------
    config:
        :class:`config.DetectorConfig`.
    seed:
        Seed handed to IsolationForest (reproducibility).
    adaptive:
        ``True``  -> adaptive threshold + freeze logic (systems S2/S3).
        ``False`` -> static, calibration-only threshold (system S1).
    """

    def __init__(
        self,
        config: Optional[DetectorConfig] = None,
        seed: int = 0,
        adaptive: bool = True,
    ):
        self.config = config or DetectorConfig()
        self.seed = int(seed)
        self.adaptive = bool(adaptive)
        self.stat = StatisticalScorer(self.config)
        self.iforest = IsolationForestScorer(self.config, seed=self.seed)
        #: Per-device thresholds: an observation is compared against the benign
        #: score distribution *of its own device*, which is what keeps the
        #: false-positive rate low across a heterogeneous fleet.
        self.thresholds: Dict[str, float] = {}
        self._benign_windows: Dict[str, Deque[float]] = defaultdict(
            lambda: deque(maxlen=int(self.config.threshold_window))
        )
        #: Fleet-level median of the per-device thresholds (reporting only).
        self.threshold: float = float(self.config.static_threshold)
        self.calibration_threshold: float = float(self.config.static_threshold)
        self.frozen_devices: set = set()
        self._benign_streak: Dict[str, int] = defaultdict(int)
        self.n_calibration_scores: int = 0
        self.threshold_history: List[Dict[str, Any]] = []
        self.freeze_events: List[Dict[str, Any]] = []
        self.unfreeze_events: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------ #
    # Calibration (benign data only)
    # ------------------------------------------------------------------ #
    def calibrate(self, features: pd.DataFrame, device_ids: Sequence[str]) -> float:
        """
        Fit both scorers and fix the calibration threshold.

        ``features`` must contain only ``FEATURE_NAMES`` columns: labels are not
        accepted here by design.
        """
        X = self._as_matrix(features)
        self.stat.fit(X, device_ids)
        self.iforest.fit(X)
        stat_scores, _ = self.stat.score_batch(X, device_ids)
        if_scores = self.iforest.score_batch(X)
        scores = self._fuse(stat_scores, if_scores)
        self.n_calibration_scores = int(scores.size)

        frame = pd.DataFrame({"device_id": list(device_ids), "score": scores})
        per_device: List[float] = []
        self.thresholds.clear()
        self._benign_windows.clear()
        fallback = float(np.quantile(scores, 0.99)) if scores.size else self.config.static_threshold
        for device_id, group in frame.groupby("device_id", sort=True):
            device_key = str(device_id)
            values = group["score"].to_numpy(dtype=float)
            if values.size >= int(self.config.min_calibration_rows_per_device):
                med = float(np.median(values))
                mad = float(np.median(np.abs(values - med)))
                empirical = float(np.quantile(values, self.config.threshold_quantile))
                robust_floor = med + self.config.threshold_robust_k * 1.4826 * mad
                value = float(max(empirical, med, min(robust_floor, empirical)))
            else:
                value = fallback
            value = float(np.clip(value * self.config.threshold_margin,
                                  self.config.threshold_min, self.config.threshold_max))
            if not self.config.static_threshold_auto:
                value = float(self.config.static_threshold)
            self.thresholds[device_key] = value
            per_device.append(value)
            # Seed the adaptive window with that device's most anomalous benign
            # calibration scores, so the 0.99-quantile target starts realistically.
            window = self._benign_windows[device_key]
            window.clear()
            for score_value in np.sort(values)[::-1][: int(self.config.threshold_window)]:
                window.append(float(score_value))
        self.calibration_threshold = float(np.median(per_device)) if per_device else fallback
        self.threshold = float(self.calibration_threshold)
        self.threshold_history = [{"step": 0, "device_id": "*", "threshold": self.threshold}]
        return self.threshold

    # ------------------------------------------------------------------ #
    # Online scoring
    # ------------------------------------------------------------------ #
    def process(self, step: int, device_id: str, features: np.ndarray) -> ScoreResult:
        """Score a single observation (convenience wrapper around the batch API)."""
        return self.process_batch(step, [device_id], np.asarray(features, dtype=float).reshape(1, -1))[0]

    def process_batch(
        self, step: int, device_ids: Sequence[str], features: np.ndarray
    ) -> List[ScoreResult]:
        """
        Score all observations of one simulation step and advance detector state.

        Devices in ``device_ids`` are the only inputs; no labels are used.
        """
        X = np.asarray(features, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        stat_scores, contrib_matrix = self.stat.score_batch(X, device_ids)
        if_scores = self.iforest.score_batch(X)
        scores = self._fuse(stat_scores, if_scores)

        results: List[ScoreResult] = []
        for i, device_id in enumerate(device_ids):
            score = float(scores[i])
            device_threshold = float(self.thresholds.get(device_id, self.threshold))
            is_anomalous = bool(score > device_threshold)
            frozen_before = device_id in self.frozen_devices

            if self.adaptive and is_anomalous and not frozen_before:
                # Freezing only makes sense where a baseline/threshold is learned:
                # the static system (S1) has nothing to protect from poisoning.
                self.frozen_devices.add(device_id)
                self.freeze_events.append(
                    {"step": int(step), "device_id": device_id,
                     "score": round(score, 6), "threshold": round(device_threshold, 6)}
                )

            if self.adaptive:
                if not frozen_before and not is_anomalous:
                    # Anti-poisoning layer 1: only benign observations of
                    # non-suspicious devices may inform the adaptive threshold.
                    self._benign_windows[device_id].append(score)
                    self._adapt_threshold(device_id, step)
                if frozen_before:
                    if score <= device_threshold * self.config.unfreeze_score_fraction:
                        self._benign_streak[device_id] += 1
                        if self._benign_streak[device_id] >= int(self.config.unfreeze_patience):
                            self.frozen_devices.discard(device_id)
                            self._benign_streak[device_id] = 0
                            self.unfreeze_events.append(
                                {"step": int(step), "device_id": device_id, "score": round(score, 6)}
                            )
                    else:
                        self._benign_streak[device_id] = 0

            contributions = {name: float(contrib_matrix[i, j]) for j, name in enumerate(FEATURE_NAMES)}
            results.append(
                ScoreResult(
                    step=int(step), device_id=device_id, score=score,
                    stat_score=float(stat_scores[i]), if_score=float(if_scores[i]),
                    threshold=device_threshold, is_anomalous=is_anomalous,
                    contributions=contributions, frozen=frozen_before,
                )
            )
        return results

    def _fuse(self, stat_score: np.ndarray, if_score: np.ndarray) -> np.ndarray:
        cfg = self.config
        total = max(cfg.stat_weight + cfg.if_weight, EPS)
        return np.clip((cfg.stat_weight * np.asarray(stat_score) + cfg.if_weight * np.asarray(if_score)) / total,
                       0.0, 1.0)

    def _adapt_threshold(self, device_id: str, step: int) -> None:
        """EWMA tracking of this device's benign 0.99 quantile (adaptive mode)."""
        window = self._benign_windows[device_id]
        if len(window) < int(self.config.threshold_min_samples):
            return
        target = float(np.quantile(np.fromiter(window, dtype=float),
                                  self.config.threshold_window_quantile))
        current = float(self.thresholds.get(device_id, target))
        new_value = (1.0 - self.config.threshold_ewma_alpha) * current + \
            self.config.threshold_ewma_alpha * target
        self.thresholds[device_id] = float(np.clip(new_value, self.config.threshold_min,
                                                   self.config.threshold_max))
        self.threshold = float(np.median(list(self.thresholds.values()))) if self.thresholds \
            else new_value
        self.threshold_history.append(
            {"step": int(step), "device_id": device_id,
             "threshold": float(self.thresholds[device_id])}
        )

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _as_matrix(features: pd.DataFrame) -> np.ndarray:
        if isinstance(features, pd.DataFrame):
            missing = [c for c in FEATURE_NAMES if c not in features.columns]
            if missing:
                raise ValueError(f"calibration frame is missing feature columns: {missing}")
            return features[list(FEATURE_NAMES)].to_numpy(dtype=float)
        return np.asarray(features, dtype=float)

    def top_contributions(self, result: ScoreResult, k: int = 3) -> List[Tuple[str, float]]:
        items = sorted(result.contributions.items(), key=lambda kv: kv[1], reverse=True)
        return [(name, value) for name, value in items[:k]]

    def threshold_frame(self) -> pd.DataFrame:
        if not self.threshold_history:
            return pd.DataFrame(columns=["step", "device_id", "threshold"])
        return pd.DataFrame(self.threshold_history, columns=["step", "device_id", "threshold"])

    def freeze_frame(self) -> pd.DataFrame:
        if not self.freeze_events:
            return pd.DataFrame(columns=["step", "device_id", "score", "threshold"])
        return pd.DataFrame(self.freeze_events)

    def status(self) -> Dict[str, Any]:
        return {
            "mode": "adaptive" if self.adaptive else "static",
            "calibration_threshold_median": round(self.calibration_threshold, 6),
            "current_threshold_median": round(self.threshold, 6),
            "per_device_thresholds": {k: round(v, 6) for k, v in sorted(self.thresholds.items())},
            "n_calibration_scores": self.n_calibration_scores,
            "frozen_devices": sorted(self.frozen_devices),
            "n_freezes": len(self.freeze_events),
            "n_unfreezes": len(self.unfreeze_events),
        }


if __name__ == "__main__":  # pragma: no cover - manual smoke test
    from config import ExperimentConfig
    from src.adaptive_learning import AdaptiveBaselineManager
    from src.feature_extractor import FeatureExtractor
    from src.traffic_simulator import TrafficSimulator

    cfg = ExperimentConfig()
    sim = TrafficSimulator(cfg)
    tel, attacks = sim.generate()
    mgr = AdaptiveBaselineManager(cfg.learning)
    mgr.fit(tel, cfg.calibration_steps)
    fe = FeatureExtractor(
        mgr,
        {d.device_id: d.allowed_protocols for d in sim.devices},
        {d.device_id: d.normal_protocol for d in sim.devices},
    )
    calib = tel[tel["step"] < cfg.calibration_steps]
    X = np.vstack([fe.extract(r).values for _, r in calib.iterrows()])
    ids = [str(r["device_id"]) for _, r in calib.iterrows()]
    det = AnomalyDetector(cfg.detector, seed=cfg.seed, adaptive=True)
    print("calibration threshold:", round(det.calibrate(pd.DataFrame(X, columns=list(FEATURE_NAMES)), ids), 4))
    for _, row in tel[tel["ground_truth"] == 1].head(3).iterrows():
        res = det.process(int(row["step"]), str(row["device_id"]), fe.extract(row).values)
        print(row["device_id"], row["attack_type"], res.to_dict())
