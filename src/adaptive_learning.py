"""
Closed-loop adaptive learning (component H): device baseline maintenance with
anti-poisoning safeguards.

Three mechanisms protect the behavioural baseline.

1. **Suspicion freeze.** Any observation whose fused anomaly score exceeds the
   adaptive threshold freezes the device baseline: no statistic may move. The
   device must produce ``freeze_patience`` consecutive benign observations (or be
   re-confirmed benign by the drift-evidence pathway below) before it is released.
2. **Learner-side gate.** Even for unfrozen devices, updates are only applied when
   the detector judged the observation benign, and updates are refused entirely
   while the gateway's own containment action is suppressing the device's traffic
   (``blocked_containment``). Learning from data we ourselves manipulated would
   bake the suppression into the baseline.
3. **Drift-evidence pathway.** A *frozen* device is not released by time alone,
   because a slow attack could simply wait. It is released only when a
   configurable set of gates says the deviation is consistent with a legitimate
   behaviour change: volume ratios stay inside a plausible band, and every
   qualitative indicator (failed authentications, protocol shift, connection
   pattern, destination diversity, CPU/memory movement) is quiet. When the gates
   hold for ``drift_patience`` consecutive steps, the freeze is released *for the
   volume features only*, at a bounded learning rate. If any qualitative
   indicator later fires, the baseline freezes again immediately.

Every accepted update *and every refusal* is logged with step, device, mode,
learning rate, score and threshold, so the dashboard and the results tables can
show exactly what the baseline learned and how often the guard intervened.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import LearningConfig  # noqa: E402

#: Raw telemetry keys tracked by a device baseline.
TRACKED_KEYS: Tuple[str, ...] = (
    "packet_rate", "bytes_per_sec", "conn_count", "dest_diversity", "cpu_usage", "memory_usage",
)
#: Keys that may move under the drift-evidence pathway (volume only).
DRIFT_RELEASABLE_KEYS: Tuple[str, ...] = ("packet_rate", "bytes_per_sec")


@dataclass
class BaselineRecord:
    """Location/scale statistics of one device's learned normal behaviour."""

    device_id: str
    loc: Dict[str, float]
    scale: Dict[str, float]
    n_updates: int = 0
    frozen: bool = False
    benign_streak: int = 0
    drift_streak: int = 0
    drift_released: bool = False
    last_update_step: int = -1

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "device_id": self.device_id, "n_updates": self.n_updates,
            "frozen": int(self.frozen), "drift_released": int(self.drift_released),
            "last_update_step": self.last_update_step,
        }
        d.update({f"loc_{k}": round(v, 4) for k, v in self.loc.items()})
        d.update({f"scale_{k}": round(v, 4) for k, v in self.scale.items()})
        return d


@dataclass
class UpdateEvent:
    """One logged baseline update (or refusal), kept for auditability."""

    step: int
    device_id: str
    mode: str            # clean | drift | drift_release | blocked_frozen |
    #                      blocked_anomalous | blocked_containment
    learning_rate: float
    score: float
    threshold: float
    changed: Dict[str, float] = field(default_factory=dict)
    #: For ``drift_release`` events: the gate values that justified the release.
    context: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step, "device_id": self.device_id, "mode": self.mode,
            "learning_rate": round(self.learning_rate, 5),
            "score": round(self.score, 5) if math.isfinite(self.score) else None,
            "threshold": round(self.threshold, 5) if math.isfinite(self.threshold) else None,
            "changed": ";".join(f"{k}:{v:.4f}" for k, v in self.changed.items()),
            "context": self.context,
        }


class AdaptiveBaselineManager:
    """
    Maintains per-device behavioural baselines and decides when they may adapt.

    Parameters
    ----------
    config:
        :class:`config.LearningConfig` instance.
    """

    def __init__(self, config: Optional[LearningConfig] = None):
        self.config = config or LearningConfig()
        self.baselines: Dict[str, BaselineRecord] = {}
        self.update_log: List[UpdateEvent] = []
        self.blocked_updates_frozen: int = 0
        self.blocked_updates_anomalous: int = 0
        self.blocked_updates_containment: int = 0
        self.drift_releases: int = 0

    # ------------------------------------------------------------------ #
    # Calibration
    # ------------------------------------------------------------------ #
    def fit(self, telemetry: pd.DataFrame, window_end: int) -> None:
        """
        Initialise baselines from the benign calibration window [0, window_end).

        Median and MAD are used, so a few generous benign bursts inside the
        calibration window cannot inflate the reference statistics the way a
        mean/standard-deviation pair would.
        """
        window = telemetry[telemetry["step"] < int(window_end)]
        for device_id, group in window.groupby("device_id", sort=True):
            loc, scale = {}, {}
            for key in TRACKED_KEYS:
                values = group[key].to_numpy(dtype=float)
                med = float(np.median(values)) if values.size else 0.0
                mad = float(np.median(np.abs(values - med))) if values.size else 0.0
                loc[key] = med
                scale[key] = self._floor_scale(1.4826 * mad, med)
            self.baselines[str(device_id)] = BaselineRecord(
                device_id=str(device_id), loc=loc, scale=scale
            )

    def _floor_scale(self, sigma: float, loc: float) -> float:
        cfg = self.config
        return float(max(sigma, cfg.min_scale_fraction * abs(loc), cfg.min_scale))

    # ------------------------------------------------------------------ #
    # Freeze / release
    # ------------------------------------------------------------------ #
    def freeze(self, device_id: str, step: int, reason: str = "anomalous") -> None:
        rec = self.baselines.get(device_id)
        if rec is None:
            return
        if not rec.frozen:
            self.update_log.append(
                UpdateEvent(step=int(step), device_id=device_id, mode="blocked_frozen",
                            learning_rate=0.0, score=float("nan"), threshold=float("nan"),
                            changed={"freeze": 1.0, "reason": float(len(reason))})
            )
        rec.frozen = True
        rec.drift_released = False
        rec.benign_streak = 0
        rec.drift_streak = 0

    def note_benign(self, device_id: str) -> bool:
        """
        Register a benign observation. Returns True when the device is released
        as a result (enough consecutive benign observations).
        """
        rec = self.baselines.get(device_id)
        if rec is None:
            return False
        rec.benign_streak += 1
        if rec.frozen and rec.benign_streak >= self.config.freeze_patience:
            rec.frozen = False
            rec.drift_released = False
            rec.benign_streak = 0
            return True
        return False

    def is_frozen(self, device_id: str) -> bool:
        rec = self.baselines.get(device_id)
        return bool(rec.frozen) if rec else False

    # ------------------------------------------------------------------ #
    # Drift-evidence gates
    # ------------------------------------------------------------------ #
    def drift_evidence(self, features: Optional[Dict[str, float]]) -> bool:
        """
        Do the current *features* look like a legitimate behaviour change?

        Uses only unlabeled feature magnitudes; no ground truth is involved.
        Gates (all configurable in ``config.LearningConfig``):

        * volume ratios inside ``[drift_min_volume_ratio, drift_max_volume_ratio]``
        * no failed authentication attempts (``drift_max_failed_auth``)
        * protocol unchanged when ``drift_require_normal_protocol``
        * ``|conn_count_z|`` and ``|dest_diversity_z|`` below ``drift_max_abs_z``
        * ``|cpu_delta|`` and ``|mem_delta|`` below ``drift_max_resource_delta``

        An attacker can in principle shape traffic to satisfy these gates; see the
        limitations section of the README.
        """
        if not features:
            return False
        cfg = self.config
        for key in ("pkt_rate_ratio", "bytes_rate_ratio"):
            value = float(features.get(key, 1.0))
            if not (cfg.drift_min_volume_ratio <= value <= cfg.drift_max_volume_ratio):
                return False
        if float(features.get("failed_auth_rate", 0.0)) > cfg.drift_max_failed_auth:
            return False
        if cfg.drift_require_normal_protocol and float(features.get("protocol_shift", 0.0)) > 0.0:
            return False
        for key in ("conn_count_z", "dest_diversity_z"):
            if abs(float(features.get(key, 0.0))) > cfg.drift_max_abs_z:
                return False
        for key in ("cpu_delta", "mem_delta"):
            if abs(float(features.get(key, 0.0))) > cfg.drift_max_resource_delta:
                return False
        return True

    def gate_context(self, features: Optional[Dict[str, float]]) -> str:
        """Compact audit string of the drift gates (stored with every release)."""
        if not features:
            return "no_features"
        return (
            f"pkt_ratio={float(features.get('pkt_rate_ratio', 0.0)):.2f};"
            f"bytes_ratio={float(features.get('bytes_rate_ratio', 0.0)):.2f};"
            f"failed_auth={float(features.get('failed_auth_rate', 0.0)):.2f};"
            f"protocol_shift={float(features.get('protocol_shift', 0.0)):.0f};"
            f"conns_z={float(features.get('conn_count_z', 0.0)):.2f};"
            f"dst_z={float(features.get('dest_diversity_z', 0.0)):.2f};"
            f"cpu_delta={float(features.get('cpu_delta', 0.0)):.3f};"
            f"mem_delta={float(features.get('mem_delta', 0.0)):.3f}"
        )

    # ------------------------------------------------------------------ #
    # Online update
    # ------------------------------------------------------------------ #
    def step(
        self,
        step: int,
        device_id: str,
        observation: Dict[str, float],
        score: float,
        threshold: float,
        is_anomalous: bool,
        features: Optional[Dict[str, float]] = None,
        traffic_allowance: float = 1.0,
    ) -> Optional[UpdateEvent]:
        """Apply, or refuse, one baseline update and log the decision."""
        rec = self.baselines.get(device_id)
        if rec is None:
            return None

        if not math.isfinite(score):
            score = 0.0
        if not math.isfinite(threshold) or threshold <= 0:
            threshold = 1.0

        # (2) never learn from telemetry that a heavy containment action distorted
        if float(traffic_allowance) < float(self.config.learning_min_allowance):
            self.blocked_updates_containment += 1
            event = UpdateEvent(step, device_id, "blocked_containment", 0.0, score, threshold,
                                {"allowance": float(traffic_allowance)})
            self.update_log.append(event)
            return event

        if is_anomalous:
            self.blocked_updates_anomalous += 1
            if self.drift_evidence(features):
                rec.drift_streak += 1
                if rec.drift_streak >= self.config.drift_patience:
                    # Enough evidence of a legitimate change: release the volume
                    # features for slow learning; qualitative features stay frozen.
                    rec.frozen = False
                    rec.drift_released = True
                    rec.benign_streak = 0
                    self.drift_releases += 1
                    lr = self.config.drift_release_learning_rate
                    changed = self._apply_update(
                        rec, observation, lr, keys=DRIFT_RELEASABLE_KEYS
                    )
                    rec.n_updates += 1
                    rec.last_update_step = int(step)
                    event = UpdateEvent(step, device_id, "drift_release", lr, score, threshold,
                                        changed, context=self.gate_context(features))
                    self.update_log.append(event)
                    return event
                event = UpdateEvent(step, device_id, "blocked_frozen", 0.0, score, threshold,
                                    {"drift_streak": float(rec.drift_streak)})
                self.update_log.append(event)
                return event
            rec.drift_streak = 0
            self.freeze(device_id, step)
            event = UpdateEvent(step, device_id, "blocked_anomalous", 0.0, score, threshold,
                                {"observation_rejected": 1.0},
                                context=self.gate_context(features))
            self.update_log.append(event)
            return event

        # --- anomalous evidence over: qualitative indicator re-freeze ----- #
        if rec.drift_released:
            rec.frozen = False
            rec.drift_released = False

        if rec.frozen:
            self.blocked_updates_frozen += 1
            self.note_benign(device_id)
            event = UpdateEvent(step, device_id, "blocked_frozen", 0.0, score, threshold)
            self.update_log.append(event)
            return event

        # Grey zone -> possible legitimate behaviour drift (reduced rate).
        drift_zone_floor = self.config.drift_zone_fraction * threshold
        if score >= drift_zone_floor:
            rec.drift_streak += 1
            mode, lr = "drift", self.config.learning_rate_drift
        else:
            rec.drift_streak = 0
            mode, lr = "clean", self.config.learning_rate_clean

        changed = self._apply_update(rec, observation, lr)
        rec.n_updates += 1
        rec.last_update_step = int(step)
        event = UpdateEvent(step, device_id, mode, lr, score, threshold, changed)
        self.update_log.append(event)
        return event

    def _apply_update(
        self,
        rec: BaselineRecord,
        observation: Dict[str, float],
        lr: float,
        keys: Sequence[str] = TRACKED_KEYS,
    ) -> Dict[str, float]:
        """EWMA update with per-step movement clamps (bounded, auditable drift)."""
        changed: Dict[str, float] = {}
        for key in keys:
            if key not in observation:
                continue
            x = float(observation[key])
            if not math.isfinite(x):
                continue
            loc = rec.loc.get(key, x)
            limit = self.config.max_single_update_fraction * max(abs(loc), self.config.min_scale)
            delta = float(np.clip(lr * (x - loc), -limit, limit))
            new_loc = loc + delta
            sigma = abs(x - new_loc)
            new_scale = (1.0 - lr) * rec.scale.get(key, sigma) + lr * sigma
            rec.loc[key] = new_loc
            rec.scale[key] = self._floor_scale(new_scale, new_loc)
            changed[key] = delta
        return changed

    # ------------------------------------------------------------------ #
    # Introspection
    # ------------------------------------------------------------------ #
    def snapshot(self) -> pd.DataFrame:
        return pd.DataFrame([rec.to_dict() for rec in self.baselines.values()])

    def update_log_frame(self) -> pd.DataFrame:
        if not self.update_log:
            return pd.DataFrame(
                columns=["step", "device_id", "mode", "learning_rate", "score", "threshold",
                         "changed", "context"]
            )
        return pd.DataFrame([e.to_dict() for e in self.update_log])

    def mode_counts(self) -> pd.DataFrame:
        log = self.update_log_frame()
        if log.empty:
            return pd.DataFrame(columns=["mode", "count"])
        counts = log["mode"].value_counts().rename_axis("mode").reset_index(name="count")
        return counts

    def baseline_of(self, device_id: str) -> Optional[BaselineRecord]:
        return self.baselines.get(device_id)

    def drift_devices(self) -> List[str]:
        """Devices whose baseline was adapted as legitimate behaviour change."""
        return sorted({e.device_id for e in self.update_log if e.mode in ("drift", "drift_release")})

    def status(self) -> Dict[str, Any]:
        return {
            "n_devices": len(self.baselines),
            "frozen_now": sorted(d for d, r in self.baselines.items() if r.frozen),
            "n_updates_accepted": sum(1 for e in self.update_log
                                      if e.mode in ("clean", "drift", "drift_release")),
            "blocked_updates_anomalous": self.blocked_updates_anomalous,
            "blocked_updates_frozen": self.blocked_updates_frozen,
            "blocked_updates_containment": self.blocked_updates_containment,
            "drift_releases": self.drift_releases,
        }


if __name__ == "__main__":  # pragma: no cover - manual smoke test
    from config import ExperimentConfig
    from src.traffic_simulator import TrafficSimulator

    cfg = ExperimentConfig(n_devices=6, n_steps=60, calibration_steps=30, warmup_steps=35)
    tel, _ = TrafficSimulator(cfg).generate()
    mgr = AdaptiveBaselineManager(cfg.learning)
    mgr.fit(tel, cfg.calibration_steps)
    print(mgr.snapshot().head().to_string(index=False))
