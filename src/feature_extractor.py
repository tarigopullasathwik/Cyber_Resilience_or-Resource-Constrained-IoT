"""
Edge monitoring, lightweight feature extraction and the simulated edge resource
model (component B of the architecture).

The feature extractor turns one raw telemetry record into the fixed-length
numeric vector declared in ``config.FEATURE_NAMES``. Only these eight features
(plus the device id, used to select the right baseline) ever reach the detector.
Criticality, attack type and ground-truth labels are deliberately *not* features.

The simulated edge gateway has a bounded CPU, memory and uplink budget. The
:class:`EdgeResourceModel` models that envelope so the experiments can report
*simulated* resource pressure; these are model estimates, clearly separated from
the *measured* host-machine timing reported by ``metrics.py``.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import STEP_SECONDS, EdgeResourceConfig, FEATURE_NAMES  # noqa: E402
from src.adaptive_learning import TRACKED_KEYS, AdaptiveBaselineManager  # noqa: E402

EPS: float = 1e-6


@dataclass
class FeatureVector:
    """One extracted feature vector with its provenance."""

    step: int
    device_id: str
    values: np.ndarray                    # shape (len(FEATURE_NAMES),)
    raw: Dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, float]:
        return {name: float(v) for name, v in zip(FEATURE_NAMES, self.values)}


class FeatureExtractor:
    """
    Stateless-ish transformer from raw telemetry to the edge feature vector.

    The device's *normal* reference is supplied by the
    :class:`~src.adaptive_learning.AdaptiveBaselineManager`, which is the single
    place where normal behaviour is learned and protected against poisoning.

    Feature definitions (all cheap scalar arithmetic, suitable for a gateway):

    ``pkt_rate_ratio``    packet rate / learned normal packet rate (1.0 = normal)
    ``bytes_rate_ratio``  bytes per second / learned normal bytes per second
    ``failed_auth_rate``  failed authentication attempts in this step (absolute
    count; normal is ~0 for most device classes)
    ``conn_count_z``      (connections - normal) / robust scale of normal
    ``dest_diversity_z``  (distinct destinations - normal) / robust scale
    ``cpu_delta``         (device CPU% - normal) / 100
    ``mem_delta``         (device memory% - normal) / 100
    ``protocol_shift``    0 = usual protocol, 1 = unusual but permitted protocol
    for this device class, 2 = protocol never seen for this device class
    """

    def __init__(
        self,
        baseline_manager: AdaptiveBaselineManager,
        allowed_protocols: Optional[Dict[str, Sequence[str]]] = None,
        normal_protocols: Optional[Dict[str, str]] = None,
    ):
        self.baselines = baseline_manager
        self.allowed_protocols = allowed_protocols or {}
        self.normal_protocols = normal_protocols or {}

    def extract(self, row: Any, step: Optional[int] = None) -> FeatureVector:
        """Extract one feature vector from a telemetry row (dict or Series)."""
        device_id = str(row["device_id"]) if not isinstance(row, dict) else str(row["device_id"])
        rec = self.baselines.baseline_of(device_id)
        if rec is None:  # unseen device -> neutral vector (monitoring only)
            return FeatureVector(
                step=int(row.get("step", step or 0)) if isinstance(row, dict) else int(row["step"]),
                device_id=device_id,
                values=np.zeros(len(FEATURE_NAMES), dtype=float),
                raw={},
            )

        loc, scale = rec.loc, rec.scale
        pkt = float(row["packet_rate"])
        byts = float(row["bytes_per_sec"])
        conns = float(row["conn_count"])
        dst = float(row["dest_diversity"])
        cpu = float(row["cpu_usage"])
        mem = float(row["memory_usage"])
        failed_auth = float(row["failed_auth_attempts"])
        proto = str(row["protocol"])

        values = np.array(
            [
                pkt / max(loc.get("packet_rate", pkt), EPS),
                byts / max(loc.get("bytes_per_sec", byts), EPS),
                failed_auth,
                (conns - loc.get("conn_count", conns)) / max(scale.get("conn_count", 1.0), EPS),
                (dst - loc.get("dest_diversity", dst)) / max(scale.get("dest_diversity", 1.0), EPS),
                (cpu - loc.get("cpu_usage", cpu)) / 100.0,
                (mem - loc.get("memory_usage", mem)) / 100.0,
                float(self._protocol_shift(device_id, proto)),
            ],
            dtype=float,
        )
        raw = {
            "packet_rate": pkt, "bytes_per_sec": byts, "conn_count": conns,
            "dest_diversity": dst, "cpu_usage": cpu, "memory_usage": mem,
            "failed_auth_attempts": failed_auth, "protocol": proto,
        }
        step_value = int(row["step"]) if not isinstance(row, dict) else int(row.get("step", step or 0))
        return FeatureVector(step=step_value, device_id=device_id, values=values, raw=raw)

    def _protocol_shift(self, device_id: str, proto: str) -> int:
        normal = self.normal_protocols.get(device_id, "")
        allowed = tuple(self.allowed_protocols.get(device_id, ()))
        if normal and proto == normal:
            return 0
        if allowed and proto in allowed:
            return 1
        return 2 if allowed else 0


# --------------------------------------------------------------------------- #
# Simulated edge resource model
# --------------------------------------------------------------------------- #
@dataclass
class StepResourceUsage:
    """Simulated resource usage of one edge processing step (model output)."""

    step: int
    cpu_used: float                 # normalised CPU units (capacity = config.cpu_capacity)
    cpu_utilisation: float          # cpu_used / capacity
    memory_used_mb: float
    memory_utilisation: float       # memory_used_mb / capacity
    uplink_kbps: float              # edge summary uplink if alerts are sent upstream
    edge_latency_ms: float          # simulated local processing latency
    cloud_uplink_kbps: float        # simulated centralised raw-telemetry upload
    cloud_latency_ms: float         # simulated centralised round-trip latency
    cpu_saturated: bool
    bandwidth_saturated: bool

    def to_dict(self) -> Dict[str, Any]:
        return dict(
            step=self.step, cpu_used=round(self.cpu_used, 6),
            cpu_utilisation=round(self.cpu_utilisation, 6),
            memory_used_mb=round(self.memory_used_mb, 4),
            memory_utilisation=round(self.memory_utilisation, 6),
            uplink_kbps=round(self.uplink_kbps, 4),
            edge_latency_ms=round(self.edge_latency_ms, 4),
            cloud_uplink_kbps=round(self.cloud_uplink_kbps, 4),
            cloud_latency_ms=round(self.cloud_latency_ms, 4),
            cpu_saturated=bool(self.cpu_saturated),
            bandwidth_saturated=bool(self.bandwidth_saturated),
        )


class EdgeResourceModel:
    """
    Deterministic *simulated* resource model of the edge gateway.

    None of the numbers produced here are measurements of the host machine; they
    are model outputs derived from the volume of telemetry and the number of
    devices. All bandwidth values are a fraction of the real telemetry rates, as
    the agent only needs small summaries at the edge, whereas the centralised
    baseline ships raw records upstream.
    """

    def __init__(self, config: Optional[EdgeResourceConfig] = None):
        self.config = config or EdgeResourceConfig()
        #: Telemetry bytes observed in the current step (used for the uplink model).
        self._step_bytes: float = 0.0
        self._step_devices: int = 0

    # -- band 1: edge-local processing ---------------------------------- #
    def begin_step(self) -> None:
        self._step_bytes = 0.0
        self._step_devices = 0

    def observe_record(self, row: Any, n_devices: int) -> None:
        """Count one processed telemetry record for the current step."""
        self._step_bytes += 1.0
        self._step_devices = max(self._step_devices, int(n_devices))

    def finish_step(self, step: int, n_devices: int, n_actions_applied: int) -> StepResourceUsage:
        """Aggregate the simulated resource usage of one gateway step.

        Bandwidth model (per step of ``config.STEP_SECONDS`` seconds):

        * edge-resident processing ships only a small heartbeat record plus one
          compact alert summary per applied containment action;
        * the centralised baseline ships one telemetry record per device per step
          (``cloud_record_bytes``), which is the cost the edge avoids.

        Both numbers are *simulated model outputs*, not network measurements.
        """
        cfg = self.config
        cpu = (
            cfg.feature_extract_cpu_per_device * n_devices
            + cfg.detect_cpu_per_device * n_devices
            + cfg.policy_cpu_per_device * n_devices
            + 0.01 * n_actions_applied
        )
        mem = cfg.memory_per_device_mb * n_devices
        edge_bytes = 200.0 + 180.0 * n_actions_applied          # heartbeat + alerts
        central_bytes = float(n_devices) * float(cfg.cloud_record_bytes)
        kbps = lambda byte_count: byte_count * 8.0 / (STEP_SECONDS * 1000.0)  # noqa: E731
        edge_uplink = kbps(edge_bytes)
        cloud_uplink = kbps(central_bytes)
        edge_latency = cfg.edge_base_latency_ms + cfg.edge_latency_ms_per_device * n_devices
        if cpu > cfg.cpu_capacity:  # simulated queueing penalty when saturated
            edge_latency *= 1.0 + 2.0 * (cpu / max(cfg.cpu_capacity, 1e-9) - 1.0)
        cloud_latency = cfg.cloud_base_latency_ms + cfg.cloud_rtt_ms
        return StepResourceUsage(
            step=int(step), cpu_used=round(cpu, 6),
            cpu_utilisation=round(cpu / max(cfg.cpu_capacity, 1e-9), 6),
            memory_used_mb=round(mem, 4),
            memory_utilisation=round(mem / max(cfg.memory_capacity_mb, 1e-9), 6),
            uplink_kbps=round(edge_uplink, 6), edge_latency_ms=round(edge_latency, 4),
            cloud_uplink_kbps=round(cloud_uplink, 6), cloud_latency_ms=round(cloud_latency, 4),
            cpu_saturated=bool(cpu > cfg.cpu_capacity),
            bandwidth_saturated=bool(cloud_uplink > cfg.bandwidth_capacity_kbps),
        )

    # -- helpers used by the containment policy -------------------------- #
    def cpu_headroom(self, cpu_used: float) -> float:
        """Free CPU fraction of the (simulated) gateway after the current step."""
        cap = max(self.config.cpu_capacity, 1e-9)
        return float(max(0.0, min(1.0, (cap - cpu_used) / cap)))

    def memory_headroom(self, memory_used_mb: float) -> float:
        cap = max(self.config.memory_capacity_mb, 1e-9)
        return float(max(0.0, min(1.0, (cap - memory_used_mb) / cap)))


def feature_frame(vectors: Iterable[FeatureVector]) -> pd.DataFrame:
    """Convert feature vectors into a tidy DataFrame (dashboard / debugging)."""
    rows = []
    for v in vectors:
        row = {"step": v.step, "device_id": v.device_id}
        row.update(v.as_dict())
        rows.append(row)
    return pd.DataFrame(rows, columns=["step", "device_id", *FEATURE_NAMES])


if __name__ == "__main__":  # pragma: no cover - manual smoke test
    from config import ExperimentConfig
    from src.traffic_simulator import TrafficSimulator

    cfg = ExperimentConfig()
    sim = TrafficSimulator(cfg)
    tel, _ = sim.generate()
    mgr = AdaptiveBaselineManager(cfg.learning)
    mgr.fit(tel, cfg.calibration_steps)
    allowed = {d.device_id: d.allowed_protocols for d in sim.devices}
    normal = {d.device_id: d.normal_protocol for d in sim.devices}
    fe = FeatureExtractor(mgr, allowed, normal)
    sample = [fe.extract(r) for _, r in tel.head(4).iterrows()]
    print(feature_frame(sample).to_string(index=False))
