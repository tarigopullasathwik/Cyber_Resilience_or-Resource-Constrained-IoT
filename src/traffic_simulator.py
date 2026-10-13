"""
IoT device and traffic simulator (component A of the architecture).

Generates a heterogeneous fleet of synthetic IoT devices (>= 20 by default), each
with its own *normal* behavioural profile, and injects controlled synthetic attack
scenarios plus legitimate behavioural regime shifts.

Everything here is synthetic and local: no network sockets are opened and no real
device is contacted.

Design notes
------------
* All randomness flows from a single ``numpy.random.Generator`` seeded by
  ``ExperimentConfig.seed``, so an identical seed reproduces identical traffic.
* Device profiles are jittered deterministically from the same seed, therefore
  device-specific normal behaviour is stable across runs with the same seed.
* Attack *labels* (``ground_truth``, ``attack_type``) are attached to the data
  only for offline evaluation. The detector never receives them (enforced in
  ``experiment_runner`` by passing only the FEATURE_NAMES columns).
"""

from __future__ import annotations

import sys
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

if __package__ is None or __package__ == "":  # allow "python src/traffic_simulator.py"
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import (  # noqa: E402
    ATTACK_CATALOG,
    BENIGN_SHIFT_FRACTIONS,
    BENIGN_SHIFT_MAGNITUDE,
    DEVICE_TYPE_CYCLE,
    DEVICE_TYPES,
    HELD_OUT_ATTACKS,
    KNOWN_ATTACKS,
    AttackSpec,
    ExperimentConfig,
)

#: Probability per device-step of a *legitimate* burst (e.g. camera motion,
#: firmware check-in). Benign by definition: it exists to keep the
#: false-positive measurement honest.
BENIGN_BURST_PROBABILITY: float = 0.004
BENIGN_BURST_MULTIPLIER: float = 1.55

TELEMETRY_COLUMNS: Tuple[str, ...] = (
    "step", "device_id", "device_type", "criticality",
    "packet_rate", "bytes_per_sec", "failed_auth_attempts", "conn_count",
    "protocol", "dest_diversity", "cpu_usage", "memory_usage",
    "benign_shift", "ground_truth", "attack_type", "attack_id", "attack_held_out",
)


@dataclass
class DeviceProfile:
    """Normal behavioural profile of one simulated IoT device."""

    device_id: str
    device_type: str
    criticality: str
    base_pkt_rate: float
    base_bytes: float
    base_conns: float
    base_dst_div: float
    base_cpu: float
    base_mem: float
    normal_protocol: str
    allowed_protocols: Tuple[str, ...]
    base_auth_fail: float
    phase: float                      # diurnal phase offset in steps

    #: Multiplicative drift applied when a legitimate behavioural shift occurs.
    drift: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


@dataclass
class AttackEvent:
    """A scheduled synthetic attack against one simulated device."""

    attack_id: str
    device_id: str
    attack_type: str
    held_out: bool
    start_step: int
    end_step: int          # inclusive
    spec: AttackSpec

    @property
    def duration(self) -> int:
        return self.end_step - self.start_step + 1

    def to_dict(self) -> Dict[str, Any]:
        return dict(
            attack_id=self.attack_id, device_id=self.device_id,
            attack_type=self.attack_type, held_out=self.held_out,
            start_step=self.start_step, end_step=self.end_step, duration=self.duration,
        )


class TrafficSimulator:
    """
    Generates the synthetic IoT fleet and its telemetry.

    Parameters
    ----------
    config:
        Experiment configuration (seed, device count, steps, shift flags ...).
    seed:
        Optional seed override. When ``None`` the config seed is used.
    """

    def __init__(self, config: Optional[ExperimentConfig] = None, seed: Optional[int] = None):
        self.config = config or ExperimentConfig()
        self.seed = int(self.config.seed if seed is None else seed)
        self.rng = np.random.default_rng(self.seed)
        self.devices: List[DeviceProfile] = self._build_devices()
        self.attacks: List[AttackEvent] = self._schedule_attacks()
        self.shift_device_ids: Tuple[str, ...] = self._select_shift_devices()
        #: Steps at which legitimate behavioural shifts happen, derived from the
        #: run length so that short and long experiments both exercise the shift.
        self.shift_steps: Tuple[int, ...] = self._compute_shift_steps()

    def _compute_shift_steps(self) -> Tuple[int, ...]:
        earliest = max(int(self.config.warmup_steps), int(self.config.calibration_steps) + 5)
        latest = int(self.config.n_steps) - 5
        steps = []
        for fraction in BENIGN_SHIFT_FRACTIONS:
            step = int(round(fraction * int(self.config.n_steps)))
            step = int(np.clip(step, earliest, max(earliest, latest)))
            if step not in steps:
                steps.append(step)
        return tuple(sorted(steps))

    # ------------------------------------------------------------------ #
    # Fleet construction
    # ------------------------------------------------------------------ #
    def _build_devices(self) -> List[DeviceProfile]:
        devices: List[DeviceProfile] = []
        for i in range(int(self.config.n_devices)):
            dtype = DEVICE_TYPE_CYCLE[i % len(DEVICE_TYPE_CYCLE)]
            spec = DEVICE_TYPES[dtype]
            # Deterministic +/-10% per-device jitter -> device-specific profiles.
            jitter = lambda: float(self.rng.uniform(0.90, 1.10))  # noqa: E731
            devices.append(
                DeviceProfile(
                    device_id=f"dev_{i:02d}",
                    device_type=dtype,
                    criticality=str(spec["criticality"]),
                    base_pkt_rate=float(spec["base_pkt_rate"]) * jitter(),
                    base_bytes=float(spec["base_bytes"]) * jitter(),
                    base_conns=float(spec["base_conns"]) * jitter(),
                    base_dst_div=float(spec["base_dst_div"]) * jitter(),
                    base_cpu=float(spec["cpu"]) * jitter(),
                    base_mem=float(spec["mem"]) * jitter(),
                    normal_protocol=str(spec["proto"]),
                    allowed_protocols=tuple(spec["allowed_protocols"]),
                    base_auth_fail=float(spec["auth_fail"]),
                    phase=float(self.rng.uniform(0.0, 143.0)),
                    drift={},
                )
            )
        return devices

    def _select_shift_devices(self) -> Tuple[str, ...]:
        """Devices that legitimately change behaviour (benign concept drift)."""
        if not self.config.include_benign_shift:
            return ()
        attacked = {a.device_id for a in self.attacks}
        candidates = [d.device_id for d in self.devices if d.device_id not in attacked]
        if not candidates:
            return ()
        n_shift = max(2, int(round(0.25 * len(candidates))))
        idx = self.rng.choice(len(candidates), size=n_shift, replace=False)
        return tuple(sorted(candidates[int(i)] for i in idx))

    # ------------------------------------------------------------------ #
    # Attack scheduling
    # ------------------------------------------------------------------ #
    def _schedule_attacks(self) -> List[AttackEvent]:
        """Deterministically assign synthetic attacks to a subset of devices."""
        devices = list(self.devices)
        n_attacked = max(1, int(round(self.config.attacked_device_fraction * len(devices))))
        order = self.rng.permutation(len(devices))
        chosen = [devices[int(i)] for i in order[:n_attacked]]

        n_held_out = int(round(self.config.held_out_fraction * n_attacked))
        held_out_pool = list(HELD_OUT_ATTACKS)
        known_pool = list(KNOWN_ATTACKS)

        events: List[AttackEvent] = []
        # Never schedule an attack inside the benign calibration window.
        earliest = max(int(self.config.warmup_steps), int(self.config.calibration_steps) + 5)
        # Stratify attack start times over the post-calibration timeline so that
        # one experiment covers the whole window instead of clustering attacks.
        timeline_end = int(self.config.n_steps) - 2
        span = max(1, timeline_end - earliest)
        n_attacked_actual = max(1, len(chosen))
        segment = span / n_attacked_actual
        for k, dev in enumerate(chosen):
            if k < n_held_out and held_out_pool:
                atype = held_out_pool[k % len(held_out_pool)]
            else:
                atype = known_pool[k % len(known_pool)]
            spec = ATTACK_CATALOG[atype]
            dur_lo, dur_hi = spec.duration_steps
            duration = int(self.rng.integers(dur_lo, dur_hi + 1))
            seg_lo = int(earliest + k * segment)
            seg_hi = int(min(timeline_end - duration, earliest + (k + 1) * segment))
            if seg_hi <= seg_lo:
                start = int(min(max(seg_lo, earliest), max(earliest, timeline_end - duration)))
            else:
                start = int(self.rng.integers(seg_lo, seg_hi + 1))
            events.append(
                AttackEvent(
                    attack_id=f"{atype}-{k:02d}",
                    device_id=dev.device_id,
                    attack_type=atype,
                    held_out=bool(spec.held_out),
                    start_step=start,
                    end_step=start + duration - 1,
                    spec=spec,
                )
            )
        events.sort(key=lambda e: (e.start_step, e.device_id))
        return events

    # ------------------------------------------------------------------ #
    # Generation
    # ------------------------------------------------------------------ #
    def generate(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Produce the full telemetry log.

        Returns
        -------
        telemetry : DataFrame
            One row per (step, device) with raw telemetry + ground-truth labels.
        attacks : DataFrame
            One row per injected synthetic attack (schedule table).
        """
        n_steps = int(self.config.n_steps)
        attack_by_device: Dict[str, List[AttackEvent]] = {}
        for ev in self.attacks:
            attack_by_device.setdefault(ev.device_id, []).append(ev)

        rows: List[Dict[str, Any]] = []
        for step in range(n_steps):
            # Apply scheduled legitimate behavioural shifts (persist afterwards).
            self._apply_pending_shifts(step)
            for dev in self.devices:
                active = self._active_attack(attack_by_device.get(dev.device_id, []), step)
                rows.append(self._sample_device_step(dev, step, active))

        telemetry = pd.DataFrame(rows, columns=list(TELEMETRY_COLUMNS))
        attacks = pd.DataFrame([e.to_dict() for e in self.attacks])
        return telemetry, attacks

    @staticmethod
    def shift_direction(device_id: str) -> float:
        """
        Stable +1/-1 direction of a legitimate behaviour shift for one device.

        ``zlib.crc32`` is used instead of the built-in ``hash`` because Python
        randomises string hashing per process, which would make the same seed
        produce different traffic in different runs (a reproducibility trap).
        """
        return 1.0 if zlib.crc32(device_id.encode("utf-8")) % 2 == 0 else -1.0

    def _apply_pending_shifts(self, step: int) -> None:
        if step not in self.shift_steps:
            return
        amount = BENIGN_SHIFT_MAGNITUDE * (1.0 if step == self.shift_steps[0] else 0.5)
        for dev in self.devices:
            if dev.device_id not in self.shift_device_ids:
                continue
            # A stable per-device direction (some up, some down) keeps the shift
            # realistic: the fleet does not move in lock-step.
            direction = self.shift_direction(dev.device_id)
            factor = 1.0 + direction * amount
            dev.drift = {
                "pkt": dev.drift.get("pkt", 1.0) * factor,
                "bytes": dev.drift.get("bytes", 1.0) * factor,
                "conns": dev.drift.get("conns", 1.0) * (1.0 + direction * amount * 0.5),
                "cpu": dev.drift.get("cpu", 1.0) * (1.0 + max(-0.2, direction * amount * 0.6)),
            }

    @staticmethod
    def _active_attack(events: List[AttackEvent], step: int) -> Optional[AttackEvent]:
        for ev in events:
            if ev.start_step <= step <= ev.end_step:
                return ev
        return None

    def _sample_device_step(
        self, dev: DeviceProfile, step: int, attack: Optional[AttackEvent]
    ) -> Dict[str, Any]:
        # ---- legitimate baseline (diurnal + noise) ---------------------- #
        diurnal = 1.0 + 0.15 * np.sin(2.0 * np.pi * (step + dev.phase) / 144.0)
        burst = 1.0
        if self.rng.random() < BENIGN_BURST_PROBABILITY:
            burst = float(self.rng.uniform(1.2, BENIGN_BURST_MULTIPLIER))

        pkt = dev.base_pkt_rate * diurnal * burst * dev.drift.get("pkt", 1.0)
        byts = dev.base_bytes * diurnal * burst * dev.drift.get("bytes", 1.0)
        conns = dev.base_conns * dev.drift.get("conns", 1.0)
        dst_div = dev.base_dst_div
        auth_fail = dev.base_auth_fail
        proto = dev.normal_protocol

        # multiplicative noise on the traffic observables
        pkt *= float(self.rng.normal(1.0, 0.07))
        byts *= float(self.rng.normal(1.0, 0.09))
        conns *= float(self.rng.normal(1.0, 0.10))

        # ---- attack overlay --------------------------------------------- #
        attacked = attack is not None
        if attacked:
            spec = attack.spec
            pkt *= spec.pkt_rate_mult
            byts *= spec.bytes_mult
            conns *= spec.conns_mult
            dst_div += spec.dst_div_add
            auth_fail += spec.auth_fail_add
            if spec.proto:
                proto = spec.proto
            # attacks fluctuate around their target scale (not a clean plateau)
            pkt *= float(self.rng.normal(1.0, 0.12))
            byts *= float(self.rng.normal(1.0, 0.14))

        pkt = float(max(0.0, pkt))
        byts = float(max(0.0, byts))
        conns = float(max(0.0, conns))
        dst_div = float(max(1.0, dst_div + float(self.rng.normal(0.0, 0.35))))
        auth_rate = float(max(0.0, auth_fail))
        failed_auth = float(self.rng.poisson(auth_rate)) if auth_rate > 0 else 0.0

        # ---- resource telemetry derived from the traffic ----------------- #
        traffic_ratio = float(np.clip(pkt / max(dev.base_pkt_rate, 1e-6), 0.0, 25.0))
        cpu = dev.base_cpu * (0.75 + 0.25 * traffic_ratio) * dev.drift.get("cpu", 1.0)
        mem = dev.base_mem * (0.85 + 0.15 * min(traffic_ratio, 12.0))
        if attacked:
            cpu *= attack.spec.cpu_mult
            mem *= attack.spec.mem_mult
        cpu = float(np.clip(cpu + self.rng.normal(0.0, 1.2), 0.5, 99.0))
        mem = float(np.clip(mem + self.rng.normal(0.0, 1.5), 0.5, 99.0))

        return {
            "step": int(step),
            "device_id": dev.device_id,
            "device_type": dev.device_type,
            "criticality": dev.criticality,
            "packet_rate": round(pkt, 4),
            "bytes_per_sec": round(byts, 2),
            "failed_auth_attempts": failed_auth,
            "conn_count": round(conns, 3),
            "protocol": proto,
            "dest_diversity": round(dst_div, 3),
            "cpu_usage": round(cpu, 3),
            "memory_usage": round(mem, 3),
            "benign_shift": bool(
                dev.device_id in self.shift_device_ids
                and self.shift_steps and step >= self.shift_steps[0]
            ),
            "ground_truth": int(attacked),
            "attack_type": attack.attack_type if attacked else "",
            "attack_id": attack.attack_id if attacked else "",
            "attack_held_out": bool(attack.held_out) if attacked else False,
        }

    # ------------------------------------------------------------------ #
    # Convenience accessors (used by tests and the dashboard)
    # ------------------------------------------------------------------ #
    @property
    def device_ids(self) -> List[str]:
        return [d.device_id for d in self.devices]

    def profile_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "device_id": d.device_id,
                    "device_type": d.device_type,
                    "criticality": d.criticality,
                    "normal_pkt_rate": round(d.base_pkt_rate, 3),
                    "normal_bytes": round(d.base_bytes, 2),
                    "normal_conns": round(d.base_conns, 3),
                    "normal_dst_div": round(d.base_dst_div, 2),
                    "normal_cpu": round(d.base_cpu, 3),
                    "normal_mem": round(d.base_mem, 3),
                    "normal_protocol": d.normal_protocol,
                }
                for d in self.devices
            ]
        )

    def calibrate_window(self) -> Tuple[int, int]:
        """Inclusive-exclusive step range used as the benign calibration window."""
        return 0, int(self.config.calibration_steps)


if __name__ == "__main__":  # pragma: no cover - manual smoke test
    sim = TrafficSimulator()
    tel, atk = sim.generate()
    print(f"devices={tel['device_id'].nunique()} rows={len(tel)} attacks={len(atk)}")
    print(tel.head(3).to_string(index=False))
    print(atk.to_string(index=False))
