"""
Central configuration for the Adaptive Edge Cyber-Resilience simulation.

Research title:
    "Design and Development of an Adaptive Edge Cyber-Resilience Mechanism for
     Real-Time Detection and Autonomous Containment of Unknown Attacks in
     Resource-Constrained IoT Networks."

Every tunable number used by the simulation lives here so that experiments stay
reproducible and so that the dashboard can display the exact configuration that
produced a result set.

IMPORTANT SCOPE NOTE
--------------------
This is a *simulation*. Nothing in this file (or the project) contacts a real
network, touches a firewall, or attacks real equipment. All "devices", traffic,
attacks and containment actions are synthetic objects living inside the Python
process.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PROJECT_ROOT: Path = Path(__file__).resolve().parent
DATA_DIR: Path = PROJECT_ROOT / "data"
RESULTS_DIR: Path = PROJECT_ROOT / "results"
FIGURES_DIR: Path = PROJECT_ROOT / "figures"

for _d in (DATA_DIR, RESULTS_DIR, FIGURES_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# Global run defaults
# --------------------------------------------------------------------------- #
DEFAULT_SEED: int = 20261013
N_DEVICES: int = 24            # >= 20 heterogeneous devices (requirement 3A)
N_STEPS: int = 360             # simulated edge time steps (1 step ~ 5 seconds of wall time)
CALIBRATION_STEPS: int = 120   # first N steps, benign-only, used to fit the detector
#: Attacks never start before the end of the benign calibration window (+margin),
#: so the threshold is never fitted on attack data (requirement 3C/E).
WARMUP_STEPS: int = 130        # earliest step at which an attack may start

#: Sampling interval of one simulation step, in seconds. Used for latency
#: conversion (steps -> seconds) and for bandwidth maths.
STEP_SECONDS: float = 5.0

#: Edge feature vector produced by ``src.feature_extractor`` (order matters).
FEATURE_NAMES: Tuple[str, ...] = (
    "pkt_rate_ratio",      # packet rate relative to device's learned normal rate
    "bytes_rate_ratio",    # bytes/s relative to device's normal bytes/s
    "failed_auth_rate",    # failed authentication attempts per step (absolute)
    "conn_count_z",        # connection-count deviation (robust z-score)
    "dest_diversity_z",    # destination-address diversity deviation
    "cpu_delta",           # CPU delta vs device normal CPU occupancy
    "mem_delta",           # memory delta vs device normal memory occupancy
    "protocol_shift",      # 0/1/2 how far the protocol mix moved from normal
)

# --------------------------------------------------------------------------- #
# Device catalogue (heterogeneous IoT fleet)
# --------------------------------------------------------------------------- #
#: criticality is an *asset-importance* label used by threat assessment and the
#: containment policy only. It is never used as a detector input.
DEVICE_TYPES: Dict[str, Dict[str, Any]] = {
    "ip_camera": dict(
        criticality="high", base_pkt_rate=48.0, base_bytes=95_000.0,
        base_conns=6, base_dst_div=2, proto="tcp", cpu=38.0, mem=46.0,
        auth_fail=0.01, allowed_protocols=("tcp", "rtsp"),
    ),
    "smart_thermostat": dict(
        criticality="medium", base_pkt_rate=8.0, base_bytes=4_500.0,
        base_conns=2, base_dst_div=1, proto="mqtt", cpu=9.0, mem=18.0,
        auth_fail=0.0, allowed_protocols=("mqtt", "udp"),
    ),
    "smart_plug": dict(
        criticality="low", base_pkt_rate=6.0, base_bytes=2_200.0,
        base_conns=2, base_dst_div=1, proto="mqtt", cpu=6.0, mem=12.0,
        auth_fail=0.0, allowed_protocols=("mqtt",),
    ),
    "smart_door_lock": dict(
        criticality="critical", base_pkt_rate=4.0, base_bytes=1_800.0,
        base_conns=2, base_dst_div=1, proto="mqtt", cpu=7.0, mem=15.0,
        auth_fail=0.02, allowed_protocols=("mqtt", "tcp"),
    ),
    "smart_meter": dict(
        criticality="high", base_pkt_rate=12.0, base_bytes=6_400.0,
        base_conns=3, base_dst_div=1, proto="udp", cpu=14.0, mem=22.0,
        auth_fail=0.0, allowed_protocols=("udp", "coap"),
    ),
    "wearable_health": dict(
        criticality="high", base_pkt_rate=10.0, base_bytes=5_100.0,
        base_conns=2, base_dst_div=1, proto="https", cpu=12.0, mem=28.0,
        auth_fail=0.005, allowed_protocols=("https", "tcp"),
    ),
    "industrial_plc": dict(
        criticality="critical", base_pkt_rate=30.0, base_bytes=18_000.0,
        base_conns=4, base_dst_div=1, proto="modbus", cpu=41.0, mem=52.0,
        auth_fail=0.0, allowed_protocols=("modbus", "tcp"),
    ),
    "voice_assistant": dict(
        criticality="medium", base_pkt_rate=16.0, base_bytes=42_000.0,
        base_conns=3, base_dst_div=3, proto="https", cpu=22.0, mem=44.0,
        auth_fail=0.01, allowed_protocols=("https", "tcp"),
    ),
    "light_controller": dict(
        criticality="low", base_pkt_rate=5.0, base_bytes=1_600.0,
        base_conns=2, base_dst_div=1, proto="zigbee_gw", cpu=5.0, mem=10.0,
        auth_fail=0.0, allowed_protocols=("zigbee_gw", "mqtt"),
    ),
    "irrigation_sensor": dict(
        criticality="medium", base_pkt_rate=7.0, base_bytes=3_100.0,
        base_conns=2, base_dst_div=1, proto="udp", cpu=8.0, mem=16.0,
        auth_fail=0.0, allowed_protocols=("udp", "coap"),
    ),
}

DEVICE_TYPE_CYCLE: Tuple[str, ...] = tuple(DEVICE_TYPES.keys())

#: Benign behavioural regime shift (legitimate change): at these points in the
#: run a subset of devices legitimately changes its traffic pattern (firmware
#: update, new schedule, added sensor). Fractions of the run length are used (not
#: absolute steps) so short and long experiments both exercise the shift. The
#: adaptive system must learn the change instead of permanently flagging it.
BENIGN_SHIFT_FRACTIONS: Tuple[float, ...] = (0.55, 0.78)
BENIGN_SHIFT_MAGNITUDE: float = 0.35  # multiplicative change on rate/bytes/conns

# --------------------------------------------------------------------------- #
# Synthetic attack catalogue
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class AttackSpec:
    """One synthetic attack pattern used by the traffic simulator."""

    name: str
    description: str
    #: True  -> pattern is withheld from *any* detector tuning and reported as
    #:          the "unknown attack" proxy in the evaluation.
    #: False -> pattern is part of the development/known set.
    held_out: bool
    duration_steps: Tuple[int, int]          # (min, max) attack duration
    #: Multipliers/offsets applied to the device's normal profile.
    pkt_rate_mult: float = 1.0
    bytes_mult: float = 1.0
    conns_mult: float = 1.0
    dst_div_add: int = 0
    auth_fail_add: float = 0.0
    cpu_mult: float = 1.0
    mem_mult: float = 1.0
    proto: str = ""            # protocol to switch to, "" = keep normal


ATTACK_CATALOG: Dict[str, AttackSpec] = {
    "flood": AttackSpec(
        name="flood",
        description="Traffic flooding / DoS-like packet surge toward the gateway.",
        held_out=False, duration_steps=(8, 16),
        pkt_rate_mult=7.5, bytes_mult=5.0, conns_mult=3.0, cpu_mult=1.7,
    ),
    "port_scan": AttackSpec(
        name="port_scan",
        description="Port scanning: many connection attempts to unusual destination diversity.",
        held_out=False, duration_steps=(10, 18),
        pkt_rate_mult=2.4, bytes_mult=0.6, conns_mult=6.0, dst_div_add=14, cpu_mult=1.3,
    ),
    "brute_force": AttackSpec(
        name="brute_force",
        description="Repeated failed authentication attempts against a device/service.",
        held_out=False, duration_steps=(12, 22),
        pkt_rate_mult=1.6, bytes_mult=1.2, conns_mult=1.4, auth_fail_add=9.0, cpu_mult=1.2,
    ),
    "exfiltration": AttackSpec(
        name="exfiltration",
        description="Data-exfiltration-like outbound traffic (large uploads to few destinations).",
        held_out=False, duration_steps=(14, 24),
        pkt_rate_mult=1.3, bytes_mult=9.0, conns_mult=1.2, mem_mult=1.25, cpu_mult=1.15,
    ),
    # ------------------------------------------------------------------ #
    # HELD-OUT (unknown-attack proxy). These patterns are never used while
    # calibrating or tuning the detector/thresholds, and the evaluation
    # reports them separately as "held-out / unseen".
    # ------------------------------------------------------------------ #
    "spoofing": AttackSpec(
        name="spoofing",
        description="Spoofing-like behavioural deviation: identity/protocol mismatch with mild volume change.",
        held_out=True, duration_steps=(12, 20),
        pkt_rate_mult=1.9, bytes_mult=1.7, conns_mult=0.6, dst_div_add=3,
        proto="tcp", cpu_mult=0.8, mem_mult=1.1,
    ),
    "stealth_exfiltration": AttackSpec(
        name="stealth_exfiltration",
        description="Low-and-slow exfiltration held below flood scales (novel parameter combination).",
        held_out=True, duration_steps=(20, 30),
        pkt_rate_mult=1.25, bytes_mult=2.6, conns_mult=1.1, dst_div_add=1,
        mem_mult=1.05, proto="https",
    ),
}

KNOWN_ATTACKS: Tuple[str, ...] = tuple(
    name for name, spec in ATTACK_CATALOG.items() if not spec.held_out
)
HELD_OUT_ATTACKS: Tuple[str, ...] = tuple(
    name for name, spec in ATTACK_CATALOG.items() if spec.held_out
)

# --------------------------------------------------------------------------- #
# Detector configuration
# --------------------------------------------------------------------------- #
@dataclass
class DetectorConfig:
    """Configuration for the anomaly detectors (see src/anomaly_detector.py)."""

    #: Robust z-score magnitudes are squashed with ``z / (z + z_scale)`` to map
    #: an unbounded deviation into [0, 1).
    z_scale: float = 4.0
    #: Weight of the largest single-feature contribution in the statistical
    #: score; the remainder is the mean contribution over all features.
    stat_max_weight: float = 0.5
    #: Absolute per-feature scale floor (guards zero-variance calibration columns).
    stat_min_scale: float = 1e-3
    #: Contamination hint handed to IsolationForest (expected anomalous ratio
    #: in the benign calibration window -- kept low on purpose).
    if_contamination: float = 0.05
    if_n_estimators: int = 120
    if_max_samples: int = 128          # keeps the model lightweight for an edge gateway
    #: Weight of the transparent statistical score in the fused score.
    stat_weight: float = 0.6
    #: Weight of the IsolationForest score in the fused score.
    if_weight: float = 0.4
    #: Quantile of benign calibration scores used as the starting threshold.
    initial_threshold_quantile: float = 0.995
    #: EWMA smoothing factor for the adaptive threshold (small = slow, stable).
    threshold_ewma_alpha: float = 0.002
    #: Absolute floor/ceiling so the threshold can never collapse or explode.
    threshold_min: float = 0.05
    threshold_max: float = 0.95
    #: Static detector: when True the threshold is calibrated once on benign data
    #: and then frozen forever (system S1); when False the hand-set constant below
    #: is used instead.
    static_threshold_auto: bool = True
    #: Static (non-adaptive) detector threshold on the fused score scale.
    static_threshold: float = 0.30
    #: Threshold calibration rule (per device, benign window only):
    #: ``t_device = quantile(benign scores of that device, threshold_quantile)
    #: * threshold_margin``, clipped to [threshold_min, threshold_max].
    #: An empirical upper quantile is used instead of a mean+k*sigma rule because
    #: the score distribution is skewed; the quantile directly targets the
    #: in-sample false-positive rate (1 - threshold_quantile).
    threshold_quantile: float = 0.995
    threshold_margin: float = 1.05
    #: Secondary robust floor: t_device >= median + robust_k * 1.4826 * MAD.
    threshold_robust_k: float = 3.0
    min_calibration_rows_per_device: int = 20
    #: Number of recent benign scores kept for the adaptive threshold estimate.
    threshold_window: int = 400
    #: Quantile of the benign score window used as the adaptive threshold target.
    threshold_window_quantile: float = 0.99
    #: Minimum samples in a device's benign window before its adaptive threshold
    #: is allowed to move.
    threshold_min_samples: int = 40
    #: A frozen device is released after this many consecutive non-anomalous steps.
    unfreeze_score_fraction: float = 1.0
    #: A device is frozen (no baseline/threshold learning) when its score
    #: exceeds this multiplier of the threshold.
    freeze_score_multiplier: float = 1.0
    #: Number of consecutive benign observations required to unfreeze a device.
    unfreeze_patience: int = 12


# --------------------------------------------------------------------------- #
# Adaptive learning configuration
# --------------------------------------------------------------------------- #
@dataclass
class LearningConfig:
    """Baseline adaptation + anti-poisoning configuration (src/adaptive_learning.py)."""

    #: EWMA learning rate for a "clean" benign observation (score well below threshold).
    learning_rate_clean: float = 0.05
    #: Reduced learning rate used in the grey zone (possible benign drift).
    learning_rate_drift: float = 0.04
    #: Learning rate applied to volume features while a freeze has been released
    #: by the drift-evidence pathway (see README, "drift gates"). Kept moderate:
    #: a release can only move volume features, and every release is logged with
    #: the gate values that justified it.
    drift_release_learning_rate: float = 0.10
    #: Scores above this *threshold* fraction are treated as grey-zone drift.
    drift_zone_fraction: float = 0.6
    #: Consecutive grey-zone observations before a device is treated as having
    #: legitimately drifted (benign behaviour change) and may adapt faster.
    drift_patience: int = 10
    #: Maximum absolute resource deviation (CPU/memory percentage points, as a
    #: fraction) still considered consistent with legitimate drift.
    drift_max_resource_delta: float = 0.15
    #: Maximum distance a single update may move a baseline value (relative).
    max_single_update_fraction: float = 0.15
    #: Absolute floor for baseline scale parameters to avoid division by zero.
    min_scale: float = 0.25
    #: Relative floor for scale parameters (fraction of the baseline location).
    min_scale_fraction: float = 0.03
    #: Consecutive benign observations required to release a frozen baseline.
    freeze_patience: int = 10
    #: Minimum traffic allowance under which learning is still accepted. The
    #: detector and the baseline both work on *allowance-normalised* telemetry, so
    #: a mild throttle (RATE_LIMIT, allowance 0.35) still leaves usable signal and
    #: may be learned through the gate-protected drift pathway. Heavier actions
    #: (REAUTHENTICATE 0.10, QUARANTINE 0.00) amplify measurement noise and are
    #: excluded: learning under them is blocked and logged as
    #: ``blocked_containment``.
    learning_min_allowance: float = 0.25
    # --- drift-evidence pathway (may release a freeze for *benign* drift) --- #
    #: Legitimate behavioural change only moves volume, and only within a
    #: plausible band; attacks move volume far outside it and/or trip a
    #: qualitative indicator (auth failures, destination diversity, protocol,
    #: CPU/memory shift). These gates are engineering heuristics that an adaptive
    #: attacker could try to evade -- see README limitations.
    drift_max_volume_ratio: float = 2.2
    drift_min_volume_ratio: float = 0.25
    drift_max_abs_z: float = 3.0
    drift_max_failed_auth: float = 0.0
    drift_require_normal_protocol: bool = True


# --------------------------------------------------------------------------- #
# Threat assessment configuration
# --------------------------------------------------------------------------- #
@dataclass
class SeverityConfig:
    """
    Severity = 100 * (w_mag*anomaly_magnitude + w_conf*confidence
                      + w_crit*criticality + w_ctx*network_context)

    Factor definitions (all normalised to [0, 1] before weighting):
      * anomaly_magnitude: how far the fused score sits above the adaptive
        threshold, saturating at ``magnitude_saturation`` x threshold.
      * confidence:        detector self-confidence, i.e. the agreement between
        the statistical score and the IsolationForest score plus the margin by
        which the threshold was exceeded.
      * criticality:       asset importance of the device (its device-class
        criticality; configurable per device by an operator).
      * network_context:   blast-radius proxy = share of fleet currently
        anomalous (0.6) + relative traffic volume of the device (0.4).
    """

    w_magnitude: float = 0.35
    w_confidence: float = 0.25
    w_criticality: float = 0.20
    w_network_context: float = 0.20
    magnitude_saturation: float = 3.0
    severity_min: float = 0.0
    severity_max: float = 100.0

    #: Criticality mapping (device criticality label -> value in [0, 1]).
    criticality_values: Dict[str, float] = field(
        default_factory=lambda: {"low": 0.25, "medium": 0.5, "high": 0.75, "critical": 1.0}
    )


# --------------------------------------------------------------------------- #
# Containment policy configuration
# --------------------------------------------------------------------------- #
@dataclass
class ContainmentConfig:
    """Autonomous containment policy thresholds (src/containment.py)."""

    #: Severity bands -> response level.
    observe_max: float = 25.0
    rate_limit_max: float = 50.0
    reauth_max: float = 75.0          # above this -> QUARANTINE
    #: Critical devices are quarantined only above this higher severity
    #: (safeguard against unnecessary isolation of critical assets).
    critical_quarantine_severity: float = 85.0
    #: Minimum steps between two containment actions on the same device.
    cooldown_steps: int = 4
    #: If severity grows by at least this much, escalate immediately (ignore cooldown).
    escalation_severity_jump: float = 18.0
    #: Consecutive verified-benign steps required before stepping down/restoring.
    recovery_steps: int = 8
    #: Response level order (index = escalation level).
    levels: Tuple[str, ...] = ("OBSERVE", "RATE_LIMIT", "REAUTHENTICATE", "QUARANTINE")
    #: Simulated edge cost of applying each action (CPU-equivalent units).
    action_cost: Dict[str, float] = field(
        default_factory=lambda: {
            "OBSERVE": 0.010, "RATE_LIMIT": 0.045, "REAUTHENTICATE": 0.090, "QUARANTINE": 0.140,
        }
    )
    #: Traffic allowance multiplier while each action is active (simulated).
    action_traffic_allowance: Dict[str, float] = field(
        default_factory=lambda: {
            "OBSERVE": 1.0, "RATE_LIMIT": 0.35, "REAUTHENTICATE": 0.10, "QUARANTINE": 0.0,
        }
    )
    #: Minimum free edge CPU (in the simulated capacity model) required to take
    #: the next action; otherwise the policy downgrades to a cheaper action.
    min_cpu_headroom_for_action: float = 0.06
    #: Safeguard against unnecessary isolation: when True a device must first fail
    #: a lighter response (RATE_LIMIT/REAUTHENTICATE) before QUARANTINE is allowed.
    require_graduated_escalation: bool = True
    #: Steps of continuous anomalous traffic at an already effective response level
    #: before the policy escalates one level further.
    sustained_anomaly_steps: int = 12


# --------------------------------------------------------------------------- #
# Edge / centralised resource model
# --------------------------------------------------------------------------- #
@dataclass
class EdgeResourceConfig:
    """
    Simulated resource envelope of the edge gateway. These are *simulated*
    capacity numbers, not measurements of the host machine.
    """

    cpu_capacity: float = 1.0                 # normalised compute budget per step
    memory_capacity_mb: float = 256.0         # edge gateway RAM budget
    bandwidth_capacity_kbps: float = 2_048.0  # uplink to the aggregation tier
    #: Simulated per-device processing cost (CPU units) of feature extraction.
    feature_extract_cpu_per_device: float = 0.004
    #: Simulated CPU cost of one detector scoring operation.
    detect_cpu_per_device: float = 0.006
    #: Simulated CPU cost of severity/containment bookkeeping per device.
    policy_cpu_per_device: float = 0.002
    #: Simulated per-device memory footprint (MB) of edge state.
    memory_per_device_mb: float = 3.2
    #: Base simulated latency of one edge processing step (ms).
    edge_base_latency_ms: float = 12.0
    #: Additional simulated latency per device processed (ms).
    edge_latency_ms_per_device: float = 0.35
    #: Centralised (cloud) baseline: RTT to the remote aggregation service (ms).
    cloud_rtt_ms: float = 95.0
    #: Simulated uplink bytes per telemetry record sent to the cloud.
    cloud_record_bytes: int = 420
    #: Simulated warm-up latency of the centralised pipeline (ms).
    cloud_base_latency_ms: float = 140.0


# --------------------------------------------------------------------------- #
# Response verification configuration
# --------------------------------------------------------------------------- #
@dataclass
class VerificationConfig:
    """Post-containment effectiveness check (src/response_verifier.py)."""

    #: Steps of observation collected after an action before judging.
    observation_window: int = 4
    #: Relative decrease of the fused anomaly score required to call it a success.
    success_score_reduction: float = 0.35
    #: Absolute score below which a device counts as "not anomalous" after action.
    success_absolute_score: float = 0.20
    #: Outcomes.
    outcome_success: str = "SUCCESS"
    outcome_failure: str = "FAILURE"
    outcome_inconclusive: str = "INCONCLUSIVE"
    #: Maximum escalations for a single device before the policy stays at the max level.
    max_escalations: int = 3


# --------------------------------------------------------------------------- #
# Experiment configuration
# --------------------------------------------------------------------------- #
@dataclass
class ExperimentConfig:
    """Full, serialisable configuration of one experiment run."""

    seed: int = DEFAULT_SEED
    n_devices: int = N_DEVICES
    n_steps: int = N_STEPS
    calibration_steps: int = CALIBRATION_STEPS
    warmup_steps: int = WARMUP_STEPS
    step_seconds: float = STEP_SECONDS
    #: Fraction of devices that are attacked at all during the run.
    attacked_device_fraction: float = 0.45
    #: Fraction of those attacks drawn from the held-out (unknown) catalogue.
    held_out_fraction: float = 0.4
    #: Include a benign behavioural regime shift for some devices.
    include_benign_shift: bool = True
    #: Simulate a low-resource (CPU-starved) edge gateway.
    low_resource_edge: bool = False
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    learning: LearningConfig = field(default_factory=LearningConfig)
    severity: SeverityConfig = field(default_factory=SeverityConfig)
    containment: ContainmentConfig = field(default_factory=ContainmentConfig)
    resources: EdgeResourceConfig = field(default_factory=EdgeResourceConfig)
    verification: VerificationConfig = field(default_factory=VerificationConfig)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    # -- convenience factories ------------------------------------------- #
    @classmethod
    def low_resource(cls, **kwargs: Any) -> "ExperimentConfig":
        """
        A CPU-constrained edge gateway (same seed and fleet as the main scenario).

        ``cpu_capacity=0.36`` against a baseline processing cost of ~0.288 leaves
        ~0.072 free units: enough for RATE_LIMIT (cost 0.045) but not for
        REAUTHENTICATE (0.09) or QUARANTINE (0.14). The resource guard therefore
        downgrades every stronger response to RATE_LIMIT -- the behaviour a
        resource-constrained deployment would exhibit.
        """
        cfg = cls(low_resource_edge=True, **kwargs)
        cfg.resources = EdgeResourceConfig(
            cpu_capacity=0.36,
            memory_capacity_mb=128.0,
            bandwidth_capacity_kbps=512.0,
            edge_base_latency_ms=26.0,
            edge_latency_ms_per_device=0.85,
            memory_per_device_mb=3.2,
        )
        return cfg

    @classmethod
    def starved_edge(cls, **kwargs: Any) -> "ExperimentConfig":
        """
        A gateway that is too starved to contain anything (extreme case).

        ``cpu_capacity=0.30`` is below the ~0.288 baseline processing cost plus the
        minimum headroom required to take any action, so the policy can only
        OBSERVE: detection still works, autonomous containment does not happen.
        This case exists to show the failure mode honestly instead of hiding it.
        """
        cfg = cls(low_resource_edge=True, **kwargs)
        cfg.resources = EdgeResourceConfig(
            cpu_capacity=0.30,
            memory_capacity_mb=96.0,
            bandwidth_capacity_kbps=256.0,
            edge_base_latency_ms=34.0,
            edge_latency_ms_per_device=1.10,
            memory_per_device_mb=3.2,
        )
        return cfg


#: Systems compared by the experiment runner (requirement 4).
SYSTEM_STATIC = "S1_static_threshold"
SYSTEM_ADAPTIVE = "S2_adaptive_no_containment"
SYSTEM_PROPOSED = "S3_adaptive_closed_loop"
SYSTEMS: Tuple[str, ...] = (SYSTEM_STATIC, SYSTEM_ADAPTIVE, SYSTEM_PROPOSED)

SYSTEM_LABELS: Dict[str, str] = {
    SYSTEM_STATIC: "S1 Static threshold",
    SYSTEM_ADAPTIVE: "S2 Adaptive detection (no containment)",
    SYSTEM_PROPOSED: "S3 Adaptive closed loop (proposed)",
}
