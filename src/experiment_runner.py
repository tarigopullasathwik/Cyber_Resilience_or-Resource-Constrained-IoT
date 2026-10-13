"""
Experiment runner (component 4 and 10 of the research plan).

Runs the three compared systems over **identical** telemetry (same seed, same
fleet, same attack schedule, same benign shifts) so differences can only come
from the detection/containment strategy:

============  ==========================================================
System        Description
============  ==========================================================
S1_static     Statistical + IsolationForest scoring with a threshold
              calibrated once on benign data and then frozen; no baseline
              adaptation, no containment.
S2_adaptive   Same scorers, adaptive threshold and adaptive device
              baselines (with the anti-poisoning guards), but no
              autonomous containment: anomalies are only reported.
S3_closed     The proposed system: adaptive detection + threat assessment
              + autonomous containment + response verification + recovery
              + baseline learning gated by the containment state.
============  ==========================================================

Causality rules enforced by the runner:

* A step's decisions only use data available at that step (network context is
  taken from the previous step's fleet state).
* Ground-truth labels are never passed to the detector, the assessor, the
  controller or the learning manager; they are written to the logs for offline
  evaluation only.
* The detector sees *allowance-normalised* telemetry, so the gateway's own
  throttling cannot mask (or create) an anomaly. The verifier instead sees the
  raw post-containment telemetry, which is what a response is supposed to change.
"""

from __future__ import annotations

import json
import platform
import sys
import time
import tracemalloc
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import (  # noqa: E402
    FEATURE_NAMES,
    FIGURES_DIR,
    RESULTS_DIR,
    SYSTEM_ADAPTIVE,
    SYSTEM_LABELS,
    SYSTEM_PROPOSED,
    SYSTEM_STATIC,
    SYSTEMS,
    ExperimentConfig,
)
from src.adaptive_learning import AdaptiveBaselineManager  # noqa: E402
from src.anomaly_detector import AnomalyDetector, ScoreResult  # noqa: E402
from src.containment import ClosedLoopController  # noqa: E402
from src.feature_extractor import EdgeResourceModel, FeatureExtractor  # noqa: E402
from src.metrics import (  # noqa: E402
    SystemMetrics,
    attack_episodes,
    comparison_table,
    summarize_system,
)
from src.response_verifier import ResponseVerifier  # noqa: E402
from src.threat_assessment import ThreatAssessor  # noqa: E402
from src.traffic_simulator import TrafficSimulator  # noqa: E402

#: Simulated containment effect on the device's observable telemetry. RATE_LIMIT
#: and QUARANTINE throttles volume features proportionally to the allowance and
#: reduces the attack-driven part of CPU/memory load. This is a *model*, clearly
#: documented in the README, not a measurement.
CONTAINMENT_CPU_EFFECT: float = 0.5
#: Floor used when normalising telemetry by a tiny allowance (division safety).
ALLOWANCE_FLOOR: float = 0.02
#: Minimum feature contribution for a feature to be recorded as an "offending
#: indicator" whose reduction is measured by the verifier.
INDICATOR_CONTRIBUTION_FLOOR: float = 0.30


@dataclass
class ExperimentResult:
    """Everything one multi-system experiment produced."""

    scenario: str
    config: ExperimentConfig
    seed: int
    telemetry: pd.DataFrame
    attacks: pd.DataFrame
    profiles: pd.DataFrame
    systems: Dict[str, SystemMetrics]
    decision_logs: Dict[str, pd.DataFrame]
    learning_logs: Dict[str, pd.DataFrame]
    resource_logs: Dict[str, pd.DataFrame]
    metadata: Dict[str, Any] = field(default_factory=dict)

    # -- derived views ------------------------------------------------- #
    @property
    def comparison(self) -> pd.DataFrame:
        return comparison_table(self.systems)

    @property
    def label(self) -> str:
        return f"{self.scenario} (seed={self.seed})"

    def decision_log(self) -> pd.DataFrame:
        frames = []
        for system, frame in self.decision_logs.items():
            if frame.empty:
                continue
            copy = frame.copy()
            copy.insert(0, "system", system)
            copy.insert(1, "system_label", SYSTEM_LABELS.get(system, system))
            frames.append(copy)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def action_log(self) -> pd.DataFrame:
        frames = []
        for system, metrics in self.systems.items():
            if metrics.actions.empty:
                continue
            copy = metrics.actions.copy()
            copy.insert(0, "system", system)
            frames.append(copy)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def verification_log(self) -> pd.DataFrame:
        frames = []
        for system, metrics in self.systems.items():
            if metrics.verification.empty:
                continue
            copy = metrics.verification.copy()
            copy.insert(0, "system", system)
            frames.append(copy)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def learning_log(self) -> pd.DataFrame:
        frames = []
        for system, frame in self.learning_logs.items():
            if frame.empty:
                continue
            copy = frame.copy()
            copy.insert(0, "system", system)
            frames.append(copy)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def summary_table(self) -> pd.DataFrame:
        """One row per system: metrics plus the measured host numbers."""
        rows = [{**metrics.summary, **metrics.host_measurements}
                for metrics in self.systems.values()]
        return pd.DataFrame(rows)

    # -- persistence ---------------------------------------------------- #
    def save(self, results_dir: Path = RESULTS_DIR, figures_dir: Path = FIGURES_DIR,
             make_figures: bool = True) -> Dict[str, Path]:
        """Write CSVs, a JSON summary, and (optionally) the evaluation figures."""
        out_dir = Path(results_dir) / self.scenario
        # One figure directory per scenario so runs cannot overwrite each other;
        # the scenario-independent documentation diagrams also go to the top level.
        fig_dir = Path(figures_dir) / self.scenario
        out_dir.mkdir(parents=True, exist_ok=True)
        fig_dir.mkdir(parents=True, exist_ok=True)
        written: Dict[str, Path] = {}

        def _write(frame: pd.DataFrame, name: str) -> None:
            path = out_dir / name
            frame.to_csv(path, index=False)
            written[name] = path

        _write(self.telemetry, "telemetry.csv")
        _write(self.attacks, "attacks.csv")
        _write(self.profiles, "device_profiles.csv")
        _write(self.decision_log(), "decision_log.csv")
        _write(self.action_log(), "containment_actions.csv")
        _write(self.verification_log(), "verifications.csv")
        _write(self.learning_log(), "learning_log.csv")
        _write(self.summary_table(), "metrics_summary.csv")
        _write(self.comparison, "comparison_table.csv")
        per_attack = []
        for system, metrics in self.systems.items():
            if metrics.per_attack_type.empty:
                continue
            copy = metrics.per_attack_type.copy()
            copy.insert(0, "system", system)
            per_attack.append(copy)
        _write(pd.concat(per_attack, ignore_index=True) if per_attack
               else pd.DataFrame(columns=["attack_type"]), "per_attack_type.csv")
        episodes = []
        for system, metrics in self.systems.items():
            if metrics.episodes.empty:
                continue
            copy = metrics.episodes.copy()
            copy.insert(0, "system", system)
            episodes.append(copy)
        _write(pd.concat(episodes, ignore_index=True) if episodes
               else pd.DataFrame(columns=["attack_id"]), "episodes.csv")
        resources = pd.concat(
            [frame.assign(system=system) for system, frame in self.resource_logs.items()
             if not frame.empty],
            ignore_index=True,
        ) if any(not frame.empty for frame in self.resource_logs.values()) else pd.DataFrame()
        _write(resources, "resources.csv")

        config_path = out_dir / "experiment_config.json"
        config_path.write_text(json.dumps({
            "scenario": self.scenario, "seed": self.seed,
            "config": self.config.to_dict(), "metadata": self.metadata,
        }, indent=2, default=str), encoding="utf-8")
        written["experiment_config.json"] = config_path

        summary_path = out_dir / "experiment_summary.json"
        summary_path.write_text(json.dumps({
            "scenario": self.scenario,
            "seed": self.seed,
            "n_devices": int(self.telemetry["device_id"].nunique()),
            "n_steps": int(self.config.n_steps),
            "calibration_steps": int(self.config.calibration_steps),
            "evaluation_steps": int(self.config.n_steps - self.config.calibration_steps),
            "n_attacks": int(self.attacks.shape[0]),
            "attacks": self.attacks.to_dict(orient="records"),
            "systems": {system: {**m.summary, **m.host_measurements}
                        for system, m in self.systems.items()},
            "comparison_table": self.comparison.to_dict(orient="records"),
            "centralization": {
                system: {k: v for k, v in m.summary.items() if k.startswith("centralization_")}
                for system, m in self.systems.items()
            },
            "metadata": self.metadata,
        }, indent=2, default=str), encoding="utf-8")
        written["experiment_summary.json"] = summary_path

        if make_figures:
            from src.figures import make_all_figures  # local import: matplotlib is optional

            written.update(make_all_figures(self, fig_dir, docs_dir=Path(figures_dir)))
        return written


class ExperimentRunner:
    """Executes the multi-system experiment for one :class:`ExperimentConfig`."""

    def __init__(self, config: Optional[ExperimentConfig] = None, scenario: str = "main"):
        self.config = config or ExperimentConfig()
        self.scenario = scenario

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def run(
        self,
        systems: Sequence[str] = SYSTEMS,
        progress: Optional[Callable[[str, int, int], None]] = None,
    ) -> ExperimentResult:
        simulator = TrafficSimulator(self.config)
        telemetry, attacks = simulator.generate()
        telemetry = telemetry.copy()
        allowed = {d.device_id: d.allowed_protocols for d in simulator.devices}
        normal = {d.device_id: d.normal_protocol for d in simulator.devices}
        criticality = {d.device_id: d.criticality for d in simulator.devices}

        metrics_by_system: Dict[str, SystemMetrics] = {}
        decision_logs: Dict[str, pd.DataFrame] = {}
        learning_logs: Dict[str, pd.DataFrame] = {}
        resource_logs: Dict[str, pd.DataFrame] = {}

        for index, system in enumerate(systems, start=1):
            if progress is not None:
                progress(system, index, len(systems))
            metrics, decisions, learning, resources = self._run_system(
                system, telemetry, allowed, normal, criticality
            )
            metrics_by_system[system] = metrics
            decision_logs[system] = decisions
            learning_logs[system] = learning
            resource_logs[system] = resources

        metadata = {
            "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scenario": self.scenario,
            "note": ("All attack, telemetry and resource data are synthetic simulation "
                     "outputs. Resource figures labelled 'simulated_' come from the edge "
                     "resource model; figures labelled 'measured_' were measured on this "
                     "machine while the run executed."),
        }
        try:  # sklearn version is nice-to-have metadata, not a hard dependency here
            import sklearn

            metadata["scikit-learn"] = sklearn.__version__
        except Exception:  # pragma: no cover
            pass

        return ExperimentResult(
            scenario=self.scenario, config=self.config, seed=int(self.config.seed),
            telemetry=telemetry, attacks=attacks, profiles=simulator.profile_frame(),
            systems=metrics_by_system, decision_logs=decision_logs,
            learning_logs=learning_logs, resource_logs=resource_logs, metadata=metadata,
        )

    # ------------------------------------------------------------------ #
    # One system
    # ------------------------------------------------------------------ #
    def _run_system(
        self,
        system: str,
        telemetry: pd.DataFrame,
        allowed: Dict[str, Sequence[str]],
        normal: Dict[str, str],
        criticality: Dict[str, str],
    ) -> Tuple[SystemMetrics, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        cfg = self.config
        adaptive = system != SYSTEM_STATIC
        learning_enabled = system != SYSTEM_STATIC
        containment_enabled = system == SYSTEM_PROPOSED

        manager = AdaptiveBaselineManager(cfg.learning)
        manager.fit(telemetry, cfg.calibration_steps)
        extractor = FeatureExtractor(manager, allowed, normal)
        detector = AnomalyDetector(cfg.detector, seed=cfg.seed, adaptive=adaptive)
        assessor = ThreatAssessor(cfg.severity, criticality)

        # Calibration: benign window only, features only, no labels.
        calibration = telemetry[telemetry["step"] < int(cfg.calibration_steps)]
        calibration_vectors = [extractor.extract(row) for _, row in calibration.iterrows()]
        X_calibration = np.vstack([v.values for v in calibration_vectors])
        ids_calibration = [v.device_id for v in calibration_vectors]
        detector.calibrate(pd.DataFrame(X_calibration, columns=list(FEATURE_NAMES)), ids_calibration)

        controller = ClosedLoopController(
            cfg.containment, cfg.verification, ResponseVerifier(cfg.verification), criticality
        ) if containment_enabled else None
        resource_model = EdgeResourceModel(cfg.resources)

        device_ids: List[str] = sorted(telemetry["device_id"].unique().tolist())
        rows_by_step = {int(step): group for step, group in telemetry.groupby("step")}

        decision_rows: List[Dict[str, Any]] = []
        resource_rows: List[Dict[str, Any]] = []
        prev_anomalous_share = 0.0
        measured_peak_mb = float("nan")
        first_step = int(cfg.calibration_steps)

        wall_start = time.perf_counter()
        for step in range(first_step, int(cfg.n_steps)):
            tracemalloc_active = step == first_step
            if tracemalloc_active:
                tracemalloc.start()
            rows, anomalous_share, n_actions = self._process_step(
                step=step, telemetry_step=rows_by_step.get(step), device_ids=device_ids,
                extractor=extractor, detector=detector, assessor=assessor, manager=manager,
                controller=controller, learning_enabled=learning_enabled,
                anomaly_share_prev=prev_anomalous_share, decision_rows=decision_rows,
                resource_model=resource_model,
            )
            prev_anomalous_share = anomalous_share
            if tracemalloc_active:
                _, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                measured_peak_mb = float(peak) / 1_000_000.0
            resource_rows.append(
                resource_model.finish_step(step, len(device_ids), n_actions).to_dict()
            )
        wall_seconds = time.perf_counter() - wall_start
        n_online_steps = max(1, int(cfg.n_steps) - first_step)

        verifier = controller.verifier if controller is not None else ResponseVerifier(cfg.verification)
        verifier.close_all_inconclusive(int(cfg.n_steps) - 1)
        actions = controller.action_log_frame() if controller is not None else pd.DataFrame(
            columns=["step", "device_id", "kind", "action", "level", "previous_level",
                     "severity", "score", "threshold", "reason"]
        )
        verifications = verifier.results_frame()
        recovery = controller.recovery_frame() if controller is not None else pd.DataFrame(
            columns=["step", "device_id", "previous_action", "recovery_time_steps"]
        )
        decisions = pd.DataFrame(decision_rows)
        decisions.attrs["step_seconds"] = float(cfg.step_seconds)
        resources = pd.DataFrame(resource_rows)
        learning_log = manager.update_log_frame()

        host_measurements = {
            "measured_wall_clock_seconds": round(wall_seconds, 4),
            "measured_per_step_ms": round(1000.0 * wall_seconds / n_online_steps, 4),
            "measured_peak_alloc_mb": round(measured_peak_mb, 4) if np.isfinite(measured_peak_mb) else None,
            "measured_online_steps": n_online_steps,
            "measured_n_observations": int(decisions.shape[0]),
            "measured_label": "measured on the host machine (time.perf_counter / tracemalloc)",
        }
        metrics = summarize_system(
            system=system, decisions=decisions, telemetry=telemetry, actions=actions,
            verification=verifications, recovery=recovery, resources=resources,
            host_measurements=host_measurements,
            learning_blocked={
                "learning_updates_accepted": manager.status()["n_updates_accepted"],
                "learning_blocked_anomalous": manager.status()["blocked_updates_anomalous"],
                "learning_blocked_frozen": manager.status()["blocked_updates_frozen"],
                "learning_blocked_containment": manager.status()["blocked_updates_containment"],
                "learning_drift_releases": manager.status()["drift_releases"],
                "detector_threshold_final": float(detector.threshold),
                "detector_threshold_calibration": float(detector.calibration_threshold),
                "detector_freezes": len(detector.freeze_events),
            },
        )
        return metrics, decisions, learning_log, resources

    # ------------------------------------------------------------------ #
    # One step of one system
    # ------------------------------------------------------------------ #
    def _process_step(
        self,
        step: int,
        telemetry_step: Optional[pd.DataFrame],
        device_ids: List[str],
        extractor: FeatureExtractor,
        detector: AnomalyDetector,
        assessor: ThreatAssessor,
        manager: AdaptiveBaselineManager,
        controller: Optional[ClosedLoopController],
        learning_enabled: bool,
        anomaly_share_prev: float,
        decision_rows: List[Dict[str, Any]],
        resource_model: EdgeResourceModel,
    ) -> Tuple[List[Dict[str, Any]], float, int]:
        if telemetry_step is None:
            return [], anomaly_share_prev, 0
        cfg = self.config
        raw_rows = {str(r["device_id"]): r for _, r in telemetry_step.iterrows()}
        resource_model.begin_step()

        # (a) allowances decided by the previous step ---------------------------------
        allowances = {d: (controller.allowance(d) if controller is not None else 1.0)
                      for d in device_ids}
        available = {d: allowances[d] > ALLOWANCE_FLOOR for d in device_ids}

        # (b) observed telemetry (containment effect) and detection view --------------
        detection_rows: Dict[str, Dict[str, float]] = {}
        observed_rows: Dict[str, Dict[str, float]] = {}
        scoring_ids: List[str] = []
        for device_id in device_ids:
            raw = raw_rows.get(device_id)
            if raw is None:
                continue
            allowance = float(max(allowances[device_id], ALLOWANCE_FLOOR))
            observed = self._apply_allowance(raw, allowances[device_id])
            detection = self._apply_allowance(raw, 1.0)  # inverse view: allowance-normalised
            detection["packet_rate"] = observed["packet_rate"] / allowance
            detection["bytes_per_sec"] = observed["bytes_per_sec"] / allowance
            detection["conn_count"] = observed["conn_count"] / min(1.0, allowance + 0.15)
            cpu_factor = max(1.0 - CONTAINMENT_CPU_EFFECT * (1.0 - allowance), ALLOWANCE_FLOOR)
            detection["cpu_usage"] = observed["cpu_usage"] / cpu_factor
            detection["memory_usage"] = observed["memory_usage"] / cpu_factor
            observed_rows[device_id] = observed
            detection_rows[device_id] = detection
            if available[device_id]:
                scoring_ids.append(device_id)
            resource_model.observe_record(raw, len(device_ids))

        # (c) feature extraction (detection view) ------------------------------------
        detection_features: Dict[str, Dict[str, float]] = {}
        if scoring_ids:
            X = np.vstack([
                extractor.extract(detection_rows[d], step=step).values for d in scoring_ids
            ])
            results = detector.process_batch(step, scoring_ids, X)
            for i, device_id in enumerate(scoring_ids):
                detection_features[device_id] = {
                    name: float(X[i, j]) for j, name in enumerate(FEATURE_NAMES)
                }
        else:
            results = []

        scores_by_device: Dict[str, ScoreResult] = {r.device_id: r for r in results}
        observed_features: Dict[str, Dict[str, float]] = {}
        fleet_bytes = float(sum(observed_rows[d]["bytes_per_sec"] for d in observed_rows)) or 1.0

        step_rows: List[Dict[str, Any]] = []
        n_actions = 0
        n_anomalous = 0
        for device_id in device_ids:
            raw = raw_rows.get(device_id)
            if raw is None:
                continue
            score_result = scores_by_device.get(device_id)
            if score_result is not None and score_result.is_anomalous:
                n_anomalous += 1

            # (d) severity (uses only label-free inputs) --------------------------
            severity = 0.0
            factors = {"magnitude": 0.0, "confidence": 0.0, "criticality": 0.0, "context": 0.0}
            if score_result is not None:
                traffic_share = float(observed_rows[device_id]["bytes_per_sec"]) / fleet_bytes
                assessment = assessor.assess(
                    step=step, device_id=device_id, score=score_result.score,
                    threshold=score_result.threshold, stat_score=score_result.stat_score,
                    if_score=score_result.if_score,
                    anomalous_share=float(anomaly_share_prev), traffic_share=traffic_share,
                )
                severity = assessment.severity
                factors = {"magnitude": assessment.magnitude, "confidence": assessment.confidence,
                           "criticality": assessment.criticality, "context": assessment.context}

            # (e) observed feature vector of offending features (verification view) -
            indicators: Dict[str, float] = {}
            if score_result is not None:
                if device_id not in observed_features:
                    observed_features[device_id] = extractor.extract(
                        observed_rows[device_id], step=step
                    ).as_dict()
                top = detector.top_contributions(score_result, k=3)
                indicators = {
                    name: float(observed_features[device_id].get(name, 0.0))
                    for name, contribution in top
                    if contribution >= INDICATOR_CONTRIBUTION_FLOOR
                }

            # (f) containment ------------------------------------------------------
            events: List[Any] = []
            if controller is not None:
                cpu_used_estimate = (
                    cfg.resources.feature_extract_cpu_per_device * len(device_ids)
                    + cfg.resources.detect_cpu_per_device * len(device_ids)
                    + cfg.resources.policy_cpu_per_device * len(device_ids)
                )
                cpu_free = max(0.0, cfg.resources.cpu_capacity - cpu_used_estimate)
                events = controller.process(
                    step=step, device_id=device_id, score_result=score_result, severity=severity,
                    observed_indicators=indicators, telemetry_available=available[device_id],
                    cpu_free_units=cpu_free,
                )
                n_actions += sum(1 for e in events if e.level >= 1 and e.kind in ("APPLY", "ESCALATE"))

            # (g) adaptive learning (only for the adaptive systems, never under
            #     the gateway's own containment, never for anomalies) --------------
            if learning_enabled and score_result is not None:
                manager.step(
                    step=step, device_id=device_id, observation=detection_rows[device_id],
                    score=score_result.score, threshold=score_result.threshold,
                    is_anomalous=score_result.is_anomalous,
                    features=detection_features.get(device_id),
                    traffic_allowance=(controller.allowance(device_id)
                                       if controller is not None else 1.0),
                )

            action_name = controller.action_name(device_id) if controller is not None else "OBSERVE"
            step_rows.append({
                "step": int(step), "device_id": device_id,
                "scored": int(score_result is not None),
                "anomaly_score": round(float(score_result.score), 6) if score_result else None,
                "stat_score": round(float(score_result.stat_score), 6) if score_result else None,
                "if_score": round(float(score_result.if_score), 6) if score_result else None,
                "threshold": round(float(score_result.threshold), 6) if score_result else None,
                "is_anomalous": int(score_result.is_anomalous) if score_result else 0,
                "frozen": int(score_result.frozen) if score_result else 0,
                "severity": round(float(severity), 4),
                "magnitude": round(factors["magnitude"], 6),
                "confidence": round(factors["confidence"], 6),
                "criticality_factor": round(factors["criticality"], 6),
                "context_factor": round(factors["context"], 6),
                "allowance": round(float(allowances[device_id]), 4),
                "action": action_name,
                "observed_bytes_per_sec": round(float(observed_rows[device_id]["bytes_per_sec"]), 2),
                "observed_packet_rate": round(float(observed_rows[device_id]["packet_rate"]), 4),
                "observed_cpu_usage": round(float(observed_rows[device_id]["cpu_usage"]), 4),
                "observed_memory_usage": round(float(observed_rows[device_id]["memory_usage"]), 4),
                "ground_truth": int(raw["ground_truth"]),
                "attack_type": raw["attack_type"],
                "attack_held_out": bool(raw["attack_held_out"]),
                "benign_shift": bool(raw["benign_shift"]),
            })
        decision_rows.extend(step_rows)
        anomalous_share = float(n_anomalous) / max(len(device_ids), 1)
        return step_rows, anomalous_share, n_actions

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _apply_allowance(raw: pd.Series, allowance: float) -> Dict[str, float]:
        """
        Simulated containment effect on the *observed* telemetry of a device.

        Volume features scale with the allowance; the attack-driven part of the
        CPU/memory load is reduced by ``CONTAINMENT_CPU_EFFECT``. A fully
        quarantined device (allowance 0) contributes no volume telemetry.
        """
        allowance = float(np.clip(allowance, 0.0, 1.0))
        conns_factor = min(1.0, allowance + 0.15)
        cpu_factor = max(1.0 - CONTAINMENT_CPU_EFFECT * (1.0 - allowance), ALLOWANCE_FLOOR)
        return {
            "packet_rate": float(raw["packet_rate"]) * allowance,
            "bytes_per_sec": float(raw["bytes_per_sec"]) * allowance,
            "conn_count": float(raw["conn_count"]) * conns_factor,
            "failed_auth_attempts": float(raw["failed_auth_attempts"]),
            "dest_diversity": float(raw["dest_diversity"]),
            "cpu_usage": float(raw["cpu_usage"]) * cpu_factor,
            "memory_usage": float(raw["memory_usage"]) * cpu_factor,
            "protocol": str(raw["protocol"]),
            "device_id": str(raw["device_id"]),
            "step": int(raw["step"]),
        }


# --------------------------------------------------------------------------- #
# Suite helper + CLI
# --------------------------------------------------------------------------- #
def run_suite(
    configs: Optional[Dict[str, ExperimentConfig]] = None,
    results_dir: Path = RESULTS_DIR,
    figures_dir: Path = FIGURES_DIR,
    make_figures: bool = True,
    progress: Optional[Callable[[str, str, int, int], None]] = None,
) -> Dict[str, ExperimentResult]:
    """Run several scenarios and persist their results and figures."""
    if configs is None:
        configs = {
            "main": ExperimentConfig(),
            "low_resource": ExperimentConfig.low_resource(),
            "starved_edge": ExperimentConfig.starved_edge(),
            "seed_2027": ExperimentConfig(seed=2027, n_steps=280),
        }
    results: Dict[str, ExperimentResult] = {}
    for scenario, config in configs.items():
        runner = ExperimentRunner(config, scenario=scenario)
        result = runner.run(
            progress=(None if progress is None else
                      (lambda system, i, n: progress(scenario, system, i, n)))
        )
        result.save(results_dir=results_dir, figures_dir=figures_dir, make_figures=make_figures)
        results[scenario] = result
    return results


def main(argv: Optional[Sequence[str]] = None) -> int:  # pragma: no cover - CLI entry point
    argv = list(sys.argv[1:] if argv is None else argv)
    quick = "--quick" in argv
    no_figures = "--no-figures" in argv
    if quick:
        configs = {
            "main": ExperimentConfig(n_devices=24, n_steps=200, calibration_steps=80),
            "low_resource": ExperimentConfig.low_resource(n_devices=24, n_steps=200,
                                                          calibration_steps=80),
        }
    else:
        configs = {
            "main": ExperimentConfig(),
            "low_resource": ExperimentConfig.low_resource(),
            "starved_edge": ExperimentConfig.starved_edge(),
            "seed_2027": ExperimentConfig(seed=2027, n_steps=280),
        }

    def progress(scenario: str, system: str, index: int, total: int) -> None:
        print(f"[{scenario}] {index}/{total} running {SYSTEM_LABELS.get(system, system)}",
              flush=True)

    results = run_suite(configs, make_figures=not no_figures, progress=progress)
    for scenario, result in results.items():
        print(f"\n=== {scenario} (seed={result.seed}) ===")
        print(result.comparison.to_string(index=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
