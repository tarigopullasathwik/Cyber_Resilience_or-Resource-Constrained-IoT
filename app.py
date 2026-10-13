"""
Interactive Streamlit dashboard for the adaptive edge cyber-resilience simulation.

Run with:

    python -m streamlit run app.py

The dashboard reads the CSV/JSON artifacts written by
``src/experiment_runner.py`` (``results/<scenario>/``) and the figures written to
``figures/``. If no results exist yet it can run the experiment itself.

Everything shown is synthetic: the devices, the traffic, the attacks and the
containment actions are simulated inside this application. Nothing is scanned,
blocked or contacted. Across the whole UI four kinds of information are labelled
explicitly, because mixing them up is the most common way to over-claim results:

* **ground truth (synthetic)** - the injected attack label, used for evaluation only
* **model prediction** - detector output (anomaly score, flag) and severity
* **simulated response** - containment actions and their modelled effects
* **measured** - wall-clock time and Python allocation measured on this machine
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from config import (  # noqa: E402
    FIGURES_DIR,
    RESULTS_DIR,
    SYSTEM_LABELS,
    SYSTEMS,
    ContainmentConfig,
    ExperimentConfig,
)

#: Response level of each action name (0 = OBSERVE ... 3 = QUARANTINE). The
#: decision log stores the action *name*, the actions table stores the level.
LEVEL_OF_ACTION = {name: index for index, name in enumerate(ContainmentConfig().levels)}

SCOPE_NOTE = (
    "**Research prototype / simulation only.** Devices, traffic, attacks and "
    "containment actions are synthetic objects inside this Python process. Nothing "
    "here scans a network, changes a firewall, or disrupts a real device, and no "
    "result below is evidence of real-world security effectiveness."
)


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def discover_scenarios(results_dir: Path = RESULTS_DIR) -> list[str]:
    if not results_dir.exists():
        return []
    return sorted(
        child.name for child in results_dir.iterdir()
        if child.is_dir() and (child / "experiment_summary.json").exists()
    )


def _signature(directory: Path) -> str:
    files = sorted(directory.glob("*.csv")) + list(directory.glob("*.json"))
    return "|".join(f"{f.name}:{f.stat().st_mtime_ns}" for f in files)


@st.cache_data(show_spinner=False)
def load_scenario(scenario: str, signature: str) -> dict:
    directory = RESULTS_DIR / scenario
    tables: dict[str, pd.DataFrame] = {}
    for name in ("telemetry", "attacks", "device_profiles", "decision_log", "containment_actions",
                 "verifications", "learning_log", "metrics_summary", "comparison_table",
                 "per_attack_type", "episodes", "resources"):
        path = directory / f"{name}.csv"
        if path.exists():
            try:
                tables[name] = pd.read_csv(path)
            except pd.errors.EmptyDataError:
                tables[name] = pd.DataFrame()
    summary = {}
    summary_path = directory / "experiment_summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    config = {}
    config_path = directory / "experiment_config.json"
    if config_path.exists():
        config = json.loads(config_path.read_text(encoding="utf-8"))
    return {"tables": tables, "summary": summary, "config": config,
            "dir": str(directory), "scenario": scenario}


def run_experiment_now(scenario: str, quick: bool, with_figures: bool) -> None:
    from src.experiment_runner import ExperimentRunner

    if quick:
        config = ExperimentConfig(n_devices=24, n_steps=200, calibration_steps=80)
    else:
        config = ExperimentConfig()
    progress = st.progress(0.0, text="starting ...")

    def callback(system: str, index: int, total: int) -> None:
        progress.progress((index - 1) / max(total, 1),
                          text=f"running {SYSTEM_LABELS.get(system, system)} ({index}/{total})")

    runner = ExperimentRunner(config, scenario=scenario)
    result = runner.run(progress=callback)
    progress.progress(0.9, text="saving results and figures ...")
    result.save(results_dir=RESULTS_DIR, figures_dir=FIGURES_DIR, make_figures=with_figures)
    progress.progress(1.0, text="done")
    st.cache_data.clear()
    st.success(f"Experiment finished and saved to results/{scenario}/")
    st.rerun()


# --------------------------------------------------------------------------- #
# Small chart helpers (matplotlib -> streamlit)
# --------------------------------------------------------------------------- #
def show_figure(fig: plt.Figure) -> None:
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)


def _fmt_value(value: object) -> str:
    """
    Format a summary value for display.

    Summary dictionaries mix floats with strings (for example
    ``missed_attack_types`` is a comma-joined string), and a mixed-type object
    column cannot be sent to the frontend as an Arrow table. Formatting every
    value to text keeps the tables renderable and predictable.
    """
    if value is None:
        return ""
    if isinstance(value, float):
        if not np.isfinite(value):
            return "n/a"
        return f"{value:.4g}"
    return str(value)


def timeline_chart(decisions: pd.DataFrame, device_id: str, show_severity: bool = True):
    frame = decisions[decisions["device_id"] == device_id].sort_values("step")
    if frame.empty:
        st.info("No rows for this device in the selected scenario.")
        return None
    fig, ax = plt.subplots(figsize=(10, 3.0))
    ax.plot(frame["step"], frame["anomaly_score"], linewidth=1.2, color="#1f3b73",
            label="anomaly score (model)")
    ax.plot(frame["step"], frame["threshold"], linewidth=1.2, linestyle="--", color="#e08a1e",
            label="adaptive threshold")
    attacks = frame[frame["ground_truth"] == 1]
    for step in attacks["step"]:
        ax.axvspan(step - 0.5, step + 0.5, color="#d9534f", alpha=0.18)
    if show_severity:
        ax2 = ax.twinx()
        ax2.plot(frame["step"], frame["severity"], linewidth=0.9, color="#7b1fa2", alpha=0.7,
                 label="severity (0-100)")
        ax2.set_ylabel("severity")
        ax2.set_ylim(0, 105)
    ax.set_xlabel("simulation step")
    ax.set_ylabel("fused anomaly score")
    ax.set_title(f"{device_id} - red bands = synthetic ground-truth attack steps", fontsize=10)
    ax.legend(fontsize=7.5, loc="upper left")
    ax.grid(alpha=0.25)
    return fig


def stream_chart(decisions: pd.DataFrame, actions: pd.DataFrame, step_from: int, step_to: int):
    window = decisions[(decisions["step"] >= step_from) & (decisions["step"] <= step_to)]
    if window.empty:
        return None
    fig, axes = plt.subplots(2, 1, figsize=(10, 4.6), sharex=True)
    axes[0].scatter(window["step"], window["anomaly_score"], s=10,
                    c=np.where(window["ground_truth"] == 1, "#c0504d", "#4f81bd"), alpha=0.8)
    axes[0].plot(window.groupby("step")["threshold"].median().index,
                 window.groupby("step")["threshold"].median().values,
                 color="#e08a1e", linestyle="--", linewidth=1.0, label="threshold (median)")
    axes[0].set_ylabel("score")
    axes[0].legend(fontsize=7.5)
    axes[0].grid(alpha=0.25)
    acted = pd.DataFrame(columns=["step", "device_id"])
    if not actions.empty:
        acted = actions[actions["system"] == window["system"].iloc[0]] \
            if "system" in actions.columns else actions
        acted = acted[(acted["level"] >= 1) & (acted["step"] >= step_from)
                      & (acted["step"] <= step_to)]
    if not acted.empty:
        axes[1].scatter(acted["step"], acted["device_id"], s=18, color="#2e7d32")
    axes[1].set_ylabel("device acted on")
    axes[1].set_xlabel("simulation step")
    axes[1].set_title("containment actions in this window (simulated)", fontsize=9)
    axes[1].grid(alpha=0.25)
    fig.suptitle(f"Event stream, steps {step_from}-{step_to} "
                 "(red = synthetic ground truth attack)", fontsize=10)
    return fig


def confusion_chart(summary: dict, system: str):
    if not summary:
        return None
    fig, ax = plt.subplots(figsize=(3.4, 3.0))
    matrix = np.array([[summary["tn"], summary["fp"]], [summary["fn"], summary["tp"]]], dtype=float)
    ax.imshow(matrix, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{int(matrix[i, j])}", ha="center", va="center", fontsize=11)
    ax.set_xticks([0, 1], ["pred benign", "pred anomaly"])
    ax.set_yticks([0, 1], ["true benign", "true attack"])
    ax.set_title(SYSTEM_LABELS.get(system, system), fontsize=8.5)
    return fig


def metric_bar(rows: list[dict], label_key: str, value_key: str, title: str, color="#4f81bd"):
    if not rows:
        return None
    fig, ax = plt.subplots(figsize=(9, 3.0))
    labels = [str(row[label_key]) for row in rows]
    values = [float(row[value_key] or 0.0) for row in rows]
    bars = ax.bar(labels, values, color=color)
    ax.bar_label(bars, fmt="%.3g", fontsize=7)
    ax.set_title(title, fontsize=10)
    ax.tick_params(axis="x", labelrotation=20, labelsize=8)
    ax.grid(alpha=0.25, axis="y")
    return fig


# --------------------------------------------------------------------------- #
# Pages
# --------------------------------------------------------------------------- #
def render_sidebar(data_by_scenario: dict[str, dict]) -> tuple[str, dict]:
    st.sidebar.header("Experiment")
    scenario = st.sidebar.selectbox("Scenario", sorted(data_by_scenario.keys()))
    data = data_by_scenario[scenario]
    summary = data.get("summary", {})
    config = data.get("config", {})
    st.sidebar.caption(f"seed = {summary.get('seed', config.get('seed', 'n/a'))}")
    st.sidebar.caption(f"devices = {summary.get('n_devices', 'n/a')}")
    st.sidebar.caption(f"steps = {summary.get('n_steps', 'n/a')} "
                       f"(calibration {summary.get('calibration_steps', 'n/a')})")
    st.sidebar.caption(f"injected attacks = {summary.get('n_attacks', 'n/a')}")
    st.sidebar.divider()
    st.sidebar.subheader("Run your own")
    quick = st.sidebar.checkbox("quick run (200 steps)", value=True)
    with_figures = st.sidebar.checkbox("generate figures", value=False)
    if st.sidebar.button("Run experiment now"):
        run_experiment_now(scenario, quick, with_figures)
    st.sidebar.divider()
    st.sidebar.download_button(
        "download experiment summary (JSON)",
        data=json.dumps(summary, indent=2, default=str),
        file_name=f"{scenario}_experiment_summary.json",
        mime="application/json",
    )
    return scenario, data


def page_overview(data: dict) -> None:
    tables = data["tables"]
    summary = data.get("summary", {})
    comparison = tables.get("comparison_table", pd.DataFrame())
    decisions = tables.get("decision_log", pd.DataFrame())

    st.subheader("Headline numbers")
    proposed = summary.get("systems", {}).get("S3_adaptive_closed_loop", {})
    columns = st.columns(4)
    columns[0].metric("Simulated IoT devices", summary.get("n_devices", "n/a"))
    if not decisions.empty:
        benign_steps = int((decisions["ground_truth"] == 0).sum())
        attack_steps = int((decisions["ground_truth"] == 1).sum())
        columns[1].metric("Benign device-steps (synthetic truth)", f"{benign_steps:,}")
        columns[2].metric("Anomalous device-steps (synthetic truth)", f"{attack_steps:,}")
    columns[3].metric("Injected attacks", summary.get("n_attacks", "n/a"))

    columns = st.columns(4)
    columns[0].metric("Detected episodes (S3)", proposed.get("n_episodes", 0) and
                      int(proposed.get("n_episodes", 0) - proposed.get("n_missed_attacks", 0)))
    columns[1].metric("Missed attacks (S3)", proposed.get("n_missed_attacks", "n/a"))
    columns[2].metric("Containment success rate (S3)",
                      f"{float(proposed.get('containment_success_rate') or 0):.2f}"
                      if proposed.get("containment_success_rate") is not None else "n/a")
    columns[3].metric("Unnecessary quarantines (S3)", proposed.get("unnecessary_quarantines", "n/a"))

    st.caption("All detection/containment figures are computed from the logs of this run; the "
               "ground-truth column is synthetic and is used for evaluation only.")

    st.subheader("System comparison")
    if not comparison.empty:
        st.dataframe(comparison, width="stretch", hide_index=True)
        st.caption("S1 = static threshold (no adaptation, no containment); "
                   "S2 = adaptive detection without containment; "
                   "S3 = proposed adaptive closed loop.")
    else:
        st.info("No comparison table found - run the experiment first.")

    st.subheader("Confusion matrices (device-steps, post-calibration window)")
    columns = st.columns(len(SYSTEMS))
    for column, system in zip(columns, SYSTEMS):
        system_summary = summary.get("systems", {}).get(system, {})
        with column:
            fig = confusion_chart(system_summary, system)
            if fig is not None:
                show_figure(fig)
            else:
                st.write("n/a")


def page_stream(data: dict) -> None:
    decisions = data["tables"].get("decision_log", pd.DataFrame())
    if decisions.empty:
        st.info("No decision log found - run the experiment first.")
        return
    system = st.selectbox("System", list(SYSTEMS),
                          format_func=lambda s: SYSTEM_LABELS.get(s, s), index=2)
    frame = decisions[decisions["system"] == system] if "system" in decisions.columns else decisions
    actions = data["tables"].get("containment_actions", pd.DataFrame())
    step_min, step_max = int(frame["step"].min()), int(frame["step"].max())
    window = st.slider("Step window", step_min, step_max, (step_min, min(step_min + 20, step_max)))
    mode = st.radio("Display", ["step window", "single step", "device timeline"], horizontal=True)

    if mode == "step window":
        fig = stream_chart(frame, actions, window[0], window[1])
        if fig is not None:
            show_figure(fig)
        rows = frame[(frame["step"] >= window[0]) & (frame["step"] <= window[1])]
        st.caption("Event stream (synthetic ground truth shown for evaluation)")
        st.dataframe(rows.sort_values(["step", "device_id"]), width="stretch",
                     hide_index=True, height=320)
    elif mode == "single step":
        step = st.slider("Step", step_min, step_max, window[0], key="single_step")
        rows = frame[frame["step"] == step].sort_values("device_id")
        st.dataframe(rows, width="stretch", hide_index=True)
    else:
        device = st.selectbox("Device", sorted(frame["device_id"].unique()))
        fig = timeline_chart(frame, device)
        if fig is not None:
            show_figure(fig)
        st.caption("Blue = model prediction, orange = adaptive threshold, red bands = synthetic "
                   "ground-truth attack steps, purple = severity.")


def page_detection(data: dict) -> None:
    decisions = data["tables"].get("decision_log", pd.DataFrame())
    summary = data.get("summary", {})
    if decisions.empty:
        st.info("No decision log found - run the experiment first.")
        return
    system = st.selectbox("System", list(SYSTEMS),
                          format_func=lambda s: SYSTEM_LABELS.get(s, s), index=2, key="det_system")
    frame = decisions[decisions["system"] == system]

    left, right = st.columns(2)
    with left:
        st.markdown("**Anomaly score vs adaptive threshold (per step, median over devices)**")
        grouped = frame.groupby("step")[["anomaly_score", "threshold"]].median()
        fig, ax = plt.subplots(figsize=(5.6, 3.2))
        ax.plot(grouped.index, grouped["anomaly_score"], label="median score", color="#1f3b73")
        ax.plot(grouped.index, grouped["threshold"], label="median threshold", color="#e08a1e",
                linestyle="--")
        ax.set_xlabel("step")
        ax.set_ylabel("score")
        ax.legend(fontsize=7.5)
        ax.grid(alpha=0.25)
        show_figure(fig)
    with right:
        st.markdown("**Score distribution by synthetic ground truth**")
        fig, ax = plt.subplots(figsize=(5.6, 3.2))
        for label, color, name in ((0, "#4f81bd", "benign"), (1, "#c0504d", "attack")):
            values = frame[frame["ground_truth"] == label]["anomaly_score"].dropna()
            if not values.empty:
                ax.hist(values, bins=40, alpha=0.7, color=color, label=f"{name} (n={values.size})")
        ax.set_xlabel("fused anomaly score")
        ax.legend(fontsize=7.5)
        ax.grid(alpha=0.25)
        show_figure(fig)

    st.subheader("Detection performance")
    table = pd.DataFrame([
        {"metric": key,
         "value": _fmt_value(summary.get("systems", {}).get(system, {}).get(key))}
        for key in ("accuracy", "precision", "recall", "f1", "false_positive_rate",
                    "benign_stable_fpr", "benign_shift_fpr", "known_attack_pointwise_recall",
                    "held_out_pointwise_recall", "detection_latency_mean_steps",
                    "detection_latency_mean_seconds", "n_missed_attacks", "missed_attack_types",
                    "drift_adaptation_median_steps", "n_steps_not_scored")
    ])
    st.dataframe(table, width="stretch", hide_index=True)
    st.caption("Detection latency is reported in simulation steps and in seconds "
               "(1 step = 5 simulated seconds). 'Held-out' recall covers attack patterns that were "
               "never used while the detector or its thresholds were set up; it is a proxy for "
               "unknown-attack sensitivity, not proof of zero-day generalisation.")

    per_attack = data["tables"].get("per_attack_type", pd.DataFrame())
    if not per_attack.empty:
        st.subheader("Detection per attack pattern")
        subset = per_attack[per_attack["system"] == system] if "system" in per_attack.columns \
            else per_attack
        st.dataframe(subset, width="stretch", hide_index=True)


def page_threat_and_containment(data: dict) -> None:
    decisions = data["tables"].get("decision_log", pd.DataFrame())
    actions = data["tables"].get("containment_actions", pd.DataFrame())
    verifications = data["tables"].get("verifications", pd.DataFrame())
    summary = data.get("summary", {})
    if decisions.empty:
        st.info("No decision log found - run the experiment first.")
        return

    proposed = system_frame(decisions, "S3_adaptive_closed_loop")
    st.subheader("Device risk / severity table (latest step of the proposed system)")
    if not proposed.empty:
        latest = int(proposed["step"].max())
        table = (proposed[proposed["step"] == latest]
                 [["device_id", "anomaly_score", "threshold", "severity", "criticality_factor",
                   "context_factor", "action", "allowance", "ground_truth", "attack_type",
                   "benign_shift"]]
                 .sort_values("severity", ascending=False))
        st.dataframe(table, width="stretch", hide_index=True)
        st.caption("`severity` is the model's operational risk score (0-100); `ground_truth` and "
                   "`attack_type` are the synthetic labels and are never used by the policy.")
        top = (proposed.groupby("device_id")
               .agg(max_severity=("severity", "max"), triggers=("is_anomalous", "sum"),
                    actions=("action", lambda s: ", ".join(sorted(set(s)))))
               .sort_values("max_severity", ascending=False).reset_index())
        st.dataframe(top, width="stretch", hide_index=True)

    st.subheader("Containment actions (simulated)")
    if not actions.empty:
        subset = actions[actions["system"] == "S3_adaptive_closed_loop"] \
            if "system" in actions.columns else actions
        applied = subset[subset["level"] >= 1]
        if not applied.empty:
            counts = applied["action"].value_counts().reset_index()
            counts.columns = ["action", "count"]
            fig = metric_bar(counts.to_dict("records"), "action", "count",
                             "Actions applied by the proposed system", "#2e7d32")
            if fig is not None:
                show_figure(fig)
        st.dataframe(subset.sort_values("step"), width="stretch", hide_index=True,
                     height=300)
    else:
        st.info("No containment actions were recorded for this scenario.")

    st.subheader("Response verification outcomes")
    if not verifications.empty:
        subset = verifications[verifications["system"] == "S3_adaptive_closed_loop"] \
            if "system" in verifications.columns else verifications
        if not subset.empty:
            counts = subset["outcome"].value_counts().reset_index()
            counts.columns = ["outcome", "count"]
            fig = metric_bar(counts.to_dict("records"), "outcome", "count",
                             "Verification of containment effectiveness", "#8e44ad")
            if fig is not None:
                show_figure(fig)
        st.dataframe(subset, width="stretch", hide_index=True, height=260)
        st.caption("SUCCESS = score and/or the offending indicators fell after the action; "
                   "FAILURE = no measurable effect over a full observation window; "
                   "INCONCLUSIVE = too few observations (e.g. the run ended).")
    else:
        st.info("No verification windows were recorded.")

    proposed_summary = summary.get("systems", {}).get("S3_adaptive_closed_loop", {})
    st.subheader("Recovery and isolation safeguards")
    st.dataframe(pd.DataFrame([
        {"metric": "containment actions",
         "value": _fmt_value(proposed_summary.get("n_containment_actions"))},
        {"metric": "verifications",
         "value": _fmt_value((proposed_summary.get("containment_success") or 0)
                             + (proposed_summary.get("containment_failure") or 0)
                             + (proposed_summary.get("containment_inconclusive") or 0))},
        {"metric": "recoveries", "value": _fmt_value(proposed_summary.get("recovery_n"))},
        {"metric": "mean recovery time (steps)",
         "value": _fmt_value(proposed_summary.get("recovery_time_mean_steps"))},
        {"metric": "quarantine actions",
         "value": _fmt_value(proposed_summary.get("quarantine_actions"))},
        {"metric": "unnecessary quarantines (benign ground truth)",
         "value": _fmt_value(proposed_summary.get("unnecessary_quarantines"))},
        {"metric": "response latency (steps, mean)",
         "value": _fmt_value(proposed_summary.get("response_latency_mean_steps"))},
    ]), width="stretch", hide_index=True)


def system_frame(decisions: pd.DataFrame, system: str) -> pd.DataFrame:
    if "system" not in decisions.columns:
        return decisions
    return decisions[decisions["system"] == system]


def page_resources(data: dict) -> None:
    resources = data["tables"].get("resources", pd.DataFrame())
    summary = data.get("summary", {})
    st.subheader("Simulated edge resource model")
    st.caption("These numbers come from the edge resource *model* (CPU/memory envelope, uplink "
               "and latency estimates). They are simulation outputs, not machine measurements.")
    if not resources.empty:
        subset = resources[resources["system"] == "S3_adaptive_closed_loop"] \
            if "system" in resources.columns else resources
        columns = st.columns(2)
        with columns[0]:
            fig, ax = plt.subplots(figsize=(5.4, 3.0))
            ax.plot(subset["step"], subset["cpu_utilisation"], color="#2e7d32")
            ax.axhline(1.0, color="#c0504d", linestyle="--", linewidth=0.9, label="capacity")
            ax.set_title("simulated edge CPU utilisation", fontsize=9.5)
            ax.set_xlabel("step")
            ax.legend(fontsize=7.5)
            ax.grid(alpha=0.25)
            show_figure(fig)
        with columns[1]:
            fig, ax = plt.subplots(figsize=(5.4, 3.0))
            ax.plot(subset["step"], subset["memory_used_mb"], color="#1f3b73")
            ax.set_title("simulated edge memory footprint (MB)", fontsize=9.5)
            ax.set_xlabel("step")
            ax.grid(alpha=0.25)
            show_figure(fig)
        columns = st.columns(2)
        with columns[0]:
            fig, ax = plt.subplots(figsize=(5.4, 3.0))
            ax.plot(subset["step"], subset["uplink_kbps"], label="edge summary uplink")
            ax.plot(subset["step"], subset["cloud_uplink_kbps"],
                    label="centralised raw-telemetry uplink")
            ax.set_yscale("log")
            ax.set_title("simulated uplink per step (log scale)", fontsize=9.5)
            ax.set_xlabel("step")
            ax.legend(fontsize=7.5)
            ax.grid(alpha=0.25)
            show_figure(fig)
        with columns[1]:
            fig, ax = plt.subplots(figsize=(5.4, 3.0))
            ax.plot(subset["step"], subset["edge_latency_ms"], label="edge (local)")
            ax.plot(subset["step"], subset["cloud_latency_ms"], label="centralised")
            ax.set_title("simulated processing latency (ms)", fontsize=9.5)
            ax.set_xlabel("step")
            ax.legend(fontsize=7.5)
            ax.grid(alpha=0.25)
            show_figure(fig)

    st.subheader("Edge vs centralised processing (simulated model outputs)")
    rows = []
    for system in SYSTEMS:
        system_summary = summary.get("systems", {}).get(system, {})
        rows.append({
            "system": SYSTEM_LABELS.get(system, system),
            "simulated CPU utilisation (mean)": system_summary.get("simulated_cpu_utilisation_mean"),
            "simulated memory (MB, mean)": system_summary.get("simulated_memory_mean_mb"),
            "simulated edge uplink (kbps)": system_summary.get("simulated_edge_uplink_mean_kbps"),
            "simulated central uplink (kbps)": system_summary.get("simulated_cloud_uplink_mean_kbps"),
            "simulated bandwidth overhead ratio": system_summary.get("simulated_bandwidth_overhead_ratio"),
            "simulated edge latency (ms)": system_summary.get("centralization_edge_latency_mean_ms"),
            "simulated central latency (ms)": system_summary.get("centralization_central_latency_mean_ms"),
        })
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    st.subheader("Measured on this machine (actual host measurements)")
    measured = pd.DataFrame([
        {"metric": key,
         "value": _fmt_value(summary.get("systems", {}).get("S3_adaptive_closed_loop", {}).get(key))}
        for key in ("measured_wall_clock_seconds", "measured_per_step_ms", "measured_peak_alloc_mb")
    ])
    st.dataframe(measured, width="stretch", hide_index=True)
    st.caption("Wall-clock and peak-allocation figures were measured while the experiment ran "
               "(time.perf_counter / tracemalloc). They describe this prototype's Python runtime, "
               "not an embedded edge device.")


def page_devices(data: dict) -> None:
    profiles = data["tables"].get("device_profiles", pd.DataFrame())
    telemetry = data["tables"].get("telemetry", pd.DataFrame())
    learning = data["tables"].get("learning_log", pd.DataFrame())
    decisions = data["tables"].get("decision_log", pd.DataFrame())
    st.subheader("Device inventory and normal behaviour profiles")
    if not profiles.empty:
        st.dataframe(profiles, width="stretch", hide_index=True)

    if not decisions.empty:
        proposed = system_frame(decisions, "S3_adaptive_closed_loop")
        device = st.selectbox("Device behavioural history",
                              sorted(proposed["device_id"].unique()))
        fig = timeline_chart(proposed, device)
        if fig is not None:
            show_figure(fig)
        if not telemetry.empty:
            rows = telemetry[telemetry["device_id"] == device].sort_values("step")
            observed = proposed[proposed["device_id"] == device].sort_values("step")
            fig, axes = plt.subplots(2, 1, figsize=(10, 4.4), sharex=True)
            axes[0].plot(rows["step"], rows["packet_rate"], linewidth=1.0, color="#1f3b73")
            axes[0].set_ylabel("raw packets/s")
            axes[0].set_title("raw synthetic telemetry (ground truth stream)", fontsize=9)
            axes[1].plot(observed["step"], observed["allowance"], color="#8e44ad", linewidth=1.2)
            axes[1].set_ylabel("allowance")
            axes[1].set_xlabel("simulation step")
            axes[1].set_title("simulated containment allowance applied to this device", fontsize=9)
            for axis in axes:
                axis.grid(alpha=0.25)
            show_figure(fig)

    st.subheader("Baseline learning activity (anti-poisoning evidence)")
    if not learning.empty:
        subset = learning[learning["system"] == "S3_adaptive_closed_loop"] \
            if "system" in learning.columns else learning
        counts = subset["mode"].value_counts().reset_index()
        counts.columns = ["mode", "count"]
        st.dataframe(counts, width="stretch", hide_index=True)
        st.caption("`clean`/`drift` = accepted updates; `drift_release` = a freeze released by the "
                   "gate-checked drift pathway; `blocked_*` = refused updates (anomalous evidence, "
                   "frozen baseline, or telemetry distorted by heavy containment).")
        st.dataframe(subset.tail(300), width="stretch", hide_index=True, height=260)


def page_experiment(data: dict) -> None:
    config = data.get("config", {})
    summary = data.get("summary", {})
    st.subheader("Experiment configuration (reproducibility)")
    st.caption("Every number in this dashboard can be regenerated from the scenario, the seed and "
               "the configuration recorded here.")
    st.json(config or {"note": "no configuration file found"})

    st.subheader("Injected synthetic attacks")
    attacks = data["tables"].get("attacks", pd.DataFrame())
    if not attacks.empty:
        st.dataframe(attacks, width="stretch", hide_index=True)
        held_out = attacks[attacks["held_out"] == True]  # noqa: E712
        st.caption(f"{len(held_out)} of {len(attacks)} attacks use held-out (unseen) patterns. "
                   "They are reported separately from development patterns; this is a proxy for "
                   "unknown attacks, not proof of zero-day generalisation.")

    st.subheader("Early-warning/failure analysis")
    per_attack = data["tables"].get("per_attack_type", pd.DataFrame())
    episodes = data["tables"].get("episodes", pd.DataFrame())
    if not episodes.empty:
        missed = episodes[episodes["detected"] == 0]
        if missed.empty:
            st.success("No missed attacks in this run (all injected episodes were flagged).")
        else:
            st.warning(f"{len(missed)} injected episodes were not flagged.")
            st.dataframe(missed, width="stretch", hide_index=True)
    if not per_attack.empty:
        st.dataframe(per_attack, width="stretch", hide_index=True)

    st.subheader("Exports")
    tables = data["tables"]
    columns = st.columns(3)
    for index, name in enumerate(sorted(tables)):
        frame = tables[name]
        if frame.empty:
            continue
        with columns[index % 3]:
            st.download_button(f"download {name}.csv", data=frame.to_csv(index=False),
                               file_name=f"{name}.csv", mime="text/csv")


def page_method(data: dict) -> None:
    st.subheader("What is synthetic, what is predicted, what is simulated, what is measured")
    st.markdown(
        """
| Kind | Examples in this dashboard | Meaning |
| --- | --- | --- |
| **Synthetic ground truth** | `ground_truth`, `attack_type`, attack intervals, benign behaviour shifts | Generated by the traffic simulator. Used *only* for evaluation. |
| **Model prediction** | `anomaly_score`, `is_anomalous`, `stat_score`, `if_score`, `threshold`, `severity` | Produced online from features only: no labels, no criticality in the detector. |
| **Simulated response** | containment actions, traffic allowance, verification outcomes, recovery events | The gateway's own actions and their *modelled* effects inside the simulation. |
| **Measured** | wall-clock seconds, per-step milliseconds, peak Python allocation | Actual measurements on the machine that ran the experiment. |
| **Simulated resources** | CPU utilisation, memory footprint, uplink kbps, processing latency | Outputs of the edge resource model, not machine or network measurements. |
        """
    )
    st.subheader("Metric definitions")
    st.markdown(
        """
* **accuracy / precision / recall / F1 / false-positive rate** - point-wise over device-steps in the
  post-calibration window. Steps where a fully isolated device emits no telemetry are excluded and
  counted separately (`n_steps_not_scored`), so containment can never fabricate a false negative.
* **detection latency** - steps (and simulated seconds) from the attack start to the first flagged
  step of that episode; a detection within the episode plus a 2-step grace window counts.
* **response latency** - steps from first detection to the first mitigation action (level >= 1);
  OBSERVE-only decisions are not counted as a response.
* **containment success rate** - SUCCESS / (SUCCESS + FAILURE) over completed verification windows;
  INCONCLUSIVE windows are reported separately and excluded.
* **recovery time** - steps from the first containment action to the full restore of that device.
* **unnecessary quarantines** - QUARANTINE actions applied on device-steps whose synthetic label is
  benign (evaluation only; the policy never sees the label).
* **drift adaptation time** - steps until a legitimately drifted device stops being flagged for a
  run of five consecutive decisions.
* **bandwidth overhead** - ratio of the simulated centralised uplink (one record per device per step)
  to the simulated edge uplink (heartbeat plus one alert per action).
        """
    )
    st.subheader("Scope, and what this prototype does not claim")
    st.markdown(
        """
* The attacks are synthetic parameter combinations, so detection quality here is **not** evidence of
  real-world detection quality.
* Held-out patterns are a *proxy* for unknown attacks: the detector never sees labels, and thresholds
  are calibrated on benign data only, but the held-out patterns were still designed by the same
  author as the detector.
* Containment is simulated: no firewall rule, network scan, or real device is ever touched.
* Resource numbers labelled *simulated* come from the resource model, not from hardware measurements.
* The drift-evidence gates that allow a frozen baseline to be released are engineering heuristics; an
  adversary that keeps its traffic inside the plausible band while suppressing qualitative indicators
  could in principle be learned. This is discussed in the README limitations section.
        """
    )
    st.subheader("Documentation diagrams")
    for name in ("architecture.png", "closed_loop_flowchart.png"):
        path = FIGURES_DIR / name
        if path.exists():
            st.image(str(path), caption=f"figures/{name}", width="stretch")

    st.subheader("Generated figures for this scenario")
    scenario_dir = FIGURES_DIR / str(data.get("scenario", "main"))
    if scenario_dir.exists():
        figures = sorted(scenario_dir.glob("*.png"))
        if figures:
            for figure in figures:
                st.image(str(figure), caption=f"figures/{data.get('scenario')}/{figure.name}",
                         width="stretch")
        else:
            st.info("No figures yet - run the experiment with 'generate figures' enabled.")
    else:
        st.info(f"No figures directory for scenario '{data.get('scenario')}' yet.")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    st.set_page_config(page_title="Adaptive Edge Cyber-Resilience (IoT)", layout="wide")
    st.title("Adaptive Edge Cyber-Resilience for Resource-Constrained IoT")
    st.caption("Real-time detection and autonomous containment of unknown attacks - "
               "local, reproducible simulation")
    st.info(SCOPE_NOTE)

    scenarios = discover_scenarios()
    if not scenarios:
        st.warning("No results found in `results/`. Run the experiment from the sidebar "
                   "(or `python -m src.experiment_runner`) and reload this page.")
        with st.spinner("Running a quick experiment so the dashboard has data ..."):
            run_experiment_now("main", quick=True, with_figures=False)
        return

    data_by_scenario = {name: load_scenario(name, _signature(RESULTS_DIR / name))
                        for name in scenarios}
    scenario, data = render_sidebar(data_by_scenario)
    st.caption(f"Scenario **{scenario}** - artifacts in `{Path(data['dir']).as_posix()}`")

    tabs = st.tabs(["Overview", "Event stream", "Detection", "Threat & containment",
                    "Devices & learning", "Resources", "Experiment & export", "Method & scope"])
    with tabs[0]:
        page_overview(data)
    with tabs[1]:
        page_stream(data)
    with tabs[2]:
        page_detection(data)
    with tabs[3]:
        page_threat_and_containment(data)
    with tabs[4]:
        page_devices(data)
    with tabs[5]:
        page_resources(data)
    with tabs[6]:
        page_experiment(data)
    with tabs[7]:
        page_method(data)


def app(environ, start_response):
    """Minimal WSGI entrypoint for Vercel's Python runtime.

    The full interactive dashboard remains a Streamlit application and should be
    launched with ``streamlit run app.py``. Vercel cannot host Streamlit's
    long-lived websocket server, so the serverless deployment exposes a small
    health/status response instead of attempting to start Streamlit at import
    time.
    """
    body = (
        "Adaptive Edge Cyber-Resilience\n\n"
        "The research dashboard is a local Streamlit simulation. "
        "Run: python -m streamlit run app.py\n"
    ).encode("utf-8")
    start_response(
        "200 OK",
        [
            ("Content-Type", "text/plain; charset=utf-8"),
            ("Content-Length", str(len(body))),
        ],
    )
    return [body]


if __name__ == "__main__":
    main()
