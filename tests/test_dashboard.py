"""
Dashboard smoke test.

Runs ``app.py`` headlessly with Streamlit's own test harness (``AppTest``) so the
data loading and every dashboard page are actually executed, not just imported.
The test is skipped when no experiment artifacts exist yet -- run
``python -m src.experiment_runner`` (or the quick variant) first.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from config import RESULTS_DIR

pytest.importorskip("streamlit.testing.v1", reason="Streamlit test harness unavailable")


def _has_results() -> bool:
    return (RESULTS_DIR / "main" / "experiment_summary.json").exists()


@pytest.mark.skipif(not _has_results(), reason="no results/ artifacts yet - run the experiment first")
def test_dashboard_renders_without_exceptions():
    from streamlit.testing.v1 import AppTest

    app_path = Path(__file__).resolve().parents[1] / "app.py"
    app_test = AppTest.from_file(str(app_path), default_timeout=240)
    app_test.run()

    assert not app_test.exception, [str(e) for e in app_test.exception]
    assert app_test.title, "the dashboard should render a title"
    assert "Adaptive Edge Cyber-Resilience" in app_test.title[0].value
    # Core widgets the pages rely on must exist.
    assert len(app_test.tabs) >= 6, f"expected the dashboard tabs, got {len(app_test.tabs)}"
    assert len(app_test.metric) >= 4, "headline metrics are missing"
    assert len(app_test.dataframe) >= 3, "result tables are missing"


@pytest.mark.skipif(not _has_results(), reason="no results/ artifacts yet - run the experiment first")
def test_dashboard_scenario_selector_covers_saved_scenarios():
    from streamlit.testing.v1 import AppTest

    app_path = Path(__file__).resolve().parents[1] / "app.py"
    app_test = AppTest.from_file(str(app_path), default_timeout=240)
    app_test.run()
    assert not app_test.exception, [str(e) for e in app_test.exception]

    saved = sorted(p.name for p in RESULTS_DIR.iterdir()
                   if p.is_dir() and (p / "experiment_summary.json").exists())
    selectors = [box for box in app_test.selectbox if box.label == "Scenario"]
    assert selectors, "no scenario selector found"
    selector = selectors[0]
    options = [option for option in selector.options if option in saved]
    assert set(options) == set(saved), (options, saved)

    # Selecting another scenario must re-render without errors.
    if len(saved) > 1:
        selector.select(saved[1]).run()
        assert not app_test.exception, [str(e) for e in app_test.exception]
        summary = json.loads((RESULTS_DIR / saved[1] / "experiment_summary.json").read_text())
        assert summary["scenario"] == saved[1]
