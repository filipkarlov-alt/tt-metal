# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Runs the whole session flow against fake_benchmark.py instead of a device binary."""

import json
import sys
from pathlib import Path

import pytest

from tests.perf import compare as cmp
from tests.perf import golden as golden_io
from tests.perf import session, update
from tests.perf.registry import Suite
from tests.perf.unit.expect import expect_error  # noqa: F401

FAKE = Path(__file__).with_name("fake_benchmark.py")
METRIC = {"name": "IterationTime", "decl": "unit=s;better=lower;aggregate=min"}
ENV = "wh_n300_civ2"


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    monkeypatch.setattr(session, "OUTPUT_ROOT", tmp_path / "generated")
    wrapper = tmp_path / "bench"
    wrapper.write_text(f'#!/bin/sh\nexec {sys.executable} {FAKE} "$@"\n')
    wrapper.chmod(0o755)
    plan = tmp_path / "plan.json"
    monkeypatch.setenv("FAKE_BENCH_PLAN", str(plan))

    def make(runs, golden_cases=None, policy=cmp.Policy(5, 5)):
        plan.write_text(json.dumps({"metric": METRIC, "runs": runs}))
        suite = Suite(name="fake", binary=wrapper, golden=tmp_path / "golden.json", policy=policy, repetitions=2)
        if golden_cases is not None:
            golden = golden_io.Golden("fake", {}, {})
            golden.metrics = {"IterationTime": session.MetricSpec("s", "lower", "min")}
            golden.environments[ENV] = golden_io.Environment(repetitions=2, cases=golden_cases)
            golden_io.save(golden, suite.golden)
        return suite

    return make, plan


def test_regression_is_retried_with_an_exact_filter_and_fails_when_it_repeats(setup):
    make, plan = setup
    golden = {"BM/a/k:1/manual_time": {"IterationTime": 1e-6}, "BM/a/k:2/manual_time": {"IterationTime": 1e-6}}
    suite = make([{"BM/a/k:1/manual_time": 1.2e-6, "BM/a/k:2/manual_time": 1e-6}], golden)
    record = session.execute(suite, ENV)
    calls = json.loads(plan.with_suffix(".calls").read_text())
    assert len(calls) == 2 and "--benchmark_filter=^(BM/a/k:1/manual_time)$" in calls[1]
    assert "--benchmark_repetitions=2" in calls[0]
    _, comparison = session.evaluate(record, suite)
    assert {r.case: r.status for r in comparison.results}["BM/a/k:1/manual_time"] == "REGRESSION"
    assert record.context == {"iommu": "on", "aiclk_mhz": "1000"}
    saved = session.Record.load(session.output_dir("fake", ENV) / "measurements.json")
    assert saved.retry == record.retry and saved.cases == record.cases


def test_noise_spike_passes_after_retry(setup):
    make, _ = setup
    golden = {"BM/a/k:1/manual_time": {"IterationTime": 1e-6}}
    suite = make([{"BM/a/k:1/manual_time": 1.2e-6}, {"BM/a/k:1/manual_time": 1.01e-6}], golden)
    record = session.execute(suite, ENV)
    golden_doc, comparison = session.evaluate(record, suite)
    assert session.failures(record, suite, golden_doc, comparison) == []


def test_update_from_record_writes_confirmed_improvement(setup):
    make, _ = setup
    golden = {"BM/a/k:1/manual_time": {"IterationTime": 1e-6}, "BM/a/k:2/manual_time": {"IterationTime": 1e-6}}
    suite = make(
        [{"BM/a/k:1/manual_time": 0.8e-6, "BM/a/k:2/manual_time": 1e-6}, {"BM/a/k:1/manual_time": 0.82e-6}], golden
    )
    record = session.execute(suite, ENV)
    golden_doc, comparison = session.evaluate(record, suite)
    session.update_golden(record, suite, golden_doc, comparison, force=False, source="test")
    reloaded = golden_io.load(suite.golden, "fake").environments[ENV]
    assert reloaded.cases["BM/a/k:1/manual_time"] == {"IterationTime": 8.2e-07}
    assert reloaded.cases["BM/a/k:2/manual_time"] == {"IterationTime": 1e-06}
    assert reloaded.provenance["source"] == "test"


def test_first_recording_of_a_new_suite_needs_force_and_report_only_suites_pass_meanwhile(setup, expect_error):
    make, _ = setup
    suite = make([{"BM/a/k:1/manual_time": 1e-6}], policy=cmp.Policy(5, 5, enforce=False))
    record = session.execute(suite, ENV)
    golden_doc, comparison = session.evaluate(record, suite)
    assert session.failures(record, suite, golden_doc, comparison) == []
    with expect_error(update.UpdateRefused):
        session.update_golden(record, suite, golden_doc, comparison, force=False, source="test")
    session.update_golden(record, suite, golden_doc, comparison, force=True, source="test")
    assert golden_io.load(suite.golden, "fake").environments[ENV].cases == {
        "BM/a/k:1/manual_time": {"IterationTime": 1e-06}
    }


def test_report_only_suite_still_fails_on_benchmark_errors(setup):
    make, _ = setup
    golden = {"BM/a/k:1/manual_time": {"IterationTime": 1e-6}}
    suite = make([{"BM/a/k:1/manual_time": None}], golden, policy=cmp.Policy(5, 5, enforce=False))
    record = session.execute(suite, ENV)
    golden_doc, comparison = session.evaluate(record, suite)
    assert session.failures(record, suite, golden_doc, comparison) == ["1 ERROR"]
