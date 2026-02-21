"""
Test cwas.cli — AVAILABLE_STEPS and main() entry point.
"""
import sys

import pytest

import cwas.cli as cli


def test_available_steps_is_list():
    assert isinstance(cli.AVAILABLE_STEPS, list)


def test_available_steps_count():
    # start, configuration, preparation, annotation, categorization,
    # binomial_test, permutation_test, burden_shift, effective_num_test,
    # correlation, risk_score, dawn, extract_variant
    assert len(cli.AVAILABLE_STEPS) >= 13


def test_available_steps_known_entries():
    for step in [
        "start", "configuration", "preparation", "annotation",
        "categorization", "binomial_test", "permutation_test",
        "burden_shift", "effective_num_test", "correlation",
        "risk_score", "dawn", "extract_variant",
    ]:
        assert step in cli.AVAILABLE_STEPS


def test_main_no_args(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["cwas"])
    with pytest.raises(SystemExit) as exc_info:
        cli.main()
    assert exc_info.value.code == 1


def test_main_invalid_step(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["cwas", "nonexistent_step"])
    with pytest.raises(ValueError, match="does not support"):
        cli.main()
