"""
Test cwas.risk_score
"""
import argparse

import numpy as np
import pytest
from pathlib import Path


# --- Tests that don't need an instance ---

def test_import():
    """Module can be imported."""
    import cwas.risk_score  # noqa: F401


# --- Tests that need a RiskScore instance ---

class TestRiskScoreInstance:
    """Tests that instantiate RiskScore."""

    @staticmethod
    def _make_args(**overrides):
        from cwas.risk_score import RiskScore

        defaults = dict(
            num_proc=1,
            categorization_result_path=Path("/tmp/test.categorization_result.zarr"),
            output_dir_path=Path("/tmp/output"),
            sample_info_path=Path("/tmp/sample.txt"),
            adj_factor_path=None,
            category_set_path=Path("/tmp/catset.txt"),
            domain_list="all",
            use_n_carrier=False,
            ctrl_thres=10,
            train_set_f=0.8,
            num_reg=5,
            fold=5,
            n_permute=100,
            predict_only=False,
            tag=None,
            seed=42,
            plotsize="6,4",
            fontsize=12.0,
            do_each_one=False,
            leave_one_out=False,
            feature_selection_group="gene_set",
        )
        defaults.update(overrides)
        return argparse.Namespace(**defaults)

    @staticmethod
    def _make_inst(**overrides):
        from cwas.risk_score import RiskScore

        class RiskScoreMock(RiskScore):
            @staticmethod
            def _print_args(args):
                pass

            @staticmethod
            def _check_args_validity(args):
                pass

        return RiskScoreMock(TestRiskScoreInstance._make_args(**overrides))

    # --- leave_one_out logic ---

    def test_leave_one_out_false_when_do_each_one(self):
        inst = self._make_inst(do_each_one=True, leave_one_out=True)
        assert inst.leave_one_out is False

    def test_leave_one_out_true(self):
        inst = self._make_inst(do_each_one=False, leave_one_out=True)
        assert inst.leave_one_out is True

    def test_leave_one_out_false(self):
        inst = self._make_inst(do_each_one=False, leave_one_out=False)
        assert inst.leave_one_out is False

    # --- feature_selection_group validation ---

    def test_feature_selection_group_valid(self):
        inst = self._make_inst(feature_selection_group="gene_set")
        assert inst.feature_selection_group == ["gene_set"]

    def test_feature_selection_group_multiple(self):
        inst = self._make_inst(
            feature_selection_group="gene_set, functional_score"
        )
        assert inst.feature_selection_group == ["gene_set", "functional_score"]

    def test_feature_selection_group_invalid(self):
        inst = self._make_inst(feature_selection_group="invalid_group")
        with pytest.raises(ValueError, match="Invalid feature selection group"):
            _ = inst.feature_selection_group

    # --- Path property tests ---

    def test_coef_path(self):
        inst = self._make_inst(tag=None)
        assert "lasso_coef" in inst.coef_path.name
        assert inst.coef_path.name.endswith(".txt")

    def test_result_path(self):
        inst = self._make_inst(tag=None)
        assert "lasso_results" in inst.result_path.name

    def test_null_model_path(self):
        inst = self._make_inst(tag=None)
        assert "lasso_null_models" in inst.null_model_path.name

    def test_plot_path(self):
        inst = self._make_inst(tag=None)
        assert "lasso_histogram" in inst.plot_path.name
        assert inst.plot_path.name.endswith(".pdf")

    def test_coef_path_with_tag(self):
        inst = self._make_inst(tag="v2")
        assert "v2" in inst.coef_path.name

    def test_result_path_with_tag(self):
        inst = self._make_inst(tag="v2")
        assert "v2" in inst.result_path.name

    # --- Simple property tests ---

    def test_ctrl_thres(self):
        inst = self._make_inst(ctrl_thres=20)
        assert inst.ctrl_thres == 20

    def test_train_set_f(self):
        inst = self._make_inst(train_set_f=0.7)
        assert inst.train_set_f == 0.7

    def test_num_reg(self):
        inst = self._make_inst(num_reg=10)
        assert inst.num_reg == 10

    def test_fold(self):
        inst = self._make_inst(fold=3)
        assert inst.fold == 3

    def test_n_permute(self):
        inst = self._make_inst(n_permute=50)
        assert inst.n_permute == 50

    def test_predict_only(self):
        inst = self._make_inst(predict_only=True)
        assert inst.predict_only is True

    def test_seed(self):
        inst = self._make_inst(seed=123)
        assert inst.seed == 123
