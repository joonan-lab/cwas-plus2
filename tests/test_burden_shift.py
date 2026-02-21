"""
Test cwas.burden_shift
"""
import argparse

import numpy as np
import pytest
from pathlib import Path

from cwas.burden_shift import BurdenShift


class BurdenShiftMock(BurdenShift):
    """Mock that bypasses validation and I/O."""

    @staticmethod
    def _print_args(args):
        pass

    @staticmethod
    def _check_args_validity(args):
        pass


def _make_args(**overrides):
    defaults = dict(
        input_path=Path("/tmp/test.burden_test.txt"),
        burden_res=Path("/tmp/test.burden_shift.txt"),
        output_dir_path=Path("/tmp/output"),
        cat_set_file=Path("/tmp/catset.txt"),
        cat_count_file=Path("/tmp/catcount.txt"),
        count_cutoff=5,
        pval=0.05,
        tag=None,
        plot_title="Test",
        cat_set_list=None,
        n_cat_sets=10,
        fontsize=12,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


@pytest.fixture
def bs_inst():
    """Create a BurdenShiftMock with minimal args."""
    return BurdenShiftMock(_make_args())


# --- Property tests ---

def test_c_cutoff(bs_inst):
    assert bs_inst.c_cutoff == 5


def test_c_cutoff_negative():
    inst = BurdenShiftMock(_make_args(count_cutoff=-1))
    with pytest.raises(ValueError, match="positive"):
        _ = inst.c_cutoff


def test_c_cutoff_zero():
    inst = BurdenShiftMock(_make_args(count_cutoff=0))
    assert inst.c_cutoff == 0


def test_pval(bs_inst):
    assert bs_inst.pval == 0.05


def test_tag_none(bs_inst):
    assert bs_inst.tag is None


def test_tag_set():
    inst = BurdenShiftMock(_make_args(tag="v1"))
    assert inst.tag == "v1"


def test_plot_title(bs_inst):
    assert bs_inst.plot_title == "Test"


def test_n_cat_sets(bs_inst):
    assert bs_inst.n_cat_sets == 10


def test_fontsize(bs_inst):
    assert bs_inst.fontsize == 12


def test_cat_set_list_none(bs_inst):
    assert bs_inst.cat_set_list is None


# --- _count_cats tests ---

def test_count_cats_basic(bs_inst):
    """Positive pvals <= threshold are case; negative pvals with |pval| <= threshold are ctrl."""
    pvals = [0.01, -0.03, 0.05, -0.06, 0.10, -0.01]
    n_case, n_ctrl = bs_inst._count_cats(pvals, 0.05)
    # Case: 0.01, 0.05 (positive and <= 0.05) -> 2
    # Ctrl: -0.03, -0.01 (negative and abs <= 0.05) -> 2
    assert n_case == 2
    assert n_ctrl == 2


def test_count_cats_no_significant(bs_inst):
    pvals = [0.5, -0.6, 0.3, -0.4]
    n_case, n_ctrl = bs_inst._count_cats(pvals, 0.05)
    assert n_case == 0
    assert n_ctrl == 0


def test_count_cats_all_case(bs_inst):
    pvals = [0.01, 0.02, 0.03]
    n_case, n_ctrl = bs_inst._count_cats(pvals, 0.05)
    assert n_case == 3
    assert n_ctrl == 0


def test_count_cats_all_ctrl(bs_inst):
    pvals = [-0.01, -0.02, -0.03]
    n_case, n_ctrl = bs_inst._count_cats(pvals, 0.05)
    assert n_case == 0
    assert n_ctrl == 3


def test_count_cats_threshold_boundary(bs_inst):
    """Values exactly at threshold should be included."""
    pvals = [0.05, -0.05]
    n_case, n_ctrl = bs_inst._count_cats(pvals, 0.05)
    assert n_case == 1
    assert n_ctrl == 1


# --- _burden_shift_size tests ---

def test_burden_shift_size_below_min(bs_inst):
    bins = [10, 50, 100, 150, 200, 250]
    assert bs_inst._burden_shift_size(5, bins) == 1


def test_burden_shift_size_at_min(bs_inst):
    bins = [10, 50, 100, 150, 200, 250]
    assert bs_inst._burden_shift_size(10, bins) == 1


def test_burden_shift_size_bin1(bs_inst):
    bins = [10, 50, 100, 150, 200, 250]
    assert bs_inst._burden_shift_size(30, bins) == 3


def test_burden_shift_size_bin2(bs_inst):
    bins = [10, 50, 100, 150, 200, 250]
    assert bs_inst._burden_shift_size(75, bins) == 5


def test_burden_shift_size_bin3(bs_inst):
    bins = [10, 50, 100, 150, 200, 250]
    assert bs_inst._burden_shift_size(120, bins) == 7


def test_burden_shift_size_bin4(bs_inst):
    bins = [10, 50, 100, 150, 200, 250]
    assert bs_inst._burden_shift_size(180, bins) == 9


def test_burden_shift_size_bin5(bs_inst):
    bins = [10, 50, 100, 150, 200, 250]
    assert bs_inst._burden_shift_size(220, bins) == 11


def test_burden_shift_size_above_max(bs_inst):
    bins = [10, 50, 100, 150, 200, 250]
    assert bs_inst._burden_shift_size(300, bins) == 13


# --- _change_cre_name tests ---

def test_change_cre_name_basic(bs_inst):
    result = bs_inst._change_cre_name("is_coding_CRE1")
    assert "CRE" not in result.replace("-CRE", "").replace(" ", "")
    assert "-CRE" in result


def test_change_cre_name_with_prefix(bs_inst):
    result = bs_inst._change_cre_name("is_noncoding_ActiveCRE2")
    assert "-CRE" in result


# --- _match_cat_sets tests ---

def test_match_cat_sets_single_match(bs_inst):
    result = bs_inst._match_cat_sets("coding", "is_coding")
    assert result == "is_coding"


def test_match_cat_sets_multi_match(bs_inst):
    result = bs_inst._match_cat_sets("coding&ASD", "is_coding_ASD")
    assert result == "is_coding_ASD"


def test_match_cat_sets_no_match(bs_inst):
    result = bs_inst._match_cat_sets("coding&ASD", "is_noncoding_ASD")
    assert result is None


def test_match_cat_sets_length_mismatch(bs_inst):
    """If pre has different number of segments than x, no match."""
    result = bs_inst._match_cat_sets("coding", "is_coding_ASD")
    assert result is None


def test_match_cat_sets_partial_overlap(bs_inst):
    """Only partial overlap should not match."""
    result = bs_inst._match_cat_sets("coding&ASD", "is_coding_SCZ")
    assert result is None
