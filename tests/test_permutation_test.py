"""
Test cwas.permutation_test
"""
import argparse

import numpy as np
import pytest
from pathlib import Path

from cwas.permutation_test import PermutationTest, _worker_data


class PermutationTestMock(PermutationTest):
    """Mock that bypasses validation and I/O."""

    @staticmethod
    def _print_args(args):
        pass

    @staticmethod
    def _check_args_validity(args):
        pass

    def save_result(self):
        pass

    def update_env(self):
        pass


@pytest.fixture
def perm_inst():
    """Create a PermutationTestMock with minimal args."""
    args = argparse.Namespace(
        num_proc=1,
        num_perm=100,
        cat_path=Path("test.categorization_result.zarr"),
        output_dir_path=Path("/tmp"),
        sample_info_path=Path("/tmp/sample.txt"),
        adj_factor_path=None,
        use_n_carrier=False,
        burden_shift=False,
        eff_test=False,
        tag=None,
        plot_size=7, plot_title="test", font_size=15, marker_size=15,
    )
    return PermutationTestMock(args)


@pytest.fixture(autouse=True)
def cleanup_worker_data():
    """Ensure _worker_data is cleared after each test."""
    yield
    _worker_data.clear()


# --- Path tests ---

def test_result_path(perm_inst):
    assert perm_inst.result_path.name == "test.permutation_test.txt.gz"


def test_perm_rrs_path(perm_inst):
    assert perm_inst.perm_rrs_path.name == "test.permutation_RRs.txt.gz"


def test_binom_pvals_path(perm_inst):
    assert perm_inst.binom_pvals_path.name == "test.binom_pvals.parquet"


# --- get_perm_pval tests ---

def test_get_perm_pval_enrichment(perm_inst):
    """RR > 1: count perm_rrs >= rr."""
    rr = np.array([2.0])
    perm_rrs = np.array([[1.0], [1.5], [2.0], [3.0], [0.5]])
    result = perm_inst.get_perm_pval(perm_rrs, rr)
    # 2 extreme values (2.0, 3.0) -> (2+1)/(5+1) = 0.5
    assert result == pytest.approx([0.5])


def test_get_perm_pval_depletion(perm_inst):
    """RR < 1: count perm_rrs <= rr."""
    rr = np.array([0.3])
    perm_rrs = np.array([[0.1], [0.2], [0.3], [0.5], [1.0]])
    result = perm_inst.get_perm_pval(perm_rrs, rr)
    # 3 extreme values (0.1, 0.2, 0.3) -> (3+1)/(5+1) = 2/3
    assert result == pytest.approx([4 / 6])


def test_get_perm_pval_no_extreme(perm_inst):
    """No permutation RR is as extreme as observed."""
    rr = np.array([5.0])
    perm_rrs = np.array([[1.0], [2.0], [3.0], [4.0]])
    result = perm_inst.get_perm_pval(perm_rrs, rr)
    # (0+1)/(4+1) = 0.2
    assert result == pytest.approx([0.2])


def test_get_perm_pval_all_extreme(perm_inst):
    """All permutation RRs are as extreme as observed."""
    rr = np.array([1.5])
    perm_rrs = np.array([[2.0], [3.0], [4.0]])
    result = perm_inst.get_perm_pval(perm_rrs, rr)
    # (3+1)/(3+1) = 1.0
    assert result == pytest.approx([1.0])


def test_get_perm_pval_multiple_categories(perm_inst):
    """Test with 2 categories simultaneously."""
    rr = np.array([2.0, 0.5])
    perm_rrs = np.array([
        [1.0, 0.3],
        [3.0, 0.8],
        [2.5, 0.4],
    ])
    result = perm_inst.get_perm_pval(perm_rrs, rr)
    # Cat 0 (RR=2.0 >= 1): extremes [3.0, 2.5] -> (2+1)/(3+1) = 0.75
    # Cat 1 (RR=0.5 < 1): extremes [0.3, 0.4] -> (2+1)/(3+1) = 0.75
    assert result == pytest.approx([0.75, 0.75])


# --- _burden_test static method tests ---

def test_burden_test_basic_shape():
    """Test output shape: num_perms x num_categories."""
    data = np.array([
        [1, 0],
        [0, 1],
        [2, 3],
        [4, 5],
    ], dtype=np.float64)
    _worker_data['data'] = data
    _worker_data['case_cnt'] = 2
    _worker_data['ctrl_cnt'] = 2
    _worker_data['use_n_carrier'] = False
    _worker_data['burden_shift'] = False

    result = PermutationTestMock._burden_test((0, 3))

    assert len(result) == 1
    assert result[0].shape == (3, 2)  # 3 permutations, 2 categories


def test_burden_test_deterministic():
    """Same seeds produce same results."""
    data = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.float64)
    _worker_data['data'] = data
    _worker_data['case_cnt'] = 2
    _worker_data['ctrl_cnt'] = 2
    _worker_data['use_n_carrier'] = False
    _worker_data['burden_shift'] = False

    result1 = PermutationTestMock._burden_test((0, 5))
    result2 = PermutationTestMock._burden_test((0, 5))

    np.testing.assert_array_equal(result1[0], result2[0])


def test_burden_test_with_burden_shift():
    """burden_shift=True returns interleaved RR + pval rows."""
    data = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.float64)
    _worker_data['data'] = data
    _worker_data['case_cnt'] = 2
    _worker_data['ctrl_cnt'] = 2
    _worker_data['use_n_carrier'] = False
    _worker_data['burden_shift'] = True

    result = PermutationTestMock._burden_test((0, 3))

    assert len(result) == 1
    # With burden_shift: 2*num_perms rows (interleaved rr, pval)
    assert result[0].shape == (6, 2)


def test_burden_test_carrier_mode():
    """use_n_carrier=True binarizes the data internally."""
    data = np.array([[5, 0], [0, 3], [2, 1], [0, 0]], dtype=np.float64)
    _worker_data['data'] = data
    _worker_data['case_cnt'] = 2
    _worker_data['ctrl_cnt'] = 2
    _worker_data['use_n_carrier'] = True
    _worker_data['burden_shift'] = False

    result = PermutationTestMock._burden_test((0, 3))

    assert result[0].shape == (3, 2)


def test_burden_test_different_seed_ranges():
    """Different seed ranges produce different results."""
    data = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.float64)
    _worker_data['data'] = data
    _worker_data['case_cnt'] = 2
    _worker_data['ctrl_cnt'] = 2
    _worker_data['use_n_carrier'] = False
    _worker_data['burden_shift'] = False

    result1 = PermutationTestMock._burden_test((0, 5))
    result2 = PermutationTestMock._burden_test((5, 10))

    assert not np.array_equal(result1[0], result2[0])
