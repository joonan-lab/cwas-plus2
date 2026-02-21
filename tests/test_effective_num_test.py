"""
Test cwas.effective_num_test
"""
import argparse

import numpy as np
import pytest
from pathlib import Path

from cwas.effective_num_test import EffectiveNumTest


class EffectiveNumTestMock(EffectiveNumTest):
    """Mock that bypasses validation and I/O."""

    @staticmethod
    def _print_args(args):
        pass

    @staticmethod
    def _check_args_validity(args):
        pass


def _make_args(**overrides):
    defaults = dict(
        input_path=Path("/tmp/test.intersection_matrix.zarr"),
        input_format="inter",
        output_dir_path=Path("/tmp/output"),
        sample_info_path=Path("/tmp/sample.txt"),
        category_count_file=None,
        domain_list="all",
        tag=None,
        category_set_path=None,
        count_thres=None,
        num_eig=100,
        eff_num_test=True,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


@pytest.fixture
def ent_inst():
    """Create an EffectiveNumTestMock with minimal args."""
    return EffectiveNumTestMock(_make_args())


# --- Path property tests ---

def test_neg_lap_path_intersection(ent_inst):
    assert ent_inst.neg_lap_path.name == "test.neg_lap.zarr"


def test_neg_lap_path_correlation():
    inst = EffectiveNumTestMock(_make_args(
        input_path=Path("/tmp/test.correlation_matrix.zarr"),
    ))
    assert inst.neg_lap_path.name == "test.neg_lap.zarr"


def test_eig_val_path(ent_inst):
    assert ent_inst.eig_val_path.name == "test.eig_vals.zarr"


def test_eig_vec_path(ent_inst):
    assert ent_inst.eig_vec_path.name == "test.eig_vecs.zarr"


def test_neg_lap_path_with_tag():
    inst = EffectiveNumTestMock(_make_args(tag="coding"))
    assert inst.neg_lap_path.name == "test.neg_lap.coding.zarr"


def test_eig_val_path_with_tag():
    inst = EffectiveNumTestMock(_make_args(tag="coding"))
    assert inst.eig_val_path.name == "test.eig_vals.coding.zarr"


def test_eig_vec_path_with_tag():
    inst = EffectiveNumTestMock(_make_args(tag="coding"))
    assert inst.eig_vec_path.name == "test.eig_vecs.coding.zarr"


def test_paths_in_output_dir(ent_inst):
    assert ent_inst.neg_lap_path.parent == ent_inst.output_dir_path
    assert ent_inst.eig_val_path.parent == ent_inst.output_dir_path
    assert ent_inst.eig_vec_path.parent == ent_inst.output_dir_path


# --- Simple property tests ---

def test_input_format(ent_inst):
    assert ent_inst.input_format == "inter"


def test_tag_none(ent_inst):
    assert ent_inst.tag is None


def test_tag_set():
    inst = EffectiveNumTestMock(_make_args(tag="noncoding"))
    assert inst.tag == "noncoding"


def test_num_eig(ent_inst):
    assert ent_inst.num_eig == 100


def test_eff_num_test_flag(ent_inst):
    assert ent_inst.eff_num_test is True


def test_count_thres_explicit():
    """When count_thres is explicitly provided, use that value."""
    inst = EffectiveNumTestMock(_make_args(count_thres=10))
    assert inst.count_thres == 10


def test_category_set_path_none(ent_inst):
    assert ent_inst.category_set_path is None


def test_category_set_path_set():
    inst = EffectiveNumTestMock(_make_args(
        category_set_path=Path("/tmp/catset.txt"),
    ))
    assert inst.category_set_path is not None


# --- binom_p tests ---

def test_binom_p(ent_inst):
    """binom_p = fraction of cases among case+ctrl."""
    import pandas as pd
    ent_inst._sample_info = pd.DataFrame(
        {"PHENOTYPE": ["case", "ctrl", "case", "ctrl", "case", "ctrl"]},
        index=pd.Index(["S1", "S2", "S3", "S4", "S5", "S6"], name="SAMPLE"),
    )
    assert ent_inst.binom_p == pytest.approx(0.5)


def test_binom_p_unequal(ent_inst):
    import pandas as pd
    ent_inst._sample_info = pd.DataFrame(
        {"PHENOTYPE": ["case", "case", "case", "ctrl"]},
        index=pd.Index(["S1", "S2", "S3", "S4"], name="SAMPLE"),
    )
    assert ent_inst.binom_p == pytest.approx(0.75)


# --- count_thres computed ---

def test_count_thres_computed():
    """When count_thres is None, it is computed from binom_p via binomtest."""
    import pandas as pd
    inst = EffectiveNumTestMock(_make_args(count_thres=None))
    # Set binom_p = 0.5 (equal case/ctrl)
    inst._binom_p = 0.5
    thres = inst.count_thres
    assert isinstance(thres, int)
    assert thres > 0


# --- get_n_etests logic (tested via eigenvalue analysis) ---

def test_get_n_etests_with_zarr(tmp_path):
    """Test get_n_etests reads eigenvalues and computes 99% variance threshold."""
    import zarr

    # Create synthetic eigenvalues (descending magnitude)
    eig_vals = np.array([100.0, 50.0, 25.0, 10.0, 5.0, 1.0, 0.5, 0.1])

    eig_val_dir = tmp_path / "test.eig_vals.zarr"
    root = zarr.open(str(eig_val_dir), mode='w')
    root.create_dataset('data', data=eig_vals, dtype='float64')

    inst = EffectiveNumTestMock(_make_args(
        input_path=tmp_path / "test.intersection_matrix.zarr",
        output_dir_path=tmp_path,
        num_eig=100,
    ))
    inst._domain = 'all'
    inst.get_n_etests()

    assert inst.eff_num_test_value is not None
    assert inst.eff_num_test_value > 0
    # 99% of total sum: cumsum must reach 0.99 * total
    total = np.sum(eig_vals)
    cumsum = np.cumsum(np.sort(np.abs(eig_vals))[::-1])
    expected = np.searchsorted(cumsum, 0.99 * total) + 1
    assert inst.eff_num_test_value == expected


def test_get_n_etests_single_dominant(tmp_path):
    """One eigenvalue dominates; effective test count should be 1."""
    import zarr

    eig_vals = np.array([1000.0, 0.01, 0.001, 0.0001])
    eig_val_dir = tmp_path / "test.eig_vals.zarr"
    root = zarr.open(str(eig_val_dir), mode='w')
    root.create_dataset('data', data=eig_vals, dtype='float64')

    inst = EffectiveNumTestMock(_make_args(
        input_path=tmp_path / "test.intersection_matrix.zarr",
        output_dir_path=tmp_path,
        num_eig=100,
    ))
    inst._domain = 'all'
    inst.get_n_etests()

    assert inst.eff_num_test_value == 1


def test_get_n_etests_equal_eigenvalues(tmp_path):
    """All equal eigenvalues: need all of them for 99% variance."""
    import zarr

    eig_vals = np.ones(100)
    eig_val_dir = tmp_path / "test.eig_vals.zarr"
    root = zarr.open(str(eig_val_dir), mode='w')
    root.create_dataset('data', data=eig_vals, dtype='float64')

    inst = EffectiveNumTestMock(_make_args(
        input_path=tmp_path / "test.intersection_matrix.zarr",
        output_dir_path=tmp_path,
        num_eig=100,
    ))
    inst._domain = 'all'
    inst.get_n_etests()

    assert inst.eff_num_test_value == 99
