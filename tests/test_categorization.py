"""
Test cwas.categorization
"""
import argparse

import numpy as np
import pandas as pd
import pytest
import zarr
from pathlib import Path

from cwas.categorization import Categorization


class CategorizationMock(Categorization):
    """Mock that bypasses argument validation and printing."""

    @staticmethod
    def _print_args(args):
        pass

    @staticmethod
    def _check_args_validity(args):
        pass


@pytest.fixture
def cat_inst(tmp_path):
    """Create a CategorizationMock with minimal args."""
    input_path = tmp_path / "test.annotated.vcf.gz"
    input_path.touch()
    args = argparse.Namespace(
        num_proc=1,
        input_path=input_path,
        output_dir_path=tmp_path,
    )
    return CategorizationMock(args)


# --- result_path property ---

def test_result_path_vcf_gz(cat_inst):
    assert cat_inst.result_path.name == "test.categorization_result.zarr"


def test_result_path_vcf(tmp_path):
    input_path = tmp_path / "sample.annotated.vcf"
    input_path.touch()
    args = argparse.Namespace(
        num_proc=1,
        input_path=input_path,
        output_dir_path=tmp_path,
    )
    inst = CategorizationMock(args)
    assert inst.result_path.name == "sample.categorization_result.zarr"


def test_result_path_in_output_dir(cat_inst):
    assert cat_inst.result_path.parent == cat_inst.output_dir_path


# --- num_proc property ---

def test_num_proc(cat_inst):
    assert cat_inst.num_proc == 1


# --- _get_redundant_categories_from_row ---

def test_get_redundant_categories_single_wildcard(cat_inst):
    cat_inst._category_domain = {
        "variant_type": ["SNV", "Indel"],
        "gene_set": ["All"],
        "functional_score": ["All"],
        "gencode": ["CodingRegion"],
        "functional_annotation": ["All"],
    }
    row = pd.Series({
        "variant_type": "*",
        "gene_set": "All",
        "functional_score": "All",
        "gencode": "CodingRegion",
        "functional_annotation": "All",
    })
    result = cat_inst._get_redundant_categories_from_row(row)
    assert result == {
        "SNV_All_All_CodingRegion_All",
        "Indel_All_All_CodingRegion_All",
    }


def test_get_redundant_categories_no_wildcard(cat_inst):
    cat_inst._category_domain = {
        "variant_type": ["SNV", "Indel"],
        "gene_set": ["All", "ASD"],
        "functional_score": ["All"],
        "gencode": ["CodingRegion"],
        "functional_annotation": ["All"],
    }
    row = pd.Series({
        "variant_type": "SNV",
        "gene_set": "All",
        "functional_score": "All",
        "gencode": "CodingRegion",
        "functional_annotation": "All",
    })
    result = cat_inst._get_redundant_categories_from_row(row)
    assert result == {"SNV_All_All_CodingRegion_All"}


def test_get_redundant_categories_multiple_wildcards(cat_inst):
    cat_inst._category_domain = {
        "variant_type": ["SNV", "Indel"],
        "gene_set": ["A", "B"],
        "functional_score": ["All"],
        "gencode": ["CodingRegion"],
        "functional_annotation": ["All"],
    }
    row = pd.Series({
        "variant_type": "*",
        "gene_set": "*",
        "functional_score": "All",
        "gencode": "CodingRegion",
        "functional_annotation": "All",
    })
    result = cat_inst._get_redundant_categories_from_row(row)
    assert len(result) == 4  # 2 variant_types * 2 gene_sets
    expected = {
        "SNV_A_All_CodingRegion_All",
        "SNV_B_All_CodingRegion_All",
        "Indel_A_All_CodingRegion_All",
        "Indel_B_All_CodingRegion_All",
    }
    assert result == expected


# --- save_result + zarr roundtrip ---

def test_save_result_zarr_roundtrip(cat_inst):
    cat_inst._sample_ids = ["S1", "S2", "S3"]
    cat_inst._categories = ["a_b_c_d_e", "f_g_h_i_j"]
    cat_inst._result = np.array([
        [5, 3],
        [0, 7],
        [2, 1],
    ], dtype=np.int32)

    cat_inst.save_result()

    root = zarr.open(str(cat_inst.result_path), mode='r')
    loaded_data = root['data'][:]
    np.testing.assert_array_equal(loaded_data, cat_inst._result)
    assert list(root['metadata'].attrs['sample_id']) == ["S1", "S2", "S3"]
    assert list(root['metadata'].attrs['category']) == ["a_b_c_d_e", "f_g_h_i_j"]


def test_save_result_preserves_shape(cat_inst):
    n_samples, n_cats = 5, 3
    cat_inst._sample_ids = [f"S{i}" for i in range(n_samples)]
    cat_inst._categories = [f"c{i}_d_e_f_g" for i in range(n_cats)]
    cat_inst._result = np.zeros((n_samples, n_cats), dtype=np.int32)

    cat_inst.save_result()

    root = zarr.open(str(cat_inst.result_path), mode='r')
    assert root['data'].shape == (n_samples, n_cats)
