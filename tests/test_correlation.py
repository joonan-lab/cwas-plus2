"""
Test cwas.correlation
"""
import argparse

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from cwas.correlation import Correlation


class CorrelationMock(Correlation):
    """Mock that bypasses validation and I/O."""

    @staticmethod
    def _print_args(args):
        pass

    @staticmethod
    def _check_args_validity(args):
        pass


@pytest.fixture
def corr_inst():
    """Create a CorrelationMock with minimal args."""
    args = argparse.Namespace(
        num_proc=1,
        cat_path=Path("test.categorization_result.zarr"),
        output_dir_path=Path("/tmp"),
        annot_path=Path("/tmp/annotated.vcf"),
        generate_corr_matrix="sample",
        generate_inter_matrix=None,
        domain_list="all",
        category_info_path=Path("/tmp/category_info.txt"),
    )
    return CorrelationMock(args)


# --- Path tests ---

def test_matrix_path(corr_inst):
    assert corr_inst.matrix_path.name == "test.correlation_matrix.zarr"


def test_intersection_matrix_path(corr_inst):
    assert corr_inst.intersection_matrix_path.name == "test.intersection_matrix.zarr"


# --- process_columns tests ---

def test_process_columns_self_intersection():
    """A column's intersection with itself equals the count of positive entries."""
    matrix = pd.DataFrame({
        "A": [1, 0, 3, 0],
        "B": [0, 1, 2, 0],
    })
    result = Correlation.process_columns([0], matrix)
    assert len(result) == 1
    # A*A = [1,0,9,0] -> 2 positive; B*A = [0,0,6,0] -> 1 positive
    assert result[0]["A"] == 2
    assert result[0]["B"] == 1
    assert result[0].name == "A"


def test_process_columns_all_columns():
    """Test full intersection matrix via all columns."""
    matrix = pd.DataFrame({
        "X": [1, 1, 0],
        "Y": [1, 0, 1],
        "Z": [0, 1, 1],
    })
    result = Correlation.process_columns(range(3), matrix)
    assert len(result) == 3

    # Column X: X*X=[1,1,0]->2; Y*X=[1,0,0]->1; Z*X=[0,1,0]->1
    assert result[0].tolist() == [2, 1, 1]
    # Column Y: X*Y=[1,0,0]->1; Y*Y=[1,0,1]->2; Z*Y=[0,0,1]->1
    assert result[1].tolist() == [1, 2, 1]
    # Column Z: X*Z=[0,1,0]->1; Y*Z=[0,0,1]->1; Z*Z=[0,1,1]->2
    assert result[2].tolist() == [1, 1, 2]


def test_process_columns_no_overlap():
    """No co-occurrence between mutually exclusive columns."""
    matrix = pd.DataFrame({
        "A": [1, 0],
        "B": [0, 1],
    })
    result = Correlation.process_columns([0, 1], matrix)
    assert result[0]["B"] == 0  # A and B never co-occur
    assert result[1]["A"] == 0


def test_process_columns_complete_overlap():
    """Columns that co-occur in every sample."""
    matrix = pd.DataFrame({
        "A": [1, 2, 3],
        "B": [4, 5, 6],
    })
    result = Correlation.process_columns([0, 1], matrix)
    # All positive -> 3 co-occurrences
    assert result[0]["B"] == 3
    assert result[1]["A"] == 3


# --- process_columns_single tests ---

def test_process_columns_single_returns_dataframe():
    """process_columns_single returns a DataFrame instead of a list."""
    matrix = pd.DataFrame({
        "A": [1, 0, 3],
        "B": [2, 1, 0],
    })
    result = Correlation.process_columns_single(range(2), matrix)
    assert isinstance(result, pd.DataFrame)
    assert result.shape == (2, 2)


def test_process_columns_single_values():
    """Verify values match process_columns."""
    matrix = pd.DataFrame({
        "A": [1, 0, 3],
        "B": [2, 1, 0],
    })
    result = Correlation.process_columns_single(range(2), matrix)
    # Column A: A*A=[1,0,9]->2 pos; B*A=[2,0,0]->1 pos
    assert result.loc["A", "A"] == 2
    assert result.loc["B", "A"] == 1
    # Column B: A*B=[2,0,0]->1 pos; B*B=[4,1,0]->2 pos
    assert result.loc["A", "B"] == 1
    assert result.loc["B", "B"] == 2


def test_process_columns_single_symmetric():
    """Intersection matrix should be symmetric."""
    matrix = pd.DataFrame({
        "A": [1, 0, 2],
        "B": [0, 1, 3],
        "C": [1, 1, 0],
    })
    result = Correlation.process_columns_single(range(3), matrix)
    np.testing.assert_array_equal(result.values, result.values.T)
