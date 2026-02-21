"""
Test cwas.burden_test
"""
import argparse

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from cwas.burden_test import BurdenTest, _contain_same_index, apply_region_mapping


class BurdenTestConcrete(BurdenTest):
    """Concrete subclass for testing the abstract BurdenTest."""

    @staticmethod
    def _print_args(args):
        pass

    @staticmethod
    def _check_args_validity(args):
        pass

    def run_burden_test(self):
        pass

    def save_result(self):
        pass

    def update_env(self):
        pass


@pytest.fixture
def burden_inst():
    """Create a BurdenTestConcrete with synthetic pre-loaded data."""
    sample_ids = ["S1", "S2", "S3", "S4"]
    categories = ["a_b_c_d_e", "f_g_h_i_j"]
    data = np.array([
        [10, 5],   # S1 - case
        [0, 3],    # S2 - ctrl
        [7, 0],    # S3 - case
        [2, 8],    # S4 - ctrl
    ], dtype=np.float64)

    args = argparse.Namespace(
        num_proc=1,
        cat_path=Path("test.categorization_result.zarr"),
        output_dir_path=Path("/tmp"),
        sample_info_path=Path("/tmp/sample.txt"),
        adj_factor_path=Path("/tmp/adj.txt"),
        use_n_carrier=False,
        eff_test=False,
        tag=None,
        plot_size=7, plot_title="test", font_size=15, marker_size=15,
    )
    inst = BurdenTestConcrete(args)
    inst._categorization_result = pd.DataFrame(
        data, index=sample_ids, columns=categories
    )
    inst._categorization_result.index.name = "SAMPLE"
    inst._sample_info = pd.DataFrame(
        {"PHENOTYPE": ["case", "ctrl", "case", "ctrl"]},
        index=pd.Index(sample_ids, name="SAMPLE"),
    )
    inst._adj_factor = pd.DataFrame(
        {"AdjustFactor": [1.0, 1.0, 1.0, 1.0]},
        index=pd.Index(sample_ids, name="SAMPLE"),
    )
    inst._raw_counts = pd.DataFrame(
        {"Raw_counts": inst._categorization_result.sum(axis=0)},
        index=pd.Index(categories, name="Category"),
    )
    return inst


# --- Standalone function tests ---

def test_contain_same_index_true():
    df1 = pd.DataFrame(index=["a", "b", "c"])
    df2 = pd.DataFrame(index=["c", "a", "b"])
    assert _contain_same_index(df1, df2) is True


def test_contain_same_index_false():
    df1 = pd.DataFrame(index=["a", "b", "c"])
    df2 = pd.DataFrame(index=["a", "b", "d"])
    assert _contain_same_index(df1, df2) is False


def test_contain_same_index_different_lengths():
    df1 = pd.DataFrame(index=["a", "b"])
    df2 = pd.DataFrame(index=["a", "b", "c"])
    assert _contain_same_index(df1, df2) is False


def test_apply_region_mapping_coding():
    df = pd.DataFrame({
        "variant_type": ["SNV", "SNV", "SNV"],
        "gene_set": ["All", "All", "All"],
        "functional_score": ["All", "All", "All"],
        "gencode": ["CodingRegion", "MissenseRegion", "PTVRegion"],
        "functional_annotation": ["All", "All", "All"],
    })
    result = apply_region_mapping(df)
    assert result["is_coding"].tolist() == [1, 1, 1]
    assert result["is_PTV"].tolist() == [0, 0, 1]
    assert result["is_missense"].tolist() == [0, 1, 0]
    assert result["is_coding_no_ptv"].tolist() == [0, 1, 0]


def test_apply_region_mapping_noncoding():
    df = pd.DataFrame({
        "variant_type": ["SNV", "SNV", "SNV"],
        "gene_set": ["All", "All", "lincRNA"],
        "functional_score": ["All", "All", "All"],
        "gencode": ["NoncodingRegion", "PromoterRegion", "lincRnaRegion"],
        "functional_annotation": ["All", "All", "All"],
    })
    result = apply_region_mapping(df)
    assert result["is_noncoding"].tolist() == [1, 1, 1]
    assert result["is_promoter"].tolist() == [0, 1, 0]
    assert result["is_lincRNA"].tolist() == [0, 0, 1]
    assert result["is_noncoding_wo_promoter"].tolist() == [0, 0, 1]


def test_apply_region_mapping_utr():
    df = pd.DataFrame({
        "variant_type": ["SNV", "SNV"],
        "gene_set": ["All", "All"],
        "functional_score": ["All", "All"],
        "gencode": ["3PrimeUTRsRegion", "5PrimeUTRsRegion"],
        "functional_annotation": ["All", "All"],
    })
    result = apply_region_mapping(df)
    assert result["is_3primeUTR"].tolist() == [1, 0]
    assert result["is_5primeUTR"].tolist() == [0, 1]
    assert result["is_UTR"].tolist() == [1, 1]


# --- Property tests ---

def test_phenotypes(burden_inst):
    expected = np.array(["case", "ctrl", "case", "ctrl"])
    np.testing.assert_array_equal(burden_inst.phenotypes, expected)


def test_case_ctrl_cnt(burden_inst):
    assert burden_inst.case_cnt == 2
    assert burden_inst.ctrl_cnt == 2


def test_case_variant_cnt(burden_inst):
    # Cases S1=[10,5], S3=[7,0] -> sum=[17,5]
    np.testing.assert_array_equal(burden_inst.case_variant_cnt, [17.0, 5.0])


def test_ctrl_variant_cnt(burden_inst):
    # Ctrls S2=[0,3], S4=[2,8] -> sum=[2,11]
    np.testing.assert_array_equal(burden_inst.ctrl_variant_cnt, [2.0, 11.0])


def test_is_carrier(burden_inst):
    expected = np.array([[1, 1], [0, 1], [1, 0], [1, 1]])
    np.testing.assert_array_equal(burden_inst.is_carrier, expected)


def test_case_carrier_cnt(burden_inst):
    # Cases: S1=[1,1], S3=[1,0] -> [2,1]
    np.testing.assert_array_equal(burden_inst.case_carrier_cnt, [2, 1])


def test_ctrl_carrier_cnt(burden_inst):
    # Ctrls: S2=[0,1], S4=[1,1] -> [1,2]
    np.testing.assert_array_equal(burden_inst.ctrl_carrier_cnt, [1, 2])


def test_category_table(burden_inst):
    ct = burden_inst.category_table
    assert list(ct.columns) == [
        "variant_type", "gene_set", "functional_score",
        "gencode", "functional_annotation",
    ]
    assert ct.loc["a_b_c_d_e", "variant_type"] == "a"
    assert ct.loc["a_b_c_d_e", "functional_annotation"] == "e"
    assert ct.loc["f_g_h_i_j", "variant_type"] == "f"
    assert ct.loc["f_g_h_i_j", "functional_annotation"] == "j"


# --- Counting and relative risk method tests ---

def test_count_variant_for_each_category(burden_inst):
    burden_inst.count_variant_for_each_category()
    result = burden_inst._result
    assert list(result.columns) == ["Case_DNV_Count", "Ctrl_DNV_Count"]
    assert result["Case_DNV_Count"].tolist() == pytest.approx([17.0, 5.0])
    assert result["Ctrl_DNV_Count"].tolist() == pytest.approx([2.0, 11.0])


def test_count_carrier_for_each_category(burden_inst):
    burden_inst.count_carrier_for_each_category()
    result = burden_inst._result
    assert list(result.columns) == ["Case_Carrier_Count", "Ctrl_Carrier_Count"]
    assert result["Case_Carrier_Count"].tolist() == [2, 1]
    assert result["Ctrl_Carrier_Count"].tolist() == [1, 2]


def test_calculate_relative_risk(burden_inst):
    burden_inst.count_variant_for_each_category()
    burden_inst.calculate_relative_risk()
    rr = burden_inst._result["Relative_Risk"]
    expected = [(17.0 / 2) / (2.0 / 2), (5.0 / 2) / (11.0 / 2)]
    assert rr.tolist() == pytest.approx(expected)


def test_calculate_relative_risk_with_n_carrier(burden_inst):
    burden_inst.count_carrier_for_each_category()
    burden_inst.calculate_relative_risk_with_n_carrier()
    rr = burden_inst._result["Relative_Risk"]
    expected = [(2 / 2) / (1 / 2), (1 / 2) / (2 / 2)]
    assert rr.tolist() == pytest.approx(expected)


def test_concat_category_info(burden_inst):
    burden_inst.count_variant_for_each_category()
    burden_inst.calculate_relative_risk()
    burden_inst.concat_category_info()
    result = burden_inst._result
    assert result.index.name == "Category"
    assert "variant_type" in result.columns
    assert "Case_DNV_Count" in result.columns
    assert "Relative_Risk" in result.columns


# --- Path property tests ---

def test_result_path(burden_inst):
    assert burden_inst.result_path.name == "test.burden_test.txt"


def test_counts_path(burden_inst):
    assert burden_inst.counts_path.name == "test.category_counts.txt"


def test_cat_info_path(burden_inst):
    assert burden_inst.cat_info_path.name == "test.category_info.txt"


# --- Adjustment tests ---

def test_adjust_categorization_result(burden_inst):
    burden_inst._adj_factor = pd.DataFrame(
        {"AdjustFactor": [2.0, 0.5, 1.5, 0.8]},
        index=pd.Index(["S1", "S2", "S3", "S4"], name="SAMPLE"),
    )
    burden_inst._adjust_categorization_result()
    result = burden_inst._categorization_result
    # S1: [10*2.0, 5*2.0] = [20, 10]
    assert result.loc["S1"].tolist() == pytest.approx([20.0, 10.0])
    # S2: [0*0.5, 3*0.5] = [0, 1.5]
    assert result.loc["S2"].tolist() == pytest.approx([0.0, 1.5])


def test_adjust_categorization_result_mismatched_samples(burden_inst):
    burden_inst._adj_factor = pd.DataFrame(
        {"AdjustFactor": [1.0]},
        index=pd.Index(["X1"], name="SAMPLE"),
    )
    with pytest.raises(ValueError, match="sample IDs"):
        burden_inst._adjust_categorization_result()


# --- save_counts_table tests ---

def test_save_counts_table_raw_variant(burden_inst):
    burden_inst.save_counts_table("raw")
    assert burden_inst._raw_counts is not None
    assert burden_inst._raw_counts.index.name == "Category"
    # Total variant counts: [10+0+7+2, 5+3+0+8] = [19, 16]
    assert burden_inst._raw_counts["Raw_counts"].tolist() == pytest.approx([19.0, 16.0])


def test_save_counts_table_raw_carrier(burden_inst):
    burden_inst._args.use_n_carrier = True
    burden_inst.save_counts_table("raw")
    # Carrier counts: (>0) per column then sum across samples: [3, 3]
    assert burden_inst._raw_counts["Raw_counts"].tolist() == [3, 3]
