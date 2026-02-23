"""
Test cwas.argparser — parser factory functions for each CWAS step.
"""
import argparse
from pathlib import Path

import pytest

import cwas.argparser as ap


# --- start ---

def test_start_returns_parser():
    parser = ap.start()
    assert isinstance(parser, argparse.ArgumentParser)


def test_start_default_workspace():
    parser = ap.start()
    args = parser.parse_args([])
    assert args.workspace == Path.home() / ".cwas"


def test_start_custom_workspace(tmp_path):
    parser = ap.start()
    args = parser.parse_args(["-w", str(tmp_path)])
    assert args.workspace == tmp_path


# --- configuration ---

def test_configuration_returns_parser():
    parser = ap.configuration()
    assert isinstance(parser, argparse.ArgumentParser)


def test_configuration_defaults():
    parser = ap.configuration()
    args = parser.parse_args([])
    assert args.data_dir is None
    assert args.gene_matrix is None
    assert args.force_overwrite == 0


def test_configuration_force_overwrite():
    parser = ap.configuration()
    args = parser.parse_args(["-f"])
    assert args.force_overwrite == 1


def test_configuration_vep_mis_thres():
    parser = ap.configuration()
    args = parser.parse_args(["--vep_mis_thres", "2.5"])
    assert args.vep_mis_thres == pytest.approx(2.5)


# --- preparation ---

def test_preparation_returns_parser():
    parser = ap.preparation()
    assert isinstance(parser, argparse.ArgumentParser)


def test_preparation_defaults():
    parser = ap.preparation()
    args = parser.parse_args([])
    assert args.num_proc == 1
    assert args.force_overwrite == 0


def test_preparation_num_proc():
    parser = ap.preparation()
    args = parser.parse_args(["-p", "4"])
    assert args.num_proc == 4


# --- categorization ---

def test_categorization_returns_parser():
    parser = ap.categorization()
    assert isinstance(parser, argparse.ArgumentParser)


def test_categorization_requires_input():
    parser = ap.categorization()
    with pytest.raises(SystemExit):
        parser.parse_args([])


def test_categorization_defaults():
    parser = ap.categorization()
    args = parser.parse_args(["-i", "/tmp/test.vcf"])
    assert args.num_proc == 1
    assert args.input_path == Path("/tmp/test.vcf")


# --- binomial_test ---

def test_binomial_test_returns_parser():
    parser = ap.binomial_test()
    assert isinstance(parser, argparse.ArgumentParser)


def test_binomial_test_requires_input_and_sample():
    parser = ap.binomial_test()
    with pytest.raises(SystemExit):
        parser.parse_args([])


def test_binomial_test_defaults():
    parser = ap.binomial_test()
    args = parser.parse_args(["-i", "/tmp/c.zarr", "-s", "/tmp/s.txt"])
    assert args.marker_size == pytest.approx(15)
    assert args.font_size == pytest.approx(15)
    assert args.plot_size == pytest.approx(7)
    assert args.use_n_carrier is False
    assert args.eff_test is False
    assert args.tag is None


# --- permutation_test ---

def test_permutation_test_returns_parser():
    parser = ap.permutation_test()
    assert isinstance(parser, argparse.ArgumentParser)


def test_permutation_test_defaults():
    parser = ap.permutation_test()
    args = parser.parse_args(["-i", "/tmp/c.zarr", "-s", "/tmp/s.txt"])
    assert args.num_perm == 10000
    assert args.num_proc == 1
    assert args.burden_shift is False
    assert args.use_n_carrier is False


# --- extract_variant ---

def test_extract_variant_returns_parser():
    parser = ap.extract_variant()
    assert isinstance(parser, argparse.ArgumentParser)


def test_extract_variant_defaults():
    parser = ap.extract_variant()
    args = parser.parse_args(["-i", "/tmp/test.vcf"])
    assert args.tag is None
    assert args.category_set_path is None
    assert args.annotation_info is False


# --- effective_num_test ---

def test_effective_num_test_returns_parser():
    parser = ap.effective_num_test()
    assert isinstance(parser, argparse.ArgumentParser)


def test_effective_num_test_defaults():
    parser = ap.effective_num_test()
    args = parser.parse_args(["-i", "/tmp/m.zarr", "-c_count", "/tmp/c.txt"])
    assert args.input_format == "corr"
    assert args.num_eig == 10000
    assert args.count_thres is None
    assert args.domain_list == "all"
    assert args.tag is None


# --- burden_shift ---

def test_burden_shift_returns_parser():
    parser = ap.burden_shift()
    assert isinstance(parser, argparse.ArgumentParser)


def test_burden_shift_defaults():
    parser = ap.burden_shift()
    args = parser.parse_args([
        "-i", "/tmp/b.txt", "-b", "/tmp/p.parquet",
        "-c_info", "/tmp/ci.txt", "-c_count", "/tmp/cc.txt",
    ])
    assert args.count_cutoff == 7
    assert args.pval == pytest.approx(0.05)
    assert args.n_cat_sets == 10
    assert args.fontsize == 10
    assert args.tag is None


# --- risk_score ---

def test_risk_score_returns_parser():
    parser = ap.risk_score()
    assert isinstance(parser, argparse.ArgumentParser)


def test_risk_score_defaults():
    parser = ap.risk_score()
    args = parser.parse_args(["-i", "/tmp/c.zarr", "-s", "/tmp/s.txt"])
    assert args.domain_list == "all"
    assert args.fold == 5
    assert args.n_permute == 1000
    assert args.seed == 42
    assert args.train_set_f == pytest.approx(0.7)
    assert args.num_reg == 10
    assert args.ctrl_thres == 3
    assert args.num_proc == 1
    assert args.plotsize == "7,7"
    assert args.fontsize == pytest.approx(10)
    assert args.predict_only is False
    assert args.do_each_one is False
    assert args.leave_one_out is False


# --- dawn ---

def test_dawn_returns_parser():
    parser = ap.dawn()
    assert isinstance(parser, argparse.ArgumentParser)


def test_dawn_defaults():
    parser = ap.dawn()
    args = parser.parse_args([
        "-e", "/tmp/ev.txt.gz", "-c", "/tmp/cm.pkl",
        "-P", "/tmp/pt.txt.gz", "-c_count", "/tmp/cc.txt",
    ])
    assert args.seed == 42
    assert args.parsimonious is True
    assert args.lambda_val == pytest.approx(5.25)
    assert args.count_threshold == 20
    assert args.corr_threshold == pytest.approx(0.12)
    assert args.size_threshold == 2
    assert args.k_range == "2,100"
    assert args.k_val is None
    assert args.tsne_method == "exact"
    assert args.num_proc == 1
    assert args.resolution == pytest.approx(1)
    assert args.leiden_clustering is None


def test_dawn_no_parsimonious_flag():
    parser = ap.dawn()
    args = parser.parse_args([
        "-e", "/tmp/ev.txt.gz", "-c", "/tmp/cm.pkl",
        "-P", "/tmp/pt.txt.gz", "-c_count", "/tmp/cc.txt",
        "--no-parsimonious",
    ])
    assert args.parsimonious is False


# --- correlation ---

def test_correlation_returns_parser():
    parser = ap.correlation()
    assert isinstance(parser, argparse.ArgumentParser)


def test_correlation_defaults():
    parser = ap.correlation()
    args = parser.parse_args(["-i", "/tmp/c.zarr", "-cm", "variant"])
    assert args.num_proc == 1
    assert args.generate_inter_matrix is False
    assert args.domain_list == "all"
    assert args.category_info_path is None


def test_correlation_sample_mode():
    parser = ap.correlation()
    args = parser.parse_args(["-i", "/tmp/c.zarr", "-cm", "sample"])
    assert args.generate_corr_matrix == "sample"


# --- All parsers callable ---

ALL_PARSER_NAMES = [
    "start", "configuration", "preparation", "annotation",
    "categorization", "binomial_test", "permutation_test",
    "extract_variant", "effective_num_test", "burden_shift",
    "risk_score", "dawn", "correlation",
]


@pytest.mark.parametrize("name", ALL_PARSER_NAMES)
def test_all_parsers_are_callable(name):
    func = getattr(ap, name)
    parser = func()
    assert isinstance(parser, argparse.ArgumentParser)
