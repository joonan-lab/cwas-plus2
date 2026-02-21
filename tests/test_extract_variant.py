"""
Test cwas.extract_variant
"""
import argparse

import pytest
from pathlib import Path

from cwas.extract_variant import ExtractVariant


class ExtractVariantMock(ExtractVariant):
    """Mock that bypasses validation."""

    @staticmethod
    def _print_args(args):
        pass

    @staticmethod
    def _check_args_validity(args):
        pass


def _make_args(**overrides):
    defaults = dict(
        input_path=Path("/tmp/test.annotated.vcf.gz"),
        output_dir_path=Path("/tmp/output"),
        annotation_info=None,
        tag=None,
        category_set_path=None,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


@pytest.fixture
def ev_inst():
    return ExtractVariantMock(_make_args())


# --- result_path tests ---

def test_result_path_no_tag(ev_inst):
    assert ev_inst.result_path.name == "test.extracted_variants.txt.gz"


def test_result_path_with_tag():
    inst = ExtractVariantMock(_make_args(tag="coding"))
    assert "coding" in inst.result_path.name
    assert inst.result_path.name == "test.coding.extracted_variants.txt.gz"


def test_result_path_vcf_input():
    inst = ExtractVariantMock(_make_args(
        input_path=Path("/tmp/sample.annotated.vcf"),
    ))
    assert inst.result_path.name == "sample.extracted_variants.txt.gz"


def test_result_path_in_output_dir(ev_inst):
    assert ev_inst.result_path.parent == ev_inst.output_dir_path


# --- extract_idx_by_int tests ---

def test_extract_idx_by_int_zero(ev_inst):
    assert ev_inst.extract_idx_by_int(0) == []


def test_extract_idx_by_int_one(ev_inst):
    assert ev_inst.extract_idx_by_int(1) == [0]


def test_extract_idx_by_int_powers_of_two(ev_inst):
    assert ev_inst.extract_idx_by_int(2) == [1]
    assert ev_inst.extract_idx_by_int(4) == [2]
    assert ev_inst.extract_idx_by_int(8) == [3]


def test_extract_idx_by_int_multiple_bits(ev_inst):
    # 5 = 101 in binary -> indices 0, 2
    assert ev_inst.extract_idx_by_int(5) == [0, 2]
    # 7 = 111 -> indices 0, 1, 2
    assert ev_inst.extract_idx_by_int(7) == [0, 1, 2]
    # 10 = 1010 -> indices 1, 3
    assert ev_inst.extract_idx_by_int(10) == [1, 3]


def test_extract_idx_by_int_large(ev_inst):
    # 255 = 11111111 -> indices 0-7
    assert ev_inst.extract_idx_by_int(255) == list(range(8))


# --- Property tests ---

def test_tag_none(ev_inst):
    assert ev_inst.tag is None


def test_tag_set():
    inst = ExtractVariantMock(_make_args(tag="test_tag"))
    assert inst.tag == "test_tag"


def test_annotation_info_none(ev_inst):
    assert ev_inst.annotation_info is None


def test_annotation_info_set():
    inst = ExtractVariantMock(_make_args(annotation_info=True))
    assert inst.annotation_info is True


def test_category_set_path_none(ev_inst):
    assert ev_inst.category_set_path is None


def test_category_set_path_set():
    inst = ExtractVariantMock(_make_args(
        category_set_path=Path("/tmp/catset.txt"),
    ))
    assert inst.category_set_path is not None
