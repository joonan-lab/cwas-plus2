"""
Test cwas.core.categorization.categorizer
"""
import numpy as np
import pandas as pd
import pytest

from cwas.core.categorization.categorizer import Categorizer


@pytest.fixture
def simple_domain():
    """Minimal category domain for testing."""
    return {
        "variant_type": ["All", "SNV", "Indel"],
        "functional_score": ["All"],
        "gene_set": ["Any"],
        "gencode": [
            "Any",
            "CodingRegion",
            "NoncodingRegion",
            "PTVRegion",
            "MissenseRegion",
        ],
        "functional_annotation": ["Any"],
    }


@pytest.fixture
def cat_gene_matrix():
    """Simple gene matrix dict: gene_name -> set of gene sets."""
    return {
        "GENE1": {"ProteinCoding"},
        "GENE2": {"ProteinCoding", "lincRNA"},
        "GENE3": set(),
    }


@pytest.fixture
def categorizer(simple_domain, cat_gene_matrix):
    return Categorizer(simple_domain, cat_gene_matrix, mis_info_key="MPC", mis_thres=2.0)


# --- annotate_variant_type ---

def test_annotate_variant_type_snv(categorizer):
    vcf = pd.DataFrame({
        "REF": ["A"],
        "ALT": ["T"],
    })
    result = categorizer.annotate_variant_type(vcf)
    assert len(result) == 1
    # Should have All + SNV bits set
    # All=index 0 -> 2^0=1, SNV=index 1 -> 2^1=2, total=3
    assert result[0] == 3  # 0b011


def test_annotate_variant_type_indel(categorizer):
    vcf = pd.DataFrame({
        "REF": ["AT"],
        "ALT": ["A"],
    })
    result = categorizer.annotate_variant_type(vcf)
    # All=1, Indel=index 2 -> 2^2=4, total=5
    assert result[0] == 5  # 0b101


def test_annotate_variant_type_mixed(categorizer):
    vcf = pd.DataFrame({
        "REF": ["A", "AT", "G"],
        "ALT": ["T", "A", "C"],
    })
    result = categorizer.annotate_variant_type(vcf)
    assert len(result) == 3
    assert result[0] == 3   # SNV + All
    assert result[1] == 5   # Indel + All
    assert result[2] == 3   # SNV + All


# --- parse_annotation_int ---

def test_parse_annotation_int_variant_type(categorizer):
    # 3 = 0b011 -> All, SNV
    result = categorizer.parse_annotation_int(3, "variant_type")
    assert result == ["All", "SNV"]


def test_parse_annotation_int_single_bit(categorizer):
    # 1 = 0b001 -> All only
    result = categorizer.parse_annotation_int(1, "variant_type")
    assert result == ["All"]


def test_parse_annotation_int_zero(categorizer):
    result = categorizer.parse_annotation_int(0, "variant_type")
    assert result == []


# --- get_all_category_names ---

def test_get_all_category_names(categorizer):
    names = categorizer.get_all_category_names()
    assert isinstance(names, list)
    assert len(names) > 0
    # All names should be underscore-delimited with 5 parts
    for name in names:
        parts = name.split("_")
        assert len(parts) == 5, f"Expected 5 parts, got {len(parts)} in '{name}'"


def test_get_all_category_names_contains_known(categorizer):
    names = categorizer.get_all_category_names()
    assert "All_Any_All_Any_Any" in names
    assert "SNV_Any_All_Any_Any" in names
    assert "Indel_Any_All_Any_Any" in names


def test_get_all_category_names_count(categorizer):
    """Total categories = product of all domain sizes."""
    names = categorizer.get_all_category_names()
    # 3 variant_type * 1 gene_set * 1 functional_score * 5 gencode * 1 functional_annotation = 15
    assert len(names) == 15


# --- _build_group_terms ---

def test_build_group_terms(categorizer):
    terms = categorizer._build_group_terms()
    assert len(terms) == 5
    assert terms[0] == ["All", "SNV", "Indel"]  # variant_type
    assert terms[1] == ["Any"]  # gene_set
    assert "All" in terms[2]  # functional_score (always has "All")
    assert "Any" in terms[3]  # gencode (always has "Any")
    assert "Any" in terms[4]  # functional_annotation (always has "Any")
