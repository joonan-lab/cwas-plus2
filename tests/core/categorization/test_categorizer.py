"""
Test cwas.core.categorization.categorizer
"""
import numpy as np
import pandas as pd
import pytest

from cwas.core.categorization.categorizer import Categorizer
from cwas.core.configuration.settings import get_default_domains


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
    """Simple gene matrix dict: gene ID -> set of gene sets."""
    return {
        "ENSG00000000001": {"ProteinCoding"},
        "ENSG00000000002": {"ProteinCoding", "lincRNA"},
        "ENSG00000000003": set(),
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


# --- annotate_gene_set ---

@pytest.fixture
def gene_set_categorizer():
    """Categorizer whose gene matrix is keyed by gene ID, not by symbol."""
    domain = {
        "variant_type": ["All"],
        "functional_score": ["All"],
        # Any=bit 0, ProteinCoding=bit 1, ASDTADAFDR03=bit 2
        "gene_set": ["Any", "ProteinCoding", "ASDTADAFDR03"],
        "gencode": ["Any"],
        "functional_annotation": ["Any"],
    }
    gene_matrix = {
        "ENSG00000000001": {"ProteinCoding", "ASDTADAFDR03"},
        "ENSG00000000002": {"ProteinCoding"},
    }
    return Categorizer(domain, gene_matrix, mis_info_key="MPC", mis_thres=2.0)


def _gene_set_vcf(gene: str, nearest: str, consequence: str) -> pd.DataFrame:
    return pd.DataFrame({
        "Gene": [gene],
        "NEAREST": [nearest],
        "Consequence": [consequence],
    })


def test_annotate_gene_set_matches_gene_id(gene_set_categorizer):
    vcf = _gene_set_vcf("ENSG00000000001", "", "missense_variant")
    result = gene_set_categorizer.annotate_gene_set(vcf)
    assert result[0] == 0b111  # Any + ProteinCoding + ASDTADAFDR03


def test_annotate_gene_set_ignores_gene_id_version(gene_set_categorizer):
    """VEP reports unversioned IDs, but a versioned one must still match."""
    vcf = _gene_set_vcf("ENSG00000000002.7", "", "missense_variant")
    result = gene_set_categorizer.annotate_gene_set(vcf)
    assert result[0] == 0b011  # Any + ProteinCoding


def test_annotate_gene_set_uses_nearest_for_intergenic(gene_set_categorizer):
    vcf = _gene_set_vcf("", "ENSG00000000001", "intergenic_variant")
    result = gene_set_categorizer.annotate_gene_set(vcf)
    assert result[0] == 0b111


def test_annotate_gene_set_uses_nearest_for_downstream(gene_set_categorizer):
    vcf = _gene_set_vcf("", "ENSG00000000002", "downstream_gene_variant")
    result = gene_set_categorizer.annotate_gene_set(vcf)
    assert result[0] == 0b011


def test_annotate_gene_set_does_not_match_gene_symbol(gene_set_categorizer):
    """A gene symbol must no longer be treated as a gene matrix key."""
    vcf = _gene_set_vcf("GENE1", "", "missense_variant")
    result = gene_set_categorizer.annotate_gene_set(vcf)
    assert result[0] == 0b001  # Any only


def test_annotate_gene_set_unknown_gene(gene_set_categorizer):
    vcf = _gene_set_vcf("ENSG00000009999", "", "missense_variant")
    result = gene_set_categorizer.annotate_gene_set(vcf)
    assert result[0] == 0b001  # Any only


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


@pytest.fixture
def tied_gene_categorizer():
    """Categorizer whose matrix can rank genes tied for nearest."""
    domain = {
        "variant_type": ["All"],
        "functional_score": ["All"],
        # Any=0, ProteinCoding=1, lincRNA=2, ASDTADAFDR03=3, DDD=4
        "gene_set": [
            "Any",
            "ProteinCoding",
            "lincRNA",
            "ASDTADAFDR03",
            "DDD",
        ],
        "gencode": ["Any"],
        "functional_annotation": ["Any"],
    }
    gene_matrix = {
        "ENSG_PC": {"ProteinCoding"},
        "ENSG_RICH": {"lincRNA", "ASDTADAFDR03", "DDD"},
        "ENSG_POOR": {"lincRNA"},
    }
    return Categorizer(domain, gene_matrix, mis_info_key="MPC", mis_thres=2.0)


def test_annotate_gene_set_resolves_tie_to_protein_coding(
    tied_gene_categorizer,
):
    """VEP joins genes tied for nearest with '&'; the matrix breaks the tie.

    Without this the whole '&'-joined string is looked up, matches nothing,
    and the variant silently keeps only the 'Any' gene set.
    """
    vcf = _gene_set_vcf("", "ENSG_RICH&ENSG_PC", "intergenic_variant")
    result = tied_gene_categorizer.annotate_gene_set(vcf)
    # Any + ProteinCoding, not the gene lists of ENSG_RICH.
    assert result[0] == 0b00011


def test_annotate_gene_set_resolves_tie_by_gene_list_count(
    tied_gene_categorizer,
):
    vcf = _gene_set_vcf("", "ENSG_POOR&ENSG_RICH", "downstream_gene_variant")
    result = tied_gene_categorizer.annotate_gene_set(vcf)
    # Any + ASDTADAFDR03 + DDD. 'lincRNA' is dropped from the gene set
    # domain by annotate_gene_set, so it is never a bit here.
    assert result[0] == 0b11001


def test_annotate_gene_set_resolves_tie_with_versioned_ids(
    tied_gene_categorizer,
):
    """Each ID of a tie must lose its version before it is ranked."""
    vcf = _gene_set_vcf("", "ENSG_RICH.3&ENSG_PC.12", "intergenic_variant")
    result = tied_gene_categorizer.annotate_gene_set(vcf)
    assert result[0] == 0b00011


# --- annotate_gencode ---

# The real domain, so that the bits under test are the ones CWAS-Plus2 uses.
GENCODE_DOMAIN = get_default_domains()["gencode"]


@pytest.fixture
def gencode_categorizer():
    """Categorizer whose gene matrix is keyed by gene ID, not by symbol."""
    domain = {
        "variant_type": ["All"],
        "functional_score": ["All"],
        "gene_set": ["Any"],
        "gencode": GENCODE_DOMAIN,
        "functional_annotation": ["Any"],
    }
    gene_matrix = {
        "ENSG_PC": {"ProteinCoding"},
        "ENSG_LINC": {"lincRNA"},
        "ENSG_OTHER": {"SomeGeneList"},
        "ENSG_RICH": {"lincRNA", "ASDTADAFDR03", "DDD"},
    }
    return Categorizer(domain, gene_matrix, mis_info_key="MPC", mis_thres=2.0)


def _gencode_vcf(
    gene, nearest, consequence, lof="", lof_flags="", mis_score=""
) -> pd.DataFrame:
    return pd.DataFrame({
        "Gene": [gene],
        "NEAREST": [nearest],
        "Consequence": [consequence],
        "LoF": [lof],
        "LoF_flags": [lof_flags],
        "MisDb_MPC": [mis_score],
    })


def _regions(annotation_int: int) -> set:
    """Name the gencode regions an annotation integer holds."""
    return {
        name
        for i, name in enumerate(GENCODE_DOMAIN)
        if annotation_int & (1 << i)
    }


def test_annotate_gencode_matches_gene_id(gencode_categorizer):
    """The gene matrix decides CodingRegion, and it is keyed by gene ID."""
    vcf = _gencode_vcf("ENSG_PC", "", "missense_variant")
    result = gencode_categorizer.annotate_gencode(vcf)
    assert _regions(result[0]) == {"Any", "CodingRegion", "MissenseRegion"}


def test_annotate_gencode_does_not_match_a_gene_symbol(gencode_categorizer):
    """A symbol must no longer reach the gene matrix.

    Looking a symbol up finds nothing, so 'ProteinCoding' is absent and a
    coding variant is filed as noncoding instead. Nothing fails while it
    happens.
    """
    vcf = _gencode_vcf("SAMD11", "", "missense_variant")
    result = gencode_categorizer.annotate_gencode(vcf)

    regions = _regions(result[0])
    assert "CodingRegion" not in regions
    assert "MissenseRegion" not in regions
    assert "NoncodingRegion" in regions


def test_annotate_gencode_ignores_the_gene_id_version(gencode_categorizer):
    vcf = _gencode_vcf("ENSG_PC.17", "", "synonymous_variant")
    result = gencode_categorizer.annotate_gencode(vcf)
    assert _regions(result[0]) == {"Any", "CodingRegion", "SilentRegion"}


@pytest.mark.parametrize("consequence", [
    "intergenic_variant", "downstream_gene_variant",
])
def test_annotate_gencode_uses_nearest_when_there_is_no_gene(
    gencode_categorizer, consequence
):
    """An intergenic variant is named by NEAREST, not by an empty Gene."""
    vcf = _gencode_vcf("", "ENSG_PC", consequence)
    result = gencode_categorizer.annotate_gencode(vcf)
    assert _regions(result[0]) == {
        "Any", "NoncodingRegion", "IntergenicRegion",
    }


def test_annotate_gencode_resolves_a_tied_nearest(
    gencode_categorizer, monkeypatch
):
    """VEP joins genes tied for nearest with '&'; the matrix breaks the tie.

    The whole '&'-joined string matches no gene, so without the tiebreak the
    variant loses whatever its gene lists would have said.
    """
    import cwas.core.categorization.categorizer as categorizer_module

    selected = []
    original_resolver = categorizer_module.resolve_tied_gene_id

    def record_selection(gene_id, gene_matrix):
        result = original_resolver(gene_id, gene_matrix)
        selected.append(result)
        return result

    monkeypatch.setattr(
        categorizer_module, "resolve_tied_gene_id", record_selection
    )
    vcf = _gencode_vcf("", "ENSG_RICH&ENSG_PC", "intergenic_variant")
    result = gencode_categorizer.annotate_gencode(vcf)

    # Both candidates lead to the same intergenic region bits, so the bits
    # alone cannot prove that the tie was resolved. Verify the selected ID too:
    # ProteinCoding must beat the richer lincRNA gene listed first by VEP.
    assert selected == ["ENSG_PC"]
    assert _regions(result[0]) == {
        "Any", "NoncodingRegion", "IntergenicRegion",
    }


def test_annotate_gencode_ptv_needs_high_confidence_lof(gencode_categorizer):
    vcf = _gencode_vcf("ENSG_PC", "", "stop_gained", lof="HC")
    assert _regions(gencode_categorizer.annotate_gencode(vcf)[0]) == {
        "Any", "CodingRegion", "PTVRegion",
    }

    low = _gencode_vcf("ENSG_PC", "", "stop_gained", lof="LC")
    assert _regions(gencode_categorizer.annotate_gencode(low)[0]) == {
        "Any", "CodingRegion",
    }


def test_annotate_gencode_damaging_missense_needs_the_threshold(
    gencode_categorizer,
):
    """mis_thres is 2.0 for this categorizer."""
    over = _gencode_vcf("ENSG_PC", "", "missense_variant", mis_score="2.5")
    assert "DamagingMissenseRegion" in _regions(
        gencode_categorizer.annotate_gencode(over)[0]
    )

    under = _gencode_vcf("ENSG_PC", "", "missense_variant", mis_score="1.5")
    assert "DamagingMissenseRegion" not in _regions(
        gencode_categorizer.annotate_gencode(under)[0]
    )


def test_annotate_gencode_splits_noncoding_by_gene_biotype(
    gencode_categorizer,
):
    """A noncoding variant falls to lincRNA or to the catch-all by gene list."""
    linc = _gencode_vcf("ENSG_LINC", "", "non_coding_transcript_exon_variant")
    assert _regions(gencode_categorizer.annotate_gencode(linc)[0]) == {
        "Any", "NoncodingRegion", "lincRnaRegion",
    }

    other = _gencode_vcf("ENSG_OTHER", "", "non_coding_transcript_exon_variant")
    assert _regions(gencode_categorizer.annotate_gencode(other)[0]) == {
        "Any", "NoncodingRegion", "OtherTranscriptRegion",
    }


def test_annotate_gencode_unknown_gene_is_noncoding(gencode_categorizer):
    """A gene the matrix does not know cannot be called coding."""
    vcf = _gencode_vcf("ENSG_MISSING", "", "intron_variant")
    assert _regions(gencode_categorizer.annotate_gencode(vcf)[0]) == {
        "Any", "NoncodingRegion", "IntronRegion",
    }


def test_annotate_gene_set_tie_of_unknown_genes(tied_gene_categorizer):
    """A tie the matrix knows nothing about still resolves, to the first."""
    vcf = _gene_set_vcf("", "ENSG_X&ENSG_Y", "intergenic_variant")
    result = tied_gene_categorizer.annotate_gene_set(vcf)
    assert result[0] == 0b00001  # Any only
