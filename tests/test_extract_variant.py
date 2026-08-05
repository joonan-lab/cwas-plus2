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


# --- annotate_variants tests ---

ANNOTATED_VCF = Path(__file__).parent / "test_file" / "test_annotated.vcf.gz"

# SAMD11 is the nearest gene of every variant of the fixture. The two other
# genes are left out of the matrix on purpose, to exercise the fallbacks for a
# gene the matrix does not know.
_MATRIX_LINES = [
    ["gene_id", "gene_name", "ProteinCoding", "lincRNA", "asd1"],
    ["ENSG00000187634.13", "SAMD11", "1", "0", "1"],
    ["ENSG00000272438.2", "AL645608.2", "1", "0", "0"],
]


def _annotate_inst(tmp_path, monkeypatch, lines=None):
    matrix = tmp_path / "gene_matrix.txt"
    matrix.write_text(
        "\n".join("\t".join(row) for row in (lines or _MATRIX_LINES)) + "\n"
    )
    env = {
        "GENE_MATRIX": str(matrix),
        "VEP_MIS_INFO_KEY": "MPC",
        "VEP_MIS_THRES": "2",
    }
    inst = ExtractVariantMock(_make_args(input_path=ANNOTATED_VCF))
    monkeypatch.setattr(inst, "get_env", lambda key: env[key])
    return inst


def test_annotate_variants_preserves_the_row_count(tmp_path, monkeypatch):
    """The annotation block is concatenated by position, not by key.

    It is built from annotated_vcf while the rest is built from the merge with
    the gene matrix, so a merge that multiplies rows would shift every variant
    against its own annotations without failing.
    """
    inst = _annotate_inst(tmp_path, monkeypatch)
    inst.annotate_variants()

    assert len(inst._result) == len(inst.annotated_vcf)


def test_annotate_variants_survives_a_gene_matrix_of_collapsing_ids(
    tmp_path, monkeypatch
):
    """Two rows of one gene must not multiply the variants of that gene.

    Stripping the version and '_PAR_Y' suffixes collapses rows onto one ID, so
    a matrix that lists a gene twice would otherwise duplicate every variant
    the merge matches to it.
    """
    inst = _annotate_inst(tmp_path, monkeypatch, [
        ["gene_id", "gene_name", "ProteinCoding", "lincRNA", "asd1"],
        ["ENSG00000187634.12", "SAMD11", "1", "0", "0"],
        ["ENSG00000187634.13", "SAMD11", "1", "0", "1"],
        ["ENSG00000272438.2", "AL645608.2", "1", "0", "0"],
    ])
    inst.annotate_variants()

    assert len(inst._result) == len(inst.annotated_vcf)


def test_annotate_variants_keeps_the_annot_block_aligned(
    tmp_path, monkeypatch
):
    """Every variant must carry the annotations of its own ANNOT integer."""
    inst = _annotate_inst(tmp_path, monkeypatch)
    inst.annotate_variants()

    fields = inst.annot_field_names
    for i in range(len(inst.annotated_vcf)):
        expected = set(
            inst.extract_idx_by_int(int(inst.annotated_vcf.loc[i, "ANNOT"]))
        )
        row = inst._result.iloc[i]
        assert {j for j, f in enumerate(fields) if row[f] == 1} == expected


def test_annotate_variants_takes_the_gene_id_of_intergenic_from_nearest(
    tmp_path, monkeypatch
):
    """An intergenic variant has no gene of its own, so NEAREST names it."""
    inst = _annotate_inst(tmp_path, monkeypatch)
    vcf = inst.annotated_vcf
    is_intergenic = vcf["Consequence"].str.contains(
        "downstream_gene_variant|intergenic_variant"
    )
    assert is_intergenic.any(), "the fixture must hold an intergenic variant"

    inst.annotate_variants()
    gene_ids = inst._result["DEF.GENE_ID"]

    assert (gene_ids[is_intergenic] == vcf.loc[is_intergenic, "NEAREST"]).all()
    assert (gene_ids[~is_intergenic] == vcf.loc[~is_intergenic, "Gene"]).all()


def test_annotate_variants_labels_with_the_gene_symbol(tmp_path, monkeypatch):
    """DEF.GENE reports the symbol, and the ID when the matrix lacks the gene."""
    inst = _annotate_inst(tmp_path, monkeypatch)
    inst.annotate_variants()
    result = inst._result

    known = result["DEF.GENE_ID"] == "ENSG00000187634"
    assert known.any()
    assert (result.loc[known, "DEF.GENE"] == "SAMD11").all()

    # ENSG00000230021 is deliberately absent from the matrix.
    unknown = result["DEF.GENE_ID"] == "ENSG00000230021"
    assert unknown.any()
    assert (result.loc[unknown, "DEF.GENE"] == "ENSG00000230021").all()


def test_annotate_variants_strips_the_gene_id_version(tmp_path, monkeypatch):
    """VEP reports unversioned IDs, the matrix carries versioned ones."""
    inst = _annotate_inst(tmp_path, monkeypatch)
    inst.annotate_variants()

    assert not inst._result["DEF.GENE_ID"].str.contains(r"\.", regex=True).any()
    # The gene lists of the matrix reached the variants of that gene.
    matched = inst._result["DEF.GENE_ID"] == "ENSG00000187634"
    assert (inst._result.loc[matched, "asd1"] == 1).all()


def test_annotate_variants_resolves_tied_nearest_to_one_gene(
    tmp_path, monkeypatch
):
    """The full extraction path must use the same winner as categorization."""
    first = "ENSG00000000001"
    winner = "ENSG00000000002"
    inst = _annotate_inst(tmp_path, monkeypatch, [
        ["gene_id", "gene_name", "ProteinCoding", "lincRNA", "asd1"],
        [first, "FIRST", "0", "1", "1"],
        [winner, "WINNER", "1", "0", "0"],
    ])
    annotated_vcf = inst.annotated_vcf.copy()
    intergenic = annotated_vcf["Consequence"].str.contains(
        "downstream_gene_variant|intergenic_variant"
    )
    row_idx = annotated_vcf.index[intergenic][0]
    annotated_vcf.loc[row_idx, "NEAREST"] = f"{first}&{winner}"
    inst._annotated_vcf = annotated_vcf

    inst.annotate_variants()
    row = inst._result.loc[row_idx]

    # The second candidate is ProteinCoding, so it must win even though VEP
    # listed it second. ID, readable symbol, and memberships must all come from
    # that one selected row of the gene matrix.
    assert row["DEF.GENE_ID"] == winner
    assert row["DEF.GENE"] == "WINNER"
    assert row["ProteinCoding"] == 1
    assert row["lincRNA"] == 0
    assert row["asd1"] == 0


# --- gene_matrix tests ---

def _gene_matrix_inst(tmp_path, monkeypatch, lines):
    path = tmp_path / "gene_matrix.txt"
    path.write_text("\n".join("\t".join(row) for row in lines) + "\n")
    inst = ExtractVariantMock(_make_args())
    monkeypatch.setattr(inst, "get_env", lambda key: str(path))
    return inst


def test_gene_matrix_strips_gene_id_version(tmp_path, monkeypatch):
    inst = _gene_matrix_inst(tmp_path, monkeypatch, [
        ["gene_id", "gene_name", "ProteinCoding"],
        ["ENSG00000000003.15", "TSPAN6", "1"],
    ])
    assert inst.gene_matrix["gene_id"].tolist() == ["ENSG00000000003"]


def test_gene_matrix_deduplicates_gene_ids(tmp_path, monkeypatch):
    """Stripping the version suffix can collapse two rows onto one gene ID.

    A duplicated merge key would multiply variant rows in annotate_variants
    and desynchronize the annotation block appended positionally afterwards,
    so the matrix must be de-duplicated, keeping the last entry.
    """
    inst = _gene_matrix_inst(tmp_path, monkeypatch, [
        ["gene_id", "gene_name", "ProteinCoding"],
        ["ENSG00000000003.14", "TSPAN6", "1"],
        ["ENSG00000000003.15", "TSPAN6", "0"],
        ["ENSG00000000005.6", "TNMD", "1"],
    ])
    gene_matrix = inst.gene_matrix
    assert gene_matrix["gene_id"].tolist() == [
        "ENSG00000000003",
        "ENSG00000000005",
    ]
    # The last entry of a duplicated ID wins.
    assert gene_matrix["ProteinCoding"].tolist() == [0, 1]


def test_gene_matrix_symbol_index_is_unique(tmp_path, monkeypatch):
    """The symbol lookup is used with Series.map, which needs a unique index."""
    inst = _gene_matrix_inst(tmp_path, monkeypatch, [
        ["gene_id", "gene_name", "ProteinCoding"],
        ["ENSG00000000003.14", "TSPAN6", "1"],
        ["ENSG00000000003.15", "TSPAN6", "0"],
    ])
    _ = inst.gene_matrix
    assert inst._gene_symbols.index.is_unique


def test_gene_matrix_accepts_alternative_column_names(tmp_path, monkeypatch):
    """The columns accepted here must match those categorization accepts."""
    inst = _gene_matrix_inst(tmp_path, monkeypatch, [
        ["ensembl_gene_id", "symbol", "ProteinCoding", "lincRNA"],
        ["ENSG00000000003.15", "TSPAN6", "1", "0"],
    ])
    gene_matrix = inst.gene_matrix
    assert gene_matrix["gene_id"].tolist() == ["ENSG00000000003"]
    # The symbol column is a label, never a gene list.
    assert "symbol" not in gene_matrix.columns
    assert gene_matrix.columns.tolist() == [
        "gene_id",
        "ProteinCoding",
        "lincRNA",
    ]
    assert inst._gene_symbols["ENSG00000000003"] == "TSPAN6"


def test_gene_matrix_without_symbol_column(tmp_path, monkeypatch):
    inst = _gene_matrix_inst(tmp_path, monkeypatch, [
        ["gene_id", "ProteinCoding"],
        ["ENSG00000000003.15", "1"],
    ])
    assert inst.gene_matrix.columns.tolist() == ["gene_id", "ProteinCoding"]
    assert inst._gene_symbols is None


def test_gene_matrix_without_gene_id_column(tmp_path, monkeypatch):
    inst = _gene_matrix_inst(tmp_path, monkeypatch, [
        ["gene_name", "ProteinCoding"],
        ["TSPAN6", "1"],
    ])
    with pytest.raises(ValueError):
        _ = inst.gene_matrix


def test_gene_matrix_rejects_two_gene_id_columns(tmp_path, monkeypatch):
    """Renaming the ID column used to collide with an existing 'gene_id'.

    That left two columns of one name, so gene_matrix['gene_id'] returned a
    DataFrame and the load died on AttributeError, while categorization read
    the same file happily. Both steps must now refuse it for the same reason.
    """
    inst = _gene_matrix_inst(tmp_path, monkeypatch, [
        ["ensembl_gene_id", "gene_id", "gene_name", "ProteinCoding"],
        ["ENSG00000000003", "ENSG00000000003", "TSPAN6", "1"],
    ])
    with pytest.raises(ValueError) as excinfo:
        _ = inst.gene_matrix
    assert "more than one gene ID column" in str(excinfo.value)


def test_gene_matrix_rejects_an_invalid_gene_id(tmp_path, monkeypatch):
    gene_matrix_path = tmp_path / "gene_matrix.txt"
    inst = _gene_matrix_inst(tmp_path, monkeypatch, [
        ["gene_id", "gene_name", "ProteinCoding"],
        ["SAMD11", "SAMD11", "1"],
    ])

    with pytest.raises(ValueError) as excinfo:
        _ = inst.gene_matrix

    message = str(excinfo.value)
    assert str(gene_matrix_path) in message
    assert "line 2: 'SAMD11'" in message


def test_gene_matrix_accepts_a_utf8_bom(tmp_path, monkeypatch):
    gene_matrix_path = tmp_path / "gene_matrix.txt"
    inst = _gene_matrix_inst(tmp_path, monkeypatch, [
        ["gene_id", "gene_name", "ProteinCoding"],
        ["ENSG00000187634", "SAMD11", "1"],
    ])
    gene_matrix_path.write_text(
        "gene_id\tgene_name\tProteinCoding\n"
        "ENSG00000187634\tSAMD11\t1\n",
        encoding="utf-8-sig",
    )

    result = inst.gene_matrix

    assert result.to_dict("records") == [
        {"gene_id": "ENSG00000187634", "ProteinCoding": 1}
    ]


def test_gene_matrix_tied_gene_sets(tmp_path, monkeypatch):
    """The tie is broken with the same gene lists categorization uses."""
    inst = _gene_matrix_inst(tmp_path, monkeypatch, [
        ["gene_id", "gene_name", "ProteinCoding", "lincRNA", "ASDTADAFDR03"],
        ["ENSG00000000001", "PC", "1", "0", "0"],
        ["ENSG00000000002", "RICH", "0", "1", "1"],
    ])
    from pandas import Series

    gene_sets = inst._tied_gene_sets(
        Series(["ENSG00000000002&ENSG00000000001"])
    )
    assert gene_sets == {
        "ENSG00000000001": {"ProteinCoding"},
        "ENSG00000000002": {"lincRNA", "ASDTADAFDR03"},
    }

    from cwas.core.common import resolve_tied_gene_id

    # The protein-coding gene wins even though VEP listed it second.
    assert (
        resolve_tied_gene_id(
            "ENSG00000000002&ENSG00000000001", gene_sets
        )
        == "ENSG00000000001"
    )


@pytest.mark.parametrize(
    "member,absent", [("1.0", "0.0"), ("TRUE", "FALSE"), ("1", "")]
)
def test_gene_matrix_rejects_values_that_are_not_0_or_1(
    tmp_path, monkeypatch, member, absent
):
    """Both loaders of a gene matrix must reject it for the same reason.

    Membership is decided on the text of a value, so '1.0' and 'TRUE' put no
    gene in any gene list. That failure is silent: the run finishes with every
    variant in the 'Any' gene set alone, so the matrix is rejected instead.
    """
    inst = _gene_matrix_inst(tmp_path, monkeypatch, [
        ["gene_id", "gene_name", "ProteinCoding", "asd1"],
        ["ENSG1", "gene1", member, absent],
    ])
    with pytest.raises(ValueError) as excinfo:
        _ = inst.gene_matrix
    assert "'0' or '1'" in str(excinfo.value)


def test_gene_matrix_gene_lists_stay_numeric(tmp_path, monkeypatch):
    """annotate_variants compares the merged gene lists against the number 1."""
    inst = _gene_matrix_inst(tmp_path, monkeypatch, [
        ["gene_id", "gene_name", "ProteinCoding"],
        ["ENSG1", "gene1", "1"],
    ])
    assert inst.gene_matrix["ProteinCoding"].tolist() == [1]


def test_tied_gene_sets_agree_with_categorization(tmp_path, monkeypatch):
    """One gene matrix must mean one thing to both steps.

    extract_variant reads the matrix with pandas and categorization reads its
    text, so the two once decided membership differently: 'row[col] == 1'
    against '== "1"'. A matrix pandas typed differently than the text read
    then sent the two steps to different genes of the same tie.
    """
    from io import StringIO

    from cwas.core.categorization.parser import _parse_gene_matrix

    lines = [
        ["gene_id", "gene_name", "ProteinCoding", "lincRNA", "asd1"],
        ["ENSG00000000001", "PC", "1", "0", "0"],
        ["ENSG00000000002", "RICH", "0", "1", "1"],
    ]
    inst = _gene_matrix_inst(tmp_path, monkeypatch, lines)
    from pandas import Series

    from_pandas = inst._tied_gene_sets(
        Series(["ENSG00000000001&ENSG00000000002"])
    )
    from_text = _parse_gene_matrix(
        StringIO("\n".join("\t".join(row) for row in lines) + "\n")
    )

    assert from_pandas == from_text


def test_gene_matrix_tied_gene_sets_ignores_untied_genes(tmp_path, monkeypatch):
    """Only the genes appearing in a tie are looked up."""
    inst = _gene_matrix_inst(tmp_path, monkeypatch, [
        ["gene_id", "ProteinCoding"],
        ["ENSG00000000001", "1"],
        ["ENSG00000000002", "1"],
        ["ENSG00000000003", "1"],
    ])
    from pandas import Series

    gene_sets = inst._tied_gene_sets(
        Series(["ENSG00000000001&ENSG00000000003"])
    )
    assert set(gene_sets) == {"ENSG00000000001", "ENSG00000000003"}


def test_gene_matrix_collapses_par_y_rows(tmp_path, monkeypatch):
    """The chrY copy of a pseudoautosomal gene shares the chrX gene ID."""
    inst = _gene_matrix_inst(tmp_path, monkeypatch, [
        ["gene_id", "gene_name", "ProteinCoding"],
        ["ENSG00000182378.14", "PLCXD1", "1"],
        ["ENSG00000182378.14_PAR_Y", "PLCXD1", "1"],
        ["ENSG00000000005.6", "TNMD", "1"],
    ])
    gene_matrix = inst.gene_matrix
    assert gene_matrix["gene_id"].tolist() == [
        "ENSG00000182378",
        "ENSG00000000005",
    ]
    assert inst._gene_symbols.index.is_unique


def test_gene_matrix_par_y_collapse_is_silent(tmp_path, monkeypatch, capsys):
    """Collapsing identical rows is expected, so it must not warn."""
    inst = _gene_matrix_inst(tmp_path, monkeypatch, [
        ["gene_id", "ProteinCoding"],
        ["ENSG1.14", "1"],
        ["ENSG1.14_PAR_Y", "1"],
    ])
    _ = inst.gene_matrix
    captured = capsys.readouterr()
    assert "duplicated" not in (captured.out + captured.err)


def test_gene_matrix_warns_on_conflicting_duplicates(tmp_path, monkeypatch, capsys):
    """Duplicated rows that disagree drop a gene list, which must warn."""
    inst = _gene_matrix_inst(tmp_path, monkeypatch, [
        ["gene_id", "ProteinCoding", "lincRNA"],
        ["ENSG1.14", "1", "1"],
        ["ENSG1.15", "1", "0"],
    ])
    gene_matrix = inst.gene_matrix
    captured = capsys.readouterr()
    assert "duplicated" in (captured.out + captured.err)
    assert gene_matrix["gene_id"].tolist() == ["ENSG1"]
    assert gene_matrix["lincRNA"].tolist() == [0]
