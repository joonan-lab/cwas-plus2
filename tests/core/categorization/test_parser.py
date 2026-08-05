from io import StringIO
import pathlib

import pytest

from cwas.core.categorization import parser
from pandas import Series


def test_parse_info_field():
    csq_field_names = ["Test1", "Test2", "Test3"]
    info_field = (
        f'##INFO=<ID=CSQ,Number=.,Type=String,Description="'
        f"Consequence annotations from Ensembl VEP. Format: "
        f"{'|'.join(csq_field_names)}\">"
    )
    assert parser._parse_vcf_info_field(info_field) == csq_field_names


def test_parse_annot_field():
    annot_field_names = ["ANNOT1", "ANNOT2", "ANNOT3"]
    annot_field = f"##INFO=<ID=ANNOT,Key={'|'.join(annot_field_names)}>"
    assert parser._parse_annot_field(annot_field) == annot_field_names


def test_parse_vcf_header_line():
    vcf_column_titles = [
        "CHROM",
        "POS",
        "ID",
        "REF",
        "ALT",
        "QUAL",
        "FILTER",
        "INFO",
    ]

    header_line = "\t".join(vcf_column_titles)
    assert parser._parse_vcf_header_line(f"#{header_line}") == vcf_column_titles


def test_parse_info_column():
    info_column = Series(
        [
            "SAMPLE=A;BATCH=1;CSQ=1|2|3;ANNOT=7",
            "SAMPLE=B;BATCH=1;CSQ=2|3|1;ANNOT=5",
            "SAMPLE=C;BATCH=1;CSQ=3|2|1;ANNOT=3",
            "SAMPLE=D;BATCH=1;CSQ=3|1|2;ANNOT=1",
        ]
    )
    csq_field_names = ["CSQ1", "CSQ2", "CSQ3"]
    info_df = parser._parse_info_column(
        info_column, csq_field_names
    )
    assert info_df["CSQ1"].to_list() == ["1", "2", "3", "3"]
    assert info_df["CSQ2"].to_list() == ["2", "3", "2", "1"]
    assert info_df["CSQ3"].to_list() == ["3", "1", "1", "2"]


def test_parse_gene_matrix():
    gene_matrix_lines = [
        "\t".join(["gene_id", "gene_name", "asd1", "asd2", "asd3", "asd4"]),
        "\t".join(["ENSG00000000001", "gene1", "1", "0", "1", "1"]),
        "\t".join(["ENSG00000000002", "gene2", "0", "0", "1", "1"]),
        "\t".join(["ENSG00000000003", "gene3", "0", "1", "1", "0"]),
    ]
    result = parser._parse_gene_matrix(StringIO("\n".join(gene_matrix_lines)))
    assert result["ENSG00000000001"] == {"asd1", "asd3", "asd4"}
    assert result["ENSG00000000002"] == {"asd3", "asd4"}
    assert result["ENSG00000000003"] == {"asd2", "asd3"}
    # The gene symbol is not a key and is not a gene list either.
    assert "gene1" not in result
    assert "gene_name" not in result["ENSG00000000001"]


def test_parse_gene_matrix_keeps_duplicated_gene_symbols():
    """Genes sharing a symbol must all be kept, as their IDs differ."""
    gene_matrix_lines = [
        "\t".join(["gene_id", "gene_name", "asd1", "asd2"]),
        "\t".join(["ENSG00000000001", "Y_RNA", "1", "0"]),
        "\t".join(["ENSG00000000002", "Y_RNA", "0", "1"]),
    ]
    result = parser._parse_gene_matrix(StringIO("\n".join(gene_matrix_lines)))
    assert len(result) == 2
    assert result["ENSG00000000001"] == {"asd1"}
    assert result["ENSG00000000002"] == {"asd2"}


def test_parse_gene_matrix_strips_gene_id_version():
    """Versioned gene IDs must match the unversioned ones VEP reports."""
    gene_matrix_lines = [
        "\t".join(["gene_id", "gene_name", "asd1"]),
        "\t".join(["ENSG00000276861.1", "gene1", "1"]),
    ]
    result = parser._parse_gene_matrix(StringIO("\n".join(gene_matrix_lines)))
    assert result["ENSG00000276861"] == {"asd1"}


def test_parse_gene_matrix_without_gene_symbol_column():
    gene_matrix_lines = [
        "\t".join(["gene_id", "asd1", "asd2"]),
        "\t".join(["ENSG00000000001", "1", "0"]),
    ]
    result = parser._parse_gene_matrix(StringIO("\n".join(gene_matrix_lines)))
    assert result["ENSG00000000001"] == {"asd1"}


def test_parse_gene_matrix_without_gene_id_column():
    gene_matrix_lines = [
        "\t".join(["gene_name", "asd1"]),
        "\t".join(["gene1", "1"]),
    ]
    with pytest.raises(ValueError):
        parser._parse_gene_matrix(StringIO("\n".join(gene_matrix_lines)))


@pytest.mark.parametrize(
    "member,absent", [("1.0", "0.0"), ("TRUE", "FALSE"), ("1", "")]
)
def test_parse_gene_matrix_rejects_values_that_are_not_0_or_1(member, absent):
    """A gene matrix that spells membership some other way must be rejected.

    Read as text, '1.0' and 'TRUE' are not '1', so every gene would end up in
    no gene list at all. Nothing would fail: the run would finish with every
    variant in the 'Any' gene set alone and report the wrong categories.
    """
    gene_matrix_lines = [
        "\t".join(["gene_id", "gene_name", "ProteinCoding", "asd1"]),
        "\t".join(["ENSG1", "gene1", member, absent]),
    ]
    with pytest.raises(ValueError) as excinfo:
        parser._parse_gene_matrix(StringIO("\n".join(gene_matrix_lines)))
    assert "'0' or '1'" in str(excinfo.value)


def _nearest_vcf(nearest_values: list):
    from pandas import DataFrame

    return DataFrame({"NEAREST": nearest_values})


def test_check_nearest_accepts_gene_ids():
    """A VCF annotated with '--nearest gene' must be accepted."""
    parser.check_nearest_is_gene_id(
        _nearest_vcf(["ENSG00000187634", "ENSG00000127561", ""]), "test.vcf"
    )


def test_check_nearest_accepts_tied_gene_ids():
    """VEP reports every gene tied for nearest, joined by '&'."""
    parser.check_nearest_is_gene_id(
        _nearest_vcf(["ENSG00000188290&ENSG00000187608"]), "test.vcf"
    )


@pytest.mark.parametrize(
    "nearest", ["ENSG00000188290&SAMD11", "SAMD11&ENSG00000188290"]
)
def test_check_nearest_rejects_a_symbol_tied_to_a_gene_id(nearest):
    """Every gene of a tied NEAREST field must be checked, not just the first.

    The pattern is anchored at the start of the string, so matching the field
    whole tests only the gene VEP listed first. That let a symbol through
    whenever an ID happened to precede it, and made the verdict depend on the
    order VEP reported the tied genes in.
    """
    with pytest.raises(ValueError) as excinfo:
        parser.check_nearest_is_gene_id(_nearest_vcf([nearest]), "test.vcf")
    assert "SAMD11" in str(excinfo.value)


def test_check_nearest_rejects_gene_symbols():
    """A VCF annotated with the old '--nearest symbol' must be rejected.

    Its symbols cannot match the gene-ID-keyed gene matrix, so accepting it
    would silently drop the gene set of every intergenic variant.
    """
    with pytest.raises(ValueError) as excinfo:
        parser.check_nearest_is_gene_id(
            _nearest_vcf(["ENSG00000187634", "SAMD11"]), "test.vcf"
        )
    assert "SAMD11" in str(excinfo.value)
    assert "re-annotated" in str(excinfo.value)


def test_check_nearest_without_nearest_column():
    """A VCF with no NEAREST field must pass through untouched."""
    from pandas import DataFrame

    parser.check_nearest_is_gene_id(DataFrame({"Gene": ["ENSG1"]}), "test.vcf")


# --- the guard as parse_annotated_vcf actually reaches it ---

TEST_FILE_DIR = pathlib.Path(__file__).parent.parent.parent / "test_file"
NEAREST_GENE_VCF = TEST_FILE_DIR / "test_annotated.vcf.gz"
NEAREST_SYMBOL_VCF = TEST_FILE_DIR / "test_annotated_nearest_symbol.vcf.gz"


def test_parse_annotated_vcf_accepts_a_nearest_gene_file():
    """The guard must let a VCF annotated with '--nearest gene' through.

    Calling the helper directly leaves the wiring untested: a guard that was
    never called would pass those tests just as well. This goes through
    parse_annotated_vcf, which is what categorization calls.
    """
    annotated_vcf = parser.parse_annotated_vcf(NEAREST_GENE_VCF)

    assert len(annotated_vcf) > 0
    assert annotated_vcf["NEAREST"].str.match(r"ENSG\d+").all()


def test_parse_annotated_vcf_rejects_a_nearest_symbol_file():
    """A VCF that predates gene IDs must be rejected by parse_annotated_vcf.

    Its symbols cannot match the ID-keyed gene matrix, so every intergenic and
    downstream variant would silently lose its gene set. Rejecting it inside
    the helper is worth nothing unless the helper is reached.
    """
    with pytest.raises(ValueError) as excinfo:
        parser.parse_annotated_vcf(NEAREST_SYMBOL_VCF)

    assert "re-annotated" in str(excinfo.value)


def test_parse_gene_matrix_collapses_par_y_rows():
    """The two pseudoautosomal copies of a gene are one gene to VEP."""
    gene_matrix_lines = [
        "\t".join(["gene_id", "gene_name", "asd1", "asd2"]),
        "\t".join(["ENSG00000182378.14", "PLCXD1", "1", "0"]),
        "\t".join(["ENSG00000182378.14_PAR_Y", "PLCXD1", "1", "0"]),
    ]
    result = parser._parse_gene_matrix(StringIO("\n".join(gene_matrix_lines)))
    assert list(result) == ["ENSG00000182378"]
    assert result["ENSG00000182378"] == {"asd1"}


def test_parse_gene_matrix_par_y_collapse_is_silent(capsys):
    """Collapsing identical rows is expected, so it must not warn."""
    gene_matrix_lines = [
        "\t".join(["gene_id", "asd1"]),
        "\t".join(["ENSG1.14", "1"]),
        "\t".join(["ENSG1.14_PAR_Y", "1"]),
    ]
    parser._parse_gene_matrix(StringIO("\n".join(gene_matrix_lines)))
    captured = capsys.readouterr()
    assert "duplicated" not in (captured.out + captured.err)


def test_parse_gene_matrix_warns_on_conflicting_duplicates(capsys):
    """Duplicated rows that disagree drop a gene list, which must warn."""
    gene_matrix_lines = [
        "\t".join(["gene_id", "asd1", "asd2"]),
        "\t".join(["ENSG1.14", "1", "1"]),
        "\t".join(["ENSG1.15", "1", "0"]),
    ]
    result = parser._parse_gene_matrix(StringIO("\n".join(gene_matrix_lines)))
    captured = capsys.readouterr()
    assert "duplicated" in (captured.out + captured.err)
    assert result["ENSG1"] == {"asd1"}


def test_parse_gene_matrix_rejects_an_invalid_gene_id(tmp_path):
    gene_matrix = tmp_path / "gene_matrix.txt"
    gene_matrix.write_text("gene_id\tasd1\nSAMD11\t1\n")

    with pytest.raises(ValueError) as excinfo:
        parser.parse_gene_matrix(gene_matrix)

    message = str(excinfo.value)
    assert str(gene_matrix) in message
    assert "line 2: 'SAMD11'" in message


def test_parse_gene_matrix_accepts_a_utf8_bom(tmp_path):
    gene_matrix = tmp_path / "gene_matrix.txt"
    gene_matrix.write_text(
        "gene_id\tgene_name\tProteinCoding\n"
        "ENSG00000187634\tSAMD11\t1\n",
        encoding="utf-8-sig",
    )

    result = parser.parse_gene_matrix(gene_matrix)

    assert result == {"ENSG00000187634": {"ProteinCoding"}}
