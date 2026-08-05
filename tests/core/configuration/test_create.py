"""
Test cwas.core.configuration.create
"""
import cwas.core.configuration.create as create
import pytest
import yaml
from pathlib import Path

@pytest.fixture(scope="module", autouse=True)
def setup(cwas_workspace):
    cwas_workspace.mkdir()

@pytest.fixture(scope="module", autouse=True)
def teardown(cwas_workspace):
    yield
    remove_workspace(cwas_workspace)

def remove_workspace(cwas_workspace):
    for f in cwas_workspace.glob("*"):
        f.unlink()
    cwas_workspace.rmdir()

def create_annotation_keys_file(cwas_workspace):
    annotation_keys = {
        'functional_score': {
            'bed_annot1.bed.gz': 'bed_annot1',
            'bed_annot4.bed.gz': 'bed_annot4'
        },
        'functional_annotation': {
            'annot5.bed': 'annot5',
            'annot6.bed': 'annot6'
        }
    }
    bed_key_conf = cwas_workspace / "annotation_keys.yaml"
    with bed_key_conf.open('w') as f:
        yaml.safe_dump(annotation_keys, f)

def test_create_annotation_key_bed(cwas_workspace, annotation_dir):
    bed_key_conf = cwas_workspace / "annotation_keys.yaml"
    create.create_annotation_key(bed_key_conf, annotation_dir, "bed")

    assert bed_key_conf.exists()
    with bed_key_conf.open() as f:
        bed_key = yaml.safe_load(f)
    assert "bed_annot1.bed.gz" in bed_key
    assert "bed.annot2.bed" not in bed_key
    assert "bed.annot3.bed" not in bed_key
    assert "bed_annot4.bed.gz" in bed_key
    assert "bed.annot7.bed" not in bed_key
    assert bed_key["bed_annot1.bed.gz"] == "bed_annot1"
    assert bed_key["bed_annot4.bed.gz"] == "bed_annot4"

    bed_key_conf.unlink()

def test_create_category_domain_list(cwas_workspace, annotation_key_conf, gene_matrix):
    create_annotation_keys_file(cwas_workspace)  # Ensure file exists
    bed_key_conf = cwas_workspace / "annotation_keys.yaml"
    domain_list_path = cwas_workspace / "category_domain.yaml"
    create.create_category_domain_list(domain_list_path, bed_key_conf, gene_matrix)

    assert domain_list_path.is_file()

    with domain_list_path.open("r") as domain_list_f:
        domain_dict = yaml.safe_load(domain_list_f)

    # Spelled out rather than re-derived from the same inputs. Deriving the
    # expectation with the rule under test, or comparing only lengths, hides
    # every change that keeps the count.
    assert domain_dict["gene_set"] == [
        "Any", "gene1", "gene2", "gene3", "gene4", "gene5",
    ]
    assert domain_dict["functional_score"] == [
        "All", "bed_annot1", "bed_annot4",
    ]
    assert domain_dict["functional_annotation"] == ["Any", "annot5", "annot6"]

    bed_key_conf.unlink()
    domain_list_path.unlink()


def test_create_category_domain_list_finds_key_columns_by_name(
    cwas_workspace, annotation_key_conf, tmp_path
):
    """The key columns are found by name, so their position must not matter.

    A gene matrix whose key columns do not sit first is what tells the rule
    apart from the one it replaced: dropping the first two columns would take
    'ProteinCoding' for a key and leave 'gene_id' as a gene list.
    """
    create_annotation_keys_file(cwas_workspace)
    bed_key_conf = cwas_workspace / "annotation_keys.yaml"
    domain_list_path = cwas_workspace / "category_domain.yaml"
    gene_matrix = tmp_path / "gene_matrix.txt"
    gene_matrix.write_text(
        "\t".join(["ProteinCoding", "gene_name", "asd1", "gene_id", "asd2"])
        + "\n"
    )

    create.create_category_domain_list(
        domain_list_path, bed_key_conf, gene_matrix
    )

    with domain_list_path.open("r") as domain_list_f:
        domain_dict = yaml.safe_load(domain_list_f)

    assert domain_dict["gene_set"] == ["Any", "ProteinCoding", "asd1", "asd2"]

    bed_key_conf.unlink()
    domain_list_path.unlink()


def test_create_category_domain_list_rejects_a_repeated_key_column(
    cwas_workspace, annotation_key_conf, tmp_path
):
    """A second ID column would otherwise be counted as a gene list."""
    create_annotation_keys_file(cwas_workspace)
    bed_key_conf = cwas_workspace / "annotation_keys.yaml"
    gene_matrix = tmp_path / "gene_matrix.txt"
    gene_matrix.write_text(
        "\t".join(["gene_id", "ensembl_gene_id", "gene_name", "asd1"]) + "\n"
    )

    with pytest.raises(ValueError) as excinfo:
        create.create_category_domain_list(
            cwas_workspace / "category_domain.yaml", bed_key_conf, gene_matrix
        )
    assert "more than one gene ID column" in str(excinfo.value)

    bed_key_conf.unlink()


def test_create_category_domain_list_rejects_an_invalid_gene_id(
    cwas_workspace, annotation_key_conf, tmp_path
):
    create_annotation_keys_file(cwas_workspace)
    bed_key_conf = cwas_workspace / "annotation_keys.yaml"
    gene_matrix = tmp_path / "gene_matrix.txt"
    gene_matrix.write_text("gene_id\tgene_name\tasd1\nSAMD11\tSAMD11\t1\n")

    with pytest.raises(ValueError) as excinfo:
        create.create_category_domain_list(
            cwas_workspace / "category_domain.yaml", bed_key_conf, gene_matrix
        )

    message = str(excinfo.value)
    assert str(gene_matrix) in message
    assert "line 2: 'SAMD11'" in message

    bed_key_conf.unlink()


def test_create_category_domain_list_accepts_a_utf8_bom(
    cwas_workspace, annotation_key_conf, tmp_path
):
    create_annotation_keys_file(cwas_workspace)
    bed_key_conf = cwas_workspace / "annotation_keys.yaml"
    domain_list_path = cwas_workspace / "category_domain.yaml"
    gene_matrix = tmp_path / "gene_matrix.txt"
    gene_matrix.write_text(
        "gene_id\tgene_name\tProteinCoding\n"
        "ENSG00000187634\tSAMD11\t1\n",
        encoding="utf-8-sig",
    )

    create.create_category_domain_list(
        domain_list_path, bed_key_conf, gene_matrix
    )

    with domain_list_path.open("r") as domain_list_f:
        domains = yaml.safe_load(domain_list_f)
    assert domains["gene_set"] == ["Any", "ProteinCoding"]

    bed_key_conf.unlink()
    domain_list_path.unlink()

def test_create_redundant_category_table(cwas_workspace):
    redundant_category_table = cwas_workspace / "redundant_category.txt"
    create.create_redundant_category_table(redundant_category_table)
    assert redundant_category_table.exists()
    redundant_category_table.unlink()

def test__load_yaml_file_error(cwas_workspace):
    test_path = cwas_workspace / "not_yaml.txt"
    with test_path.open("w") as test_f:
        print('{Hello: World!}"', file=test_f)
    #with pytest.raises(yaml.YAMLError):
    #    _ = create.split_annotation_key(None, None, test_path)
    test_path.unlink()
