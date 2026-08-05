"""
This module includes functions for parsing files used in the CWAS
categorization step. By parsing those files, these functions make the
pandas.DataFrame objects that can be directly used in the categorization
algorithm.
"""
from io import TextIOWrapper
import pathlib
import re
import gzip

import numpy as np
import pandas as pd
from cwas.core.common import (
    ENSEMBL_GENE_ID_PATTERN,
    GENE_ID_SEPARATOR,
    GENE_LIST_MEMBER,
    check_gene_matrix_ids,
    check_gene_list_values,
    find_gene_key_columns,
    int_to_bit_arr,
    normalize_gene_id,
)
from cwas.utils.log import print_err, print_warn

try:
    from cwas_core import parse_vcf as _rust_parse_vcf
    _USE_RUST_VCF = True
except ImportError:
    _USE_RUST_VCF = False


# TODO: Make the code much clearer
def parse_annotated_vcf(vcf_path: pathlib.Path) -> pd.DataFrame:
    """ Parse a Variant Calling File (VCF) that includes Variant Effect
    Predictor (VEP) and CWAS annotation information and make a
    pandas.DataFrame object listing annotated variants.
    """
    if _USE_RUST_VCF:
        result = _parse_annotated_vcf_rust(vcf_path)
    else:
        result = _parse_annotated_vcf_python(vcf_path)

    check_nearest_is_gene_id(result, vcf_path)
    return result


# The values a NEAREST field carries when it names no gene.
_NEAREST_PLACEHOLDERS = ("", "-", "nan", "None")


def check_nearest_is_gene_id(
    annotated_vcf: pd.DataFrame, vcf_path: pathlib.Path
) -> None:
    """ Raise if the NEAREST field holds gene symbols instead of gene IDs.

    An annotated VCF made before genes were matched by ID was produced by VEP
    with '--nearest symbol'. Its symbols cannot match the ID-keyed gene matrix,
    so every intergenic and downstream variant would silently lose its gene set
    annotation. Such a VCF is rejected rather than quietly mis-categorized.
    """
    if "NEAREST" not in annotated_vcf.columns:
        return

    nearest = annotated_vcf["NEAREST"].astype(str)
    nearest = nearest[~nearest.isin(_NEAREST_PLACEHOLDERS)]

    # A NEAREST field holds every gene tied for nearest, joined by '&', so it
    # is split before it is checked. Matching the field whole would only ever
    # test the gene VEP listed first, because the pattern is anchored at the
    # start of the string, and 'ENSG1&SAMD11' would pass while 'SAMD11&ENSG1'
    # was rejected.
    genes = nearest.str.split(GENE_ID_SEPARATOR).explode()
    genes = genes[~genes.isin(_NEAREST_PLACEHOLDERS)]

    if genes.empty:
        return

    is_gene_id = genes.str.match(ENSEMBL_GENE_ID_PATTERN)

    if is_gene_id.all():
        return

    examples = ", ".join(genes[~is_gene_id].unique()[:5])
    raise ValueError(
        f'The NEAREST field of "{vcf_path}" holds gene symbols '
        f"(e.g. {examples}) instead of Ensembl gene IDs. This VCF was "
        "annotated by a CWAS-Plus version that ran VEP with "
        "'--nearest symbol'. Genes are now matched by ID, so this VCF must "
        "be re-annotated with 'cwas annotation' before it is categorized."
    )


def _parse_annotated_vcf_rust(vcf_path: pathlib.Path) -> pd.DataFrame:
    """Rust-accelerated VCF parsing."""
    parsed = _rust_parse_vcf(str(vcf_path))
    columns = list(parsed["columns"])
    data = parsed["data"]
    result = pd.DataFrame(data, columns=columns)
    return result


def _parse_annotated_vcf_python(vcf_path: pathlib.Path) -> pd.DataFrame:
    """Original Python VCF parsing (fallback)."""
    variant_col_names = []
    variant_rows = []  # Item: a list of values of each column
    csq_field_names = []  # CSQ is information from VEP
    annot_field_names = []  # Custom annotation field names

    # Read and parse the input VCF
    with gzip.open(vcf_path, "rb") as vep_vcf_file:
        for line_bytes in vep_vcf_file:
            line = line_bytes.decode("utf-8")
            if line.startswith("#"):  # Comments
                if line.startswith("##INFO=<ID=CSQ"):
                    csq_field_names = _parse_vcf_info_field(line)
                elif line.startswith("##INFO=<ID=ANNOT"):
                    annot_field_names = _parse_annot_field(line)
                elif line.startswith("#CHROM"):
                    variant_col_names = _parse_vcf_header_line(line)
            else:  # Rows of variant information follow the comments.
                assert variant_col_names, "The VCF does not have column names."
                assert csq_field_names, "The VCF does not have CSQ information."
                assert annot_field_names, (
                    "The VCF does not have annotation " "information."
                )
                variant_row = line.rstrip("\n").split("\t")
                variant_rows.append(variant_row)

    result = pd.DataFrame(variant_rows, columns=variant_col_names)
    try:
        info_df = _parse_info_column(
            result["INFO"], csq_field_names
        )
    except KeyError:
        print_err(
            "The VCF does not have INFO column or "
            "the INFO values do not have expected field keys."
        )
        raise

    result.drop(columns="INFO", inplace=True)
    result = pd.concat([result, info_df], axis="columns")

    return result


def _parse_vcf_info_field(line):
    csq_line = line.rstrip('">\n')
    info_format_start_idx = re.search(r"Format: ", csq_line).span()[1]
    csq_field_names = csq_line[info_format_start_idx:].split("|")

    return csq_field_names


def _parse_annot_field(line):
    annot_line = line.rstrip('">\n')
    annot_field_str_idx = re.search(r"Key=", annot_line).span()[1]
    annot_field_names = annot_line[annot_field_str_idx:].split("|")

    return annot_field_names


def _parse_vcf_header_line(line):
    variant_col_names = line[1:].rstrip("\n").split("\t")
    return variant_col_names


def _parse_info_column(
    info_column: pd.Series, csq_field_names: list
) -> pd.DataFrame:
    """ Parse the INFO column and make a pd.DataFrame object """
    info_values = info_column.values
    info_dicts = list(map(_parse_info_str, info_values))
    info_df = pd.DataFrame(info_dicts)
    csq_df = _parse_csq_column(info_df["CSQ"], csq_field_names)
    info_df.drop(columns=["CSQ"], inplace=True)
    info_df = pd.concat([info_df, csq_df], axis="columns")

    return info_df


def _parse_info_str(info_str: str) -> dict:
    """ Parse the string of the INFO field to make a dictionary """
    info_dict = {}
    key_value_pairs = info_str.split(";")

    for key_value_pair in key_value_pairs:
        key, value = key_value_pair.split("=", 1)
        info_dict[key] = value

    return info_dict


def _parse_csq_column(
    csq_column: pd.Series, csq_field_names: list
) -> pd.DataFrame:
    """ Parse the CSQ strings in the CSQ column and make a pd.DataFrame
    object
    """
    csq_values = csq_column.values
    csq_records = list(map(lambda csq_str: csq_str.split("|"), csq_values))
    csq_df = pd.DataFrame(csq_records, columns=csq_field_names)

    return csq_df


def _parse_annot_column(
    annot_column: pd.Series, annot_field_names: list
) -> pd.DataFrame:
    """ Parse the annotation integer in the ANNOT column and make a
    pd.DataFrame object
    """
    annot_ints = np.array([int(x) for x in annot_column])
    annot_field_cnt = len(annot_field_names)
    annot_records = list(
        map(
            lambda annot_int: int_to_bit_arr(annot_int, annot_field_cnt),
            annot_ints,
        )
    )
    annot_df = pd.DataFrame(annot_records, columns=annot_field_names)

    return annot_df


def parse_gene_matrix(gene_matrix_path: pathlib.Path) -> dict:
    """ Parse the gene matrix file and make a dictionary.
    The keys and values of the dictionary are gene IDs
    and a set of type names where the gene is associated,
    respectively.
    """
    with gene_matrix_path.open("r", encoding="utf-8-sig") as gene_matrix_file:
        return _parse_gene_matrix(gene_matrix_file, gene_matrix_path)


def _parse_gene_matrix(
    gene_matrix_file: TextIOWrapper, gene_matrix_path=None
) -> dict:
    result = dict()
    header = gene_matrix_file.readline()
    header_cols = header.rstrip("\n").split("\t")

    gene_id_idx, gene_name_idx = find_gene_key_columns(header_cols)
    gene_type_idxs = [
        i
        for i in range(len(header_cols))
        if i != gene_id_idx and i != gene_name_idx
    ]
    all_gene_types = np.array([header_cols[i] for i in gene_type_idxs])

    rows = [line.rstrip("\n").split("\t") for line in gene_matrix_file]
    check_gene_matrix_ids(
        [cols[gene_id_idx] for cols in rows], gene_matrix_path
    )

    for cols in rows:
        gene_id = normalize_gene_id(cols[gene_id_idx])
        gene_matrix_values = np.array([cols[i] for i in gene_type_idxs])

        check_gene_list_values(gene_matrix_values)
        gene_types = set(all_gene_types[gene_matrix_values == GENE_LIST_MEMBER])

        # Rows collapse onto one ID when a gene matrix carries several versions
        # of a gene, or the two pseudoautosomal copies of one. That is expected
        # and silent. Only rows that disagree on their gene lists are worth a
        # warning, because there the entry that is dropped carried something
        # the kept one does not.
        if gene_id in result and result[gene_id] != gene_types:
            print_warn(
                f'The gene ID "{gene_id}" is duplicated in the gene matrix '
                "with differing gene lists. Only the last entry is kept."
            )

        result[gene_id] = gene_types

    return result
