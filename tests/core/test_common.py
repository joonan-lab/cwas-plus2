"""
Test the methods in cwas.core.common
"""
import cwas.core.common as common
import numpy as np
import pytest


def test_cmp_two_arr():
    # Empty arrays
    arr1 = np.array([])
    arr2 = np.array([])
    assert common.cmp_two_arr(arr1, arr2)

    # Different lengths
    arr1 = np.array([1, 2, 3, 4, 5])
    arr2 = np.array([3, 4, 5])
    assert not common.cmp_two_arr(arr1, arr2)

    # Same arrays
    arr2 = np.array([1, 2, 3, 4, 5])
    assert common.cmp_two_arr(arr1, arr2)

    # Arrays with the same items but not in the same order
    np.random.shuffle(arr2)
    assert common.cmp_two_arr(arr1, arr2)

    # False if the items of each list are not the same.
    arr2 = np.array([1, 14, 3, 6, 5])
    assert not common.cmp_two_arr(arr1, arr2)


def test_div_dist_num():
    # Raise ValueError if one of any argument is not a positive integer.
    with pytest.raises(ValueError):
        _ = common.div_dist_num(-1, 5)

    with pytest.raises(ValueError):
        _ = common.div_dist_num(5, -1)

    # Raise ValueError if 'num' is less than 'num_group'.
    with pytest.raises(ValueError):
        _ = common.div_dist_num(4, 5)

    num = 5
    num_group = 5
    expected = [1, 1, 1, 1, 1]
    assert expected == common.div_dist_num(num, num_group)

    for i in range(5):
        num += 1
        expected[i] += 1
        assert expected == common.div_dist_num(num, num_group)


def test_chunk_list():
    # ValueError will be raised if the input list is empty.
    with pytest.raises(ValueError):
        _ = common.chunk_list([], 5)

    _list = [1, 2, 3, 4, 5]

    # ValueError will be raised
    # if the number of lists is not a positive integer.
    with pytest.raises(ValueError):
        _ = common.chunk_list(_list, -1)

    with pytest.raises(ValueError):
        _ = common.chunk_list(_list, 0)

    result = common.chunk_list(_list, 1)
    expected = [[1, 2, 3, 4, 5]]
    assert result == expected

    result = common.chunk_list(_list, 2)
    expected = [[1, 2, 3], [4, 5]]
    assert result == expected

    result = common.chunk_list(_list, 3)
    expected = [[1, 2], [3, 4], [5]]
    assert result == expected

    result = common.chunk_list(_list, 4)
    expected = [[1, 2], [3], [4], [5]]
    assert result == expected

    result = common.chunk_list(_list, 5)
    expected = [[1], [2], [3], [4], [5]]
    assert result == expected

    # ValueError is raised if the number of lists exceeds the length of the
    # input list.
    with pytest.raises(ValueError):
        _ = common.chunk_list(_list, 6)


def test_swap_label():
    # Make an input for testing
    label_group_pairs = []

    for n in range(1, 6):
        group_id = f"g{n}"
        label_group_pairs.append((f"{group_id}_l1", group_id))
        label_group_pairs.append((f"{group_id}_l2", group_id))

    np.random.shuffle(label_group_pairs)

    labels = np.array([pair[0] for pair in label_group_pairs])
    group_ids = np.array([pair[1] for pair in label_group_pairs])

    # Basic testing with re-shuffling if swap results are the same as original
    for _ in range(10):  # Try up to 10 times to shuffle differently
        swap_labels = common.swap_label(labels, group_ids)
        if np.any(labels != swap_labels):
            break
    else:
        assert False, "Swapping failed to change any labels after multiple attempts."

    # Basic testing
    # swap_labels = common.swap_label(labels, group_ids)
    assert len(swap_labels) == len(group_ids)
    assert np.any(labels != swap_labels)  # Swapping is succeeded.

    # Check if the labels are swapped within their groups.
    for swap_label, group_id in zip(swap_labels, group_ids):
        assert swap_label.startswith(group_id)

    # Test the case where more than 2 labels are in the same group.
    group_id = f"g{np.random.randint(1, 6)}"
    new_label = f"{group_id}_l3"
    labels = np.append(labels, new_label)
    group_ids = np.append(group_ids, group_id)
    np.random.shuffle(labels)
    np.random.shuffle(group_ids)

    with pytest.raises(AssertionError) as e_info:
        assert group_id in str(e_info.value)


def test_int_to_bit_arr():
    assert common.int_to_bit_arr(5, 0).size == 0
    assert common.int_to_bit_arr(5, 3).size == 3
    assert common.int_to_bit_arr(5, 5).size == 5
    assert (common.int_to_bit_arr(1, 3) == np.array([1, 0, 0])).all()
    assert (common.int_to_bit_arr(2, 3) == np.array([0, 1, 0])).all()
    assert (common.int_to_bit_arr(3, 3) == np.array([1, 1, 0])).all()
    assert (common.int_to_bit_arr(4, 3) == np.array([0, 0, 1])).all()
    assert (common.int_to_bit_arr(5, 3) == np.array([1, 0, 1])).all()
    assert (common.int_to_bit_arr(6, 3) == np.array([0, 1, 1])).all()
    assert (common.int_to_bit_arr(7, 3) == np.array([1, 1, 1])).all()


def test_int_to_bit_arr_invalid_args():
    # A negative integer is not allowed as an argument.
    with pytest.raises(ValueError):
        _ = common.int_to_bit_arr(-1, 5)

    with pytest.raises(ValueError):
        _ = common.int_to_bit_arr(5, -1)


# --- resolve_tied_gene_id ---

# VEP reports every gene tied for nearest, joined by '&', in genomic
# coordinate order. That order is not a ranking, so the gene matrix decides.
_TIE_MATRIX = {
    "ENSG_PC": {"ProteinCoding", "ASDTADAFDR03"},
    "ENSG_PC2": {"ProteinCoding"},
    "ENSG_RICH": {"lincRNA", "ASDTADAFDR03", "DDD"},
    "ENSG_POOR": {"lincRNA", "DDD"},
    "ENSG_BARE": set(),
}


def test_resolve_tied_gene_id_without_tie():
    assert common.resolve_tied_gene_id("ENSG_PC", _TIE_MATRIX) == "ENSG_PC"
    # A gene the matrix does not know about is still returned unchanged.
    assert common.resolve_tied_gene_id("ENSG_X", _TIE_MATRIX) == "ENSG_X"


def test_resolve_tied_gene_id_prefers_protein_coding():
    """A protein-coding gene wins even when listed second and in fewer lists."""
    assert (
        common.resolve_tied_gene_id("ENSG_RICH&ENSG_PC2", _TIE_MATRIX)
        == "ENSG_PC2"
    )


def test_resolve_tied_gene_id_prefers_more_gene_lists():
    """Without a protein-coding gene, the gene in more gene lists wins."""
    assert (
        common.resolve_tied_gene_id("ENSG_POOR&ENSG_RICH", _TIE_MATRIX)
        == "ENSG_RICH"
    )


def test_resolve_tied_gene_id_ignores_biotype_lists():
    """ProteinCoding and lincRNA describe the biotype, so they do not count.

    ENSG_PC2 and ENSG_POOR both belong to one non-biotype list... none and
    one respectively, so the biotype lists must not tip the comparison.
    """
    assert common.tied_gene_priority(_TIE_MATRIX["ENSG_RICH"]) == (False, 2)
    assert common.tied_gene_priority(_TIE_MATRIX["ENSG_POOR"]) == (False, 1)
    assert common.tied_gene_priority(_TIE_MATRIX["ENSG_PC2"]) == (True, 0)


def test_resolve_tied_gene_id_prefers_known_gene():
    """A gene missing from the matrix ranks lowest."""
    assert (
        common.resolve_tied_gene_id("ENSG_UNKNOWN&ENSG_POOR", _TIE_MATRIX)
        == "ENSG_POOR"
    )


def test_resolve_tied_gene_id_falls_back_to_the_first_gene():
    """Equally ranked genes resolve to the one VEP listed first."""
    assert (
        common.resolve_tied_gene_id("ENSG_BARE&ENSG_UNKNOWN", _TIE_MATRIX)
        == "ENSG_BARE"
    )
    assert (
        common.resolve_tied_gene_id("ENSG_UNKNOWN&ENSG_BARE", _TIE_MATRIX)
        == "ENSG_UNKNOWN"
    )


def test_normalize_gene_id_of_tied_gene_ids():
    """Every ID of a tie must lose its version, not just the last one."""
    assert (
        common.normalize_gene_id("ENSG1.14&ENSG2.15") == "ENSG1&ENSG2"
    )
    assert common.normalize_gene_id("ENSG1.14") == "ENSG1"


def test_normalize_gene_id_strips_par_y_suffix():
    """GENCODE marks the chrY copy of a pseudoautosomal gene with '_PAR_Y'.

    VEP never emits that suffix. Both copies share one Ensembl stable ID, and
    VEP reports it for a variant on either chromosome, so the chrY row of a
    gene matrix only matches once the suffix is gone.
    """
    assert (
        common.normalize_gene_id("ENSG00000182378.14_PAR_Y")
        == "ENSG00000182378"
    )
    # The version is not always present.
    assert common.normalize_gene_id("ENSG00000182378_PAR_Y") == "ENSG00000182378"
    # The chrX copy of the same gene normalizes to the same key.
    assert common.normalize_gene_id("ENSG00000182378.14") == "ENSG00000182378"


def test_normalize_gene_id_strips_par_y_of_tied_gene_ids():
    assert (
        common.normalize_gene_id("ENSG1.14_PAR_Y&ENSG2.15")
        == "ENSG1&ENSG2"
    )


def test_normalize_gene_id_keeps_other_suffixes():
    """Only the version and '_PAR_Y' are dropped."""
    assert common.normalize_gene_id("ENSG1_PAR_X") == "ENSG1_PAR_X"
    assert common.normalize_gene_id("ENSG1.14_OTHER") == "ENSG1.14_OTHER"
    assert common.normalize_gene_id("ENSG1") == "ENSG1"


@pytest.mark.parametrize(
    "header,expected",
    [
        (["gene_id", "gene_name", "ProteinCoding"], (0, 1)),
        (["ensembl_gene_id", "symbol", "ProteinCoding"], (0, 1)),
        (["ProteinCoding", "GENE_NAME", "Gene_ID"], (2, 1)),
        (["gene_id", "ProteinCoding"], (0, None)),
    ],
)
def test_find_gene_key_columns(header, expected):
    """The key columns are found by name, whatever their case or order."""
    assert common.find_gene_key_columns(header) == expected


def test_find_gene_key_columns_without_gene_id():
    with pytest.raises(ValueError):
        common.find_gene_key_columns(["gene_name", "ProteinCoding"])


@pytest.mark.parametrize(
    "header,kind",
    [
        (["gene_id", "ensembl_gene_id", "ProteinCoding"], "gene ID"),
        (["ensembl_gene_id", "gene_id", "ProteinCoding"], "gene ID"),
        (["gene_id", "gene_name", "symbol", "ProteinCoding"], "gene symbol"),
    ],
)
def test_find_gene_key_columns_rejects_a_repeated_key_column(header, kind):
    """A key column named twice is neither a second key nor a gene list.

    Reading only the first left the second to be read as a gene list. In
    extract_variant the rename of the ID column then produced two columns of
    the same name, and the matrix died on an unrelated-looking AttributeError
    while categorization read the very same file without complaint.
    """
    with pytest.raises(ValueError) as excinfo:
        common.find_gene_key_columns(header)
    assert f"more than one {kind} column" in str(excinfo.value)


@pytest.mark.parametrize(
    "gene_id",
    [
        "ENSG00000187634",
        "ENSG00000187634.15",
        "ENSG00000182378_PAR_Y",
        "ENSG00000182378.14_PAR_Y",
        "ENSMUSG00000000001.2",
    ],
)
def test_check_gene_matrix_ids_accepts_complete_ensembl_ids(gene_id):
    common.check_gene_matrix_ids([gene_id], "gene_matrix.txt")


@pytest.mark.parametrize(
    "gene_id",
    [
        "",
        "SAMD11",
        " ENSG00000187634",
        "ENSG00000187634 ",
        "ENSG00000187634junk",
        "ENSG00000187634,ENSG00000188290",
        "ENSG00000187634&ENSG00000188290",
    ],
)
def test_check_gene_matrix_ids_rejects_invalid_values(gene_id):
    with pytest.raises(ValueError) as excinfo:
        common.check_gene_matrix_ids(
            ["ENSG00000187634", gene_id], "gene_matrix.txt"
        )

    message = str(excinfo.value)
    assert "gene_matrix.txt" in message
    assert "line 3" in message
    assert repr(gene_id) in message
