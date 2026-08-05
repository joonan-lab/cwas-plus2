"""
Common algorithms for CWAS
"""
import re

import numpy as np

# VEP joins every gene tied for nearest with '&' (e.g. ENSG1&ENSG2), so a
# gene field can hold more than one ID.
GENE_ID_SEPARATOR = "&"

# Gene matrices carry versioned Ensembl gene IDs (e.g. ENSG00000276861.1)
# while VEP emits unversioned ones (e.g. ENSG00000276861), so the version
# suffix is stripped from both sides before they are matched.
#
# GENCODE also suffixes the chrY copy of a pseudoautosomal gene with '_PAR_Y'
# (e.g. ENSG00000182378.14_PAR_Y) to keep it distinct from the chrX copy in
# its GTF. VEP never emits that suffix: both copies share one Ensembl stable
# ID, and VEP reports that ID for a variant on either chromosome. The suffix
# is therefore stripped too, which collapses the two rows of a pseudoautosomal
# gene onto one key. That is correct, as they are one gene.
#
# Both suffixes are stripped by a single pattern so that no caller can handle
# one and forget the other. The lookahead applies it to every ID of a tied
# gene field, not just the last one.
_GENE_ID_END = r"(?=" + GENE_ID_SEPARATOR + r"|$)"
GENE_ID_SUFFIX_PATTERN = (
    r"(?:\.\d+)?_PAR_Y" + _GENE_ID_END + r"|\.\d+" + _GENE_ID_END
)
_GENE_ID_SUFFIX_RE = re.compile(GENE_ID_SUFFIX_PATTERN)

# The gene lists that describe a gene's own biotype rather than a property
# that makes it a more informative choice among genes tied for nearest.
GENE_SET_BIOTYPES = frozenset(("ProteinCoding", "lincRNA"))

# Genes are matched by ID because gene symbols are not unique. A symbol
# column is still accepted, but only to label the output for readability.
GENE_ID_COLUMNS = ("gene_id", "ensembl_gene_id")
GENE_NAME_COLUMNS = ("gene_name", "gene_symbol", "symbol")

# VEP reports the nearest gene as an Ensembl gene ID only when it is run with
# '--nearest gene'. Releases before genes were matched by ID ran it with
# '--nearest symbol', so this pattern is used to tell the two apart and reject
# an annotated VCF that predates the change.
ENSEMBL_GENE_ID_PATTERN = r"^ENS[A-Z]*G\d+"

# A gene matrix has one gene per row, unlike VEP's NEAREST field, and may use
# the versioned IDs and pseudoautosomal suffixes found in GENCODE resources.
# Match the whole value so that a symbol, whitespace, a second ID, or trailing
# garbage cannot become a key that silently fails to match VEP output.
GENE_MATRIX_GENE_ID_PATTERN = (
    r"ENS[A-Z]*G\d+(?:\.\d+)?(?:_PAR_Y)?"
)
_GENE_MATRIX_GENE_ID_RE = re.compile(GENE_MATRIX_GENE_ID_PATTERN)

# A gene matrix marks membership in a gene list with '1' and non-membership
# with '0'. The values are read as text, and every loader of a gene matrix
# decides membership through is_gene_list_member() so that none of them can
# drift into a second reading of the same file.
GENE_LIST_MEMBER = "1"
GENE_LIST_VALUES = frozenset(("0", GENE_LIST_MEMBER))


def is_gene_list_member(value) -> bool:
    """ Return True if a gene matrix value marks membership in a gene list. """
    return str(value) == GENE_LIST_MEMBER


def check_gene_list_values(values) -> None:
    """ Raise if a gene matrix holds anything but '0' and '1'.

    A matrix written with '1.0' or 'TRUE' would otherwise be read as putting
    no gene in any gene list at all. Nothing would fail: every variant would
    simply fall into the 'Any' gene set, so the run would finish and quietly
    report the wrong categories. Such a matrix is rejected instead.
    """
    invalid = sorted({str(value) for value in values} - GENE_LIST_VALUES)

    if not invalid:
        return

    examples = ", ".join(repr(value) for value in invalid[:5])
    raise ValueError(
        "A gene matrix must mark every gene list with '0' or '1', but this "
        f"one also holds {examples}. A matrix written with '1.0', 'TRUE' or "
        "a blank instead of '1' would put no gene in any gene list, and the "
        "run would finish with every variant in the 'Any' gene set alone, so "
        "it is rejected rather than silently mis-categorized. Rewrite the "
        "gene list columns with 0 and 1."
    )


def check_gene_matrix_ids(
    gene_ids, gene_matrix_path=None, line_numbers=None
) -> None:
    """Raise if a gene-matrix key is not one complete Ensembl gene ID.

    Gene matrices are user-provided, so a mislabeled symbol column or a typo
    would otherwise load successfully and then fail every lookup silently.
    Version suffixes and GENCODE ``_PAR_Y`` suffixes are valid here because
    normalize_gene_id() removes them only after this validation.
    """
    gene_ids = [str(gene_id) for gene_id in gene_ids]
    if line_numbers is None:
        line_numbers = range(2, len(gene_ids) + 2)

    invalid = [
        (line_number, gene_id)
        for line_number, gene_id in zip(line_numbers, gene_ids)
        if _GENE_MATRIX_GENE_ID_RE.fullmatch(gene_id) is None
    ]
    if not invalid:
        return

    location = (
        f' "{gene_matrix_path}"' if gene_matrix_path is not None else ""
    )
    examples = ", ".join(
        f"line {line_number}: {gene_id!r}"
        for line_number, gene_id in invalid[:5]
    )
    raise ValueError(
        f"The gene matrix{location} contains {len(invalid)} invalid Ensembl "
        f"gene ID value(s) ({examples}). Each row must contain one complete "
        "Ensembl gene ID, such as ENSG00000187634 or ENSG00000187634.15. "
        "Version and _PAR_Y suffixes are accepted, but symbols, blanks, "
        "whitespace, and multiple IDs are not."
    )


def tied_gene_priority(gene_set: set) -> tuple:
    """ Rank a gene among the genes VEP reported as tied for nearest.

    A protein-coding gene wins. Otherwise the gene that belongs to more gene
    lists wins, ignoring the lists that only describe the gene's own biotype.
    """
    return (
        "ProteinCoding" in gene_set,
        len(gene_set - GENE_SET_BIOTYPES),
    )


def resolve_tied_gene_id(gene_id: str, gene_matrix: dict) -> str:
    """ Pick one gene when VEP reports several tied for nearest.

    VEP reports every gene at the minimum distance, joined by '&', and it
    reports them in genomic coordinate order. That order is an artifact of
    how VEP searches, not a ranking, so the gene is chosen by how informative
    the gene matrix says it is. A gene the matrix does not know about ranks
    lowest. Ties fall back to the gene VEP listed first, which keeps the
    choice deterministic.
    """
    candidates = gene_id.split(GENE_ID_SEPARATOR)

    if len(candidates) == 1:
        return gene_id

    # max() keeps the first of equally ranked candidates.
    return max(
        candidates,
        key=lambda candidate: tied_gene_priority(
            gene_matrix.get(candidate, set())
        ),
    )


def normalize_gene_id(gene_id: str) -> str:
    """ Reduce an Ensembl gene ID to the form VEP reports.

    Drops the version suffix and the GENCODE '_PAR_Y' suffix, for every ID of
    a tied gene field.
    """
    return _GENE_ID_SUFFIX_RE.sub("", gene_id)


def find_gene_key_columns(header_cols: list) -> tuple:
    """ Find the gene ID and the gene symbol columns of a gene matrix header.

    Return the index of each column. The index of the symbol column is None
    when the gene matrix has no symbol column. The gene ID column is required
    because it is the key used to match variants against the gene matrix.

    A header that names either key column twice is rejected. Reading only the
    first would leave the second to be read as a gene list, which is what the
    two names of a key column never mean.
    """
    gene_id_idxs = []
    gene_name_idxs = []

    for i, col in enumerate(header_cols):
        col_name = col.strip().lower()

        if col_name in GENE_ID_COLUMNS:
            gene_id_idxs.append(i)
        if col_name in GENE_NAME_COLUMNS:
            gene_name_idxs.append(i)

    if not gene_id_idxs:
        raise ValueError(
            "The gene matrix must have a gene ID column (one of: "
            f"{', '.join(GENE_ID_COLUMNS)}). Found: {header_cols}"
        )

    for idxs, kind, names in (
        (gene_id_idxs, "gene ID", GENE_ID_COLUMNS),
        (gene_name_idxs, "gene symbol", GENE_NAME_COLUMNS),
    ):
        if len(idxs) > 1:
            found = ", ".join(repr(header_cols[i]) for i in idxs)
            raise ValueError(
                f"The gene matrix has more than one {kind} column ({found}). "
                f"The names {', '.join(names)} are all read as the {kind} "
                "column, so only one of them may be present. Keep one and "
                "rename or drop the rest."
            )

    return gene_id_idxs[0], (gene_name_idxs[0] if gene_name_idxs else None)


def cmp_two_arr(array1: np.ndarray, array2: np.ndarray) -> bool:
    """ Return True if two arrays have the same items regardless of the order.
    Otherwise, it returns False.
    """
    if len(array1) != len(array2):
        return False

    array1_item_set = set(array1)

    for item in array2:
        if item not in array1_item_set:
            return False

    return True


def div_dist_num(num: int, num_group: int) -> list:
    """ Divide and distribute the number to each group almost equally. """
    if num <= 0 or num_group <= 0:
        raise ValueError("Only positive integers are accepted as arguments.")

    if num < num_group:
        raise ValueError(
            'The argument "num" must be larger than the argument "num_group".'
        )

    num_per_groups = []
    num_per_group = num // num_group
    remain_num = num % num_group

    for _ in range(num_group):
        if remain_num == 0:
            num_per_groups.append(num_per_group)
        else:
            num_per_groups.append(num_per_group + 1)
            remain_num -= 1

    return num_per_groups


def chunk_list(_list: list, num_chunk: int) -> list:
    """ Split the input list into multiple chunks """
    if len(_list) == 0:
        raise ValueError(
            "This function does not accept an empty list as an argument."
        )

    if num_chunk <= 0:
        raise ValueError("The number of chunks must be a positive integer.")

    if len(_list) < num_chunk:
        raise ValueError(
            "The length of the input list must be larger "
            "than the number of chunks."
        )

    chunks = []
    chunk_lens = div_dist_num(len(_list), num_chunk)

    # Make chunks
    i = 0
    for chunk_len in chunk_lens:
        chunk = _list[i : i + chunk_len]
        chunks.append(chunk)
        i += chunk_len

    return chunks


def swap_label(labels: np.ndarray, group_ids: np.ndarray) -> np.ndarray:
    """ Randomly swap labels (case or control) in each group
    and return a list of swapped labels.

    :param labels: Array of labels
    :param group_ids: Array of group IDs corresponding to each label
    :return: Swapped labels
    """
    # Key: a group ID, Value: No. times the key is referred
    group_to_hit_cnt = {group_id: 0 for group_id in group_ids}
    # Key: A group, Value: The index of a label firstly matched with the group
    group_to_idx = {}
    swap_labels = np.copy(labels)

    # Make an array for random swapping
    num_group = len(group_to_hit_cnt.keys())
    do_swaps = np.random.binomial(1, 0.5, size=num_group)
    group_idx = 0

    for i, label in enumerate(labels):
        group_id = group_ids[i]
        group_hit_cnt = group_to_hit_cnt.get(group_id, 0)
        assert (
            group_hit_cnt == 0 or group_hit_cnt == 1
        ), f'Too many labels (more than 2) in a group "{group_id}".'

        if group_hit_cnt == 0:
            group_to_hit_cnt[group_id] = 1
            group_to_idx[group_id] = i
        else:
            group_to_hit_cnt[group_id] += 1
            prev_idx = group_to_idx[group_id]

            # Swap
            do_swap = do_swaps[group_idx]
            if do_swap:
                swap_labels[i] = labels[prev_idx]
                swap_labels[prev_idx] = label

            group_idx += 1

    return swap_labels


class DomainListMixin:
    @property
    def domain_list(self) -> str:
        if self.args.domain_list == 'all':
            return ['all']
        elif self.args.domain_list == 'run_all':
            all_domains = ['all'] + [col[3:] for col in self.category_set.columns if col.startswith('is_')]
            return all_domains
        else:
            if 'all' in self.args.domain_list:
                all_domains = [col[3:] for col in self.category_set.columns if col.startswith('is_')]
                matching_values = ['all'] + [self._check_domain_list(str.lower(d.strip()), all_domains) for d in self.args.domain_list.split(',')]
                return matching_values
            else:
                all_domains = [col[3:] for col in self.category_set.columns if col.startswith('is_')]
                matching_values = [self._check_domain_list(str.lower(d.strip()), all_domains) for d in self.args.domain_list.split(',')]
                return matching_values

    def _check_domain_list(self, d, all_domain_list):
        if not d in map(str.lower, all_domain_list):
            raise ValueError(
                "Invalid domain name: "
                "{}".format(d)
            )
        else:
            idx = list(map(str.lower, all_domain_list)).index(d)
            return all_domain_list[idx]


def int_to_bit_arr(n0: float, bit_arr_len: int) -> np.ndarray:
    n = int(n0)
    if n < 0 or bit_arr_len < 0:
        raise ValueError("A negative integer argument does not allowed.")

    bit_arr = np.zeros(bit_arr_len)

    for i in range(bit_arr_len):
        bit = n % 2
        bit_arr[i] += bit
        n >>= 1

        if n == 0:
            break

    return bit_arr
