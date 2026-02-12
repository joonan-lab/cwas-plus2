"""
Module for the categorization of the variants into CWAS categories and
counting the number of variants in each category.
CWAS category is a combination of annotation terms
from each of annotation groups.

There are currently 5 groups of annotation terms.

--- The groups of the annotation terms for CWAS ---
    1. Variant types (variant_type)
    2. Functional score (functional_score)
    3. Gene lists (gene_set)
    4. GENCODE annotation categories (gencode)
    5. Functional annotation categories (functional_annotation)

"""
from collections import defaultdict
from itertools import product

import numpy as np
import pandas as pd
from tqdm import tqdm
from cwas.core.categorization.utils import extract_sublist_by_int, get_idx_dict

try:
    from cwas_core import categorize_variants as _rust_categorize
    from cwas_core import compute_intersection_matrix as _rust_intersection
    from cwas_core import compute_intersection_matrix_sparse as _rust_intersection_sparse
    from cwas_core import build_category_names as _rust_build_names
    _USE_RUST = True
except ImportError:
    _USE_RUST = False


class Categorizer:
    def __init__(self, category_domain: dict, gene_matrix: dict, mis_info_key: str, mis_thres: float) -> None:
        self._category_domain = category_domain
        self._gene_matrix = gene_matrix
        self._mis_info_key = mis_info_key
        self._mis_thres = mis_thres

    def categorize_variant(self, annotated_vcf: pd.DataFrame):
        if _USE_RUST:
            return self._categorize_variant_rust(annotated_vcf)
        return self._categorize_variant_python(annotated_vcf)

    def _categorize_variant_rust(self, annotated_vcf: pd.DataFrame):
        """Rust-accelerated categorization using bitmask integer operations."""
        annotations, group_sizes, group_terms = self._prepare_annotations(annotated_vcf)
        n_variants = len(annotated_vcf)

        # All variants belong to sample_id=0 (single sample call)
        sample_ids = np.zeros(n_variants, dtype=np.uintp)
        n_samples = 1

        # Call Rust: returns (n_samples, n_categories) array
        count_matrix = _rust_categorize(annotations, group_sizes, sample_ids, n_samples)
        counts = count_matrix[0]  # single sample

        # Build category names and create result dict
        category_names = _rust_build_names(group_terms)

        result = defaultdict(int)
        for idx, count in enumerate(counts):
            if count > 0:
                result[category_names[idx]] = int(count)

        return result

    def _categorize_variant_python(self, annotated_vcf: pd.DataFrame):
        """Original Python categorization (fallback)."""
        result = defaultdict(int)

        for annotation_term_lists in self.annotate_each_variant(annotated_vcf):
            for combination in product(*annotation_term_lists):
                result["_".join(combination)] += 1

        return result

    def _prepare_annotations(self, annotated_vcf: pd.DataFrame):
        """Prepare annotation bitmask arrays for Rust functions.

        Returns:
            annotations: np.ndarray of shape (n_variants, 5) with dtype u64
            group_sizes: np.ndarray of shape (5,) with dtype uintp
            group_terms: list of 5 lists of term strings
        """
        variant_type_ints = self.annotate_variant_type(annotated_vcf)
        gene_set_ints = self.annotate_gene_set(annotated_vcf)
        gencode_ints = self.annotate_gencode(annotated_vcf)
        functional_ints = annotated_vcf['ANNOT'].values

        n_variants = len(annotated_vcf)

        # Build the combined functional_score + functional_annotation term list
        cat_list = [*self._category_domain['functional_score'],
                    *self._category_domain['functional_annotation']]
        cat_list = [item for item in cat_list if item not in ['Any', 'All']]

        # For parse_annotation_int_, functional_score terms get prepended with "All"
        # and functional_annotation terms get prepended with "Any"
        fs_terms = ['All'] + [t for t in cat_list if t in self._category_domain['functional_score']]
        fa_terms = ['Any'] + [t for t in cat_list if t in self._category_domain['functional_annotation']]

        # For Rust, we need bitmask integers per group.
        # Groups: variant_type, gene_set, functional_score, gencode, functional_annotation
        # The parse_annotation_int_ splits the ANNOT bitmask into fs and fa terms.
        # We need to compute separate bitmasks for functional_score and functional_annotation.

        # Pre-compute fs and fa bitmasks from the ANNOT integer
        fs_domain_set = set(self._category_domain['functional_score'])
        fa_domain_set = set(self._category_domain['functional_annotation'])

        fs_bitmasks = np.zeros(n_variants, dtype=np.uint64)
        fa_bitmasks = np.zeros(n_variants, dtype=np.uint64)

        for v_idx in range(n_variants):
            annot_int = int(functional_ints[v_idx])
            labels = extract_sublist_by_int(cat_list, annot_int)

            # functional_score bitmask: bit 0 = "All", then matching terms
            fs_mask = 1  # "All" is always set (bit 0)
            for label in labels:
                if label in fs_domain_set:
                    pos = fs_terms.index(label)
                    fs_mask |= (1 << pos)
            fs_bitmasks[v_idx] = fs_mask

            # functional_annotation bitmask: bit 0 = "Any", then matching terms
            fa_mask = 1  # "Any" is always set (bit 0)
            for label in labels:
                if label in fa_domain_set:
                    pos = fa_terms.index(label)
                    fa_mask |= (1 << pos)
            fa_bitmasks[v_idx] = fa_mask

        # Stack into (n_variants, 5) array
        annotations = np.column_stack([
            variant_type_ints.astype(np.uint64),
            gene_set_ints.astype(np.uint64),
            fs_bitmasks,
            gencode_ints.astype(np.uint64),
            fa_bitmasks,
        ])

        group_sizes = np.array([
            len(self._category_domain['variant_type']),
            len(self._category_domain['gene_set']),
            len(fs_terms),
            len(self._category_domain['gencode']),
            len(fa_terms),
        ], dtype=np.uintp)

        group_terms = [
            self._category_domain['variant_type'],
            self._category_domain['gene_set'],
            fs_terms,
            self._category_domain['gencode'],
            fa_terms,
        ]

        return annotations, group_sizes, group_terms

    def get_intersection(self, annotated_vcf: pd.DataFrame):
        if _USE_RUST:
            return self._get_intersection_rust(annotated_vcf)
        return self._get_intersection_python(annotated_vcf)

    def get_intersection_as_dataframe(self, annotated_vcf: pd.DataFrame, categories: pd.Index) -> pd.DataFrame:
        """Return intersection matrix directly as a DataFrame aligned to categories.

        When Rust is available, uses sparse path for memory efficiency.
        """
        if _USE_RUST:
            return self._get_intersection_sparse_df(annotated_vcf, categories)
        # Python fallback: go through dict path
        raw = self._get_intersection_python(annotated_vcf)
        return pd.DataFrame(raw, index=categories, columns=categories).fillna(0).astype(int)

    def _get_intersection_rust(self, annotated_vcf: pd.DataFrame):
        """Rust-accelerated intersection matrix (returns dict for backward compat)."""
        annotations, group_sizes, group_terms = self._prepare_annotations(annotated_vcf)

        matrix = _rust_intersection(annotations, group_sizes)
        category_names = _rust_build_names(group_terms)

        # Convert to nested defaultdict format matching Python output
        result = defaultdict(lambda: defaultdict(int))
        n = len(category_names)
        for i in range(n):
            row = matrix[i]
            for j in range(n):
                val = int(row[j])
                if val > 0:
                    result[category_names[i]][category_names[j]] = val

        return result

    def _get_intersection_rust_df(self, annotated_vcf: pd.DataFrame, categories: pd.Index) -> pd.DataFrame:
        """Rust-accelerated intersection matrix → DataFrame directly.

        Skips the dict intermediate entirely:
          Rust numpy matrix (all categories) → select rows/cols → DataFrame
        """
        annotations, group_sizes, group_terms = self._prepare_annotations(annotated_vcf)

        # Rust returns full (n_all_cats × n_all_cats) numpy matrix
        full_matrix = _rust_intersection(annotations, group_sizes)
        all_names = _rust_build_names(group_terms)

        # Build index mapping: which positions in all_names match the requested categories
        name_to_idx = {name: i for i, name in enumerate(all_names)}
        keep_indices = [name_to_idx[cat] for cat in categories if cat in name_to_idx]

        if len(keep_indices) == len(all_names):
            # No filtering needed — all categories match
            return pd.DataFrame(
                full_matrix.astype(np.int32),
                index=all_names,
                columns=all_names,
            ).reindex(index=categories, columns=categories, fill_value=0)

        # Slice the matrix to only the requested categories
        keep = np.array(keep_indices)
        sub_matrix = full_matrix[np.ix_(keep, keep)].astype(np.int32)
        matched_names = [all_names[i] for i in keep_indices]

        df = pd.DataFrame(sub_matrix, index=matched_names, columns=matched_names)
        return df.reindex(index=categories, columns=categories, fill_value=0).astype(int)

    def get_all_category_names(self):
        """Build the full list of category names using Rust (or Python fallback).

        Returns:
            list of str: all category names in canonical index order
        """
        if _USE_RUST:
            group_terms = self._build_group_terms()
            return _rust_build_names(group_terms)
        # Python fallback: generate via product
        group_terms = self._build_group_terms()
        return ['_'.join(combo) for combo in product(*group_terms)]

    def _build_group_terms(self):
        """Build the 5 group term lists (same logic as _prepare_annotations)."""
        cat_list = [*self._category_domain['functional_score'],
                    *self._category_domain['functional_annotation']]
        cat_list = [item for item in cat_list if item not in ['Any', 'All']]
        fs_terms = ['All'] + [t for t in cat_list if t in self._category_domain['functional_score']]
        fa_terms = ['Any'] + [t for t in cat_list if t in self._category_domain['functional_annotation']]
        return [
            self._category_domain['variant_type'],
            self._category_domain['gene_set'],
            fs_terms,
            self._category_domain['gencode'],
            fa_terms,
        ]

    def get_intersection_as_sparse(self, annotated_vcf: pd.DataFrame):
        """Compute intersection matrix as scipy.sparse.csr_matrix via Rust sparse COO.

        Returns:
            scipy.sparse.csr_matrix of shape (n_categories, n_categories)
        """
        from scipy import sparse

        annotations, group_sizes, _ = self._prepare_annotations(annotated_vcf)
        coo_dict = _rust_intersection_sparse(annotations, group_sizes)

        row = coo_dict['row']
        col = coo_dict['col']
        data = coo_dict['data']
        shape = coo_dict['shape']

        # Build upper-triangle COO matrix
        upper = sparse.coo_matrix((data, (row, col)), shape=shape)
        # Restore symmetry: full = upper + upper.T - diag
        upper_csr = upper.tocsr()
        diag = sparse.diags(upper_csr.diagonal())
        full = upper_csr + upper_csr.T - diag

        return full.tocsr()

    def _get_intersection_sparse_df(self, annotated_vcf: pd.DataFrame, categories: pd.Index) -> pd.DataFrame:
        """Sparse intersection → subset DataFrame for requested categories."""
        full_sparse = self.get_intersection_as_sparse(annotated_vcf)
        all_names = self.get_all_category_names()

        name_to_idx = {name: i for i, name in enumerate(all_names)}
        keep_indices = [name_to_idx[cat] for cat in categories if cat in name_to_idx]

        if not keep_indices:
            return pd.DataFrame(0, index=categories, columns=categories, dtype=np.int32)

        keep = np.array(keep_indices)
        sub = full_sparse[keep][:, keep]
        matched_names = [all_names[i] for i in keep_indices]

        df = pd.DataFrame(sub.toarray().astype(np.int32), index=matched_names, columns=matched_names)
        return df.reindex(index=categories, columns=categories, fill_value=0).astype(int)

    def _get_intersection_python(self, annotated_vcf: pd.DataFrame):
        """Original Python intersection matrix (fallback)."""
        result = defaultdict(lambda: defaultdict(int))

        for annotation_term_lists in tqdm(self.annotate_each_variant(annotated_vcf), total=len(annotated_vcf)):
            for combination in product(*annotation_term_lists):
                for combination2 in product(*annotation_term_lists):
                    result["_".join(combination)]["_".join(combination2)] += 1

        return result

    def get_intersection_variant_level(self, annotated_vcf, category_combinations):
      # Generate unique category combinations
      #category_combinations = set()
      #for annotation_term_lists in annotate_each_variant(annotated_vcf):
      #    for combination in product(*annotation_term_lists):
      #        category_combinations.add("_".join(combination))

      # Create a matrix with zeros
      num_variants = annotated_vcf.shape[0]
      num_categories = len(category_combinations)
      #matrix = np.zeros((num_variants, num_categories), dtype=int)
      matrix = np.zeros((num_variants, num_categories), dtype='uint64')

      # Create a dictionary to map categories to matrix column indices
      category_to_index = {category: index for index, category in enumerate(category_combinations)}

      # Populate the matrix
      for variant_index, annotation_term_lists in enumerate(self.annotate_each_variant(annotated_vcf)):
          for combination in product(*annotation_term_lists):
              category = "_".join(combination)
              if category in category_to_index:
                  category_index = category_to_index[category]
                  matrix[variant_index, category_index] = 1

      df = pd.DataFrame(matrix, columns=category_combinations)

      return df

    def annotate_each_variant(self, annotated_vcf):
        """Newly annotated each variant using CWAS annotation terms.
        In order to annotate each variant with multiple annotation terms
        from each group efficiently, "Annotation integer" has defined.

        Annotation integer: A bitwise representation of the annotation
        of each variant where each bit means each annotation term

        e.g. If the annotation terms is ['A', 'B', 'C', 'D'] and
        the annotation integer is 0b1011, it means that
        the variant is annotated as 'A', 'B' and 'D'.
        """
        variant_type_annotation_ints = self.annotate_variant_type(annotated_vcf)
        #functional_score_annotation_ints = self.annotate_functional_score(annotated_vcf)
        gene_set_annotation_ints = self.annotate_gene_set(annotated_vcf)
        gencode_annotation_ints = self.annotate_gencode(annotated_vcf)
        #region_annotation_ints = self.annotate_region(annotated_vcf)
        functional_ints = annotated_vcf['ANNOT']

        for (
            variant_type_annotation_int,
            functional_score_annotation_int,
            gene_set_annotation_int,
            gencode_annotation_int,
            region_annotation_int,
        ) in zip(
            variant_type_annotation_ints,
            functional_ints,
            gene_set_annotation_ints,
            gencode_annotation_ints,
            functional_ints,
        ):
            yield (
                self.parse_annotation_int(
                    variant_type_annotation_int, "variant_type",
                ),
                self.parse_annotation_int(
                    gene_set_annotation_int, "gene_set"
                ),
                self.parse_annotation_int_(
                    functional_score_annotation_int, "functional_score",
                ),
                self.parse_annotation_int(gencode_annotation_int, "gencode"),
                self.parse_annotation_int_(region_annotation_int, "functional_annotation"),
            )

    def annotate_variant_type(self, annotated_vcf: pd.DataFrame) -> list:
        variant_type_annotation_idx = get_idx_dict(
            self._category_domain["variant_type"]
        )
        refs = annotated_vcf["REF"].values
        alts = annotated_vcf["ALT"].values

        is_snv_arr = (
            (np.vectorize(len)(refs) == 1) & (np.vectorize(len)(alts) == 1)
        ).astype(np.int32)
        annotation_int_conv = (
            lambda is_snv: 2 ** variant_type_annotation_idx["SNV"]
            if is_snv
            else 2 ** variant_type_annotation_idx["Indel"]
        )
        annotation_ints = np.vectorize(annotation_int_conv)(is_snv_arr)
        annotation_ints += 2 ** variant_type_annotation_idx["All"]

        return annotation_ints

    def annotate_functional_score(self, annotated_vcf: pd.DataFrame) -> list:
        functional_score_annotation_idx = get_idx_dict(
            self._category_domain["functional_score"]
        )
        annotation_ints = np.zeros(len(annotated_vcf.index), dtype=int)

        for score in functional_score_annotation_idx:
            if score == "All":
                continue

            score_vals = annotated_vcf[score].values.astype(np.int32)
            annotation_int_conv_func = (
                lambda x: 2 ** functional_score_annotation_idx[score] * x
            )
            annotation_ints += np.vectorize(annotation_int_conv_func)(
                score_vals
            )

        annotation_ints += 2 ** functional_score_annotation_idx["All"]

        return annotation_ints

    def annotate_gene_set(self, annotated_vcf: pd.DataFrame) -> list:
        gene_set_annotation_idx = get_idx_dict(
            self._category_domain["gene_set"]
        )

        # Remove the key 'lincRNA'
        if 'lincRNA' in gene_set_annotation_idx:
            gene_set_annotation_idx.pop('lincRNA')

        gene_symbols = annotated_vcf["SYMBOL"].values
        gene_nearests = annotated_vcf["NEAREST"].values
        gencodes = annotated_vcf["Consequence"].values  # GENCODE annotations

        annotation_int_list = []
        annotation_int_dict = {}

        for symbol, nearest, gencode in zip(
            gene_symbols, gene_nearests, gencodes
        ):
            gene = (
                nearest
                if "downstream_gene_variant" in gencode
                or "intergenic_variant" in gencode
                else symbol
            )
            annotation_int = annotation_int_dict.get(gene, 0)

            if annotation_int == 0:
                gene_set_ = self._gene_matrix.get(gene, set())

                if gene_set_:
                    for gene_cat in gene_set_annotation_idx:
                        if gene_cat in gene_set_:
                            annotation_int += (
                                2 ** gene_set_annotation_idx[gene_cat]
                            )

            annotation_int_list.append(annotation_int)

        annotation_ints = np.asarray(annotation_int_list)
        annotation_ints += 2 ** gene_set_annotation_idx["Any"]

        return annotation_ints

    def annotate_gencode(self, annotated_vcf: pd.DataFrame) -> list:
        gencode_annotation_idx = get_idx_dict(self._category_domain["gencode"])
        gene_symbols = annotated_vcf["SYMBOL"].values
        gene_nearests = annotated_vcf["NEAREST"].values
        gencodes = annotated_vcf["Consequence"].values
        lofs = annotated_vcf["LoF"].values
        lof_flags = annotated_vcf["LoF_flags"].values
        mis_scores = annotated_vcf["MisDb_" + self._mis_info_key].values

        annotation_int_list = []

        for symbol, nearest, gencode, lof, lof_flag, mis_score in zip(
            gene_symbols, gene_nearests, gencodes, lofs, lof_flags, mis_scores
        ):
            gene = (
                nearest
                if "downstream_gene_variant" in gencode
                or "intergenic_variant" in gencode
                else symbol
            )
            gene_set_ = self._gene_matrix.get(gene, set())
            annotation_int = 0
            is_in_coding = False

            if "ProteinCoding" in gene_set_:
                is_in_coding = True
                annotation_int += 2 ** gencode_annotation_idx["CodingRegion"]

                # Coding region
                if ((
                    "stop_gained" in gencode
                    or "splice_donor" in gencode
                    or "splice_acceptor" in gencode
                )
                and (lof == "HC")
                and ((lof_flag=='SINGLE_EXON') or (lof_flag==""))
                ):
                    annotation_int += 2 ** gencode_annotation_idx["PTVRegion"]
                elif ((
                    "frameshift_variant" in gencode
                )
                and (lof == "HC")
                and ((lof_flag=='SINGLE_EXON') or (lof_flag==""))
                ):
                    annotation_int += 2 ** gencode_annotation_idx["PTVRegion"]
                    annotation_int += (
                        2 ** gencode_annotation_idx["FrameshiftRegion"]
                    )
                elif (
                    "missense_variant" in gencode
                    or "protein_altering_variant" in gencode
                    or "start_lost" in gencode
                    or "stop_lost" in gencode
                ):
                    annotation_int += (
                        2 ** gencode_annotation_idx["MissenseRegion"]
                    )

                    if ((len(mis_score)!=0)
                        and (mis_score!='NA')):
                        if '&' in mis_score:
                            mis_score = max([float(i) for i in mis_score.split('&')])
                        else:
                            mis_score = float(mis_score)
                        if mis_score >= self._mis_thres:
                            annotation_int += (
                                2 ** gencode_annotation_idx["DamagingMissenseRegion"]
                            )

                elif (
                    "inframe_deletion" in gencode
                    or "inframe_insertion" in gencode
                ):
                    annotation_int += (
                        2 ** gencode_annotation_idx["InFrameRegion"]
                    )
                elif "synonymous_variant" in gencode:
                    annotation_int += (
                        2 ** gencode_annotation_idx["SilentRegion"]
                    )
                elif (
                    "stop_retained_variant" not in gencode
                    and "incomplete_terminal_codon_variant" not in gencode
                    and "protein_altering_variant" not in gencode
                    and "coding_sequence_variant" not in gencode
                    and "stop_gained" not in gencode
                    and "splice_donor" not in gencode
                    and "splice_acceptor" not in gencode
                    and "frameshift_variant" not in gencode
                ):
                    # Noncoding
                    annotation_int = 0
                    is_in_coding = False

            if not is_in_coding:
                annotation_int += 2 ** gencode_annotation_idx["NoncodingRegion"]

                if "5_prime_UTR_variant" in gencode:
                    annotation_int += 2 ** gencode_annotation_idx["5PrimeUTRsRegion"]
                elif "3_prime_UTR_variant" in gencode:
                    annotation_int += 2 ** gencode_annotation_idx["3PrimeUTRsRegion"]
                elif "upstream_gene_variant" in gencode:
                    annotation_int += (
                        2 ** gencode_annotation_idx["PromoterRegion"]
                    )
                elif "splice_region_variant" in gencode:
                    annotation_int += (
                        2 ** gencode_annotation_idx["SpliceSiteRegion"]
                    )
                elif "intron_variant" in gencode:
                    annotation_int += (
                        2 ** gencode_annotation_idx["IntronRegion"]
                    )
                elif (
                    "downstream_gene_variant" in gencode
                    or "intergenic_variant" in gencode
                ):
                    annotation_int += (
                        2 ** gencode_annotation_idx["IntergenicRegion"]
                    )
                elif "ProteinCoding" not in gene_set_:
                    if "lincRNA" in gene_set_:
                        annotation_int += (
                            2 ** gencode_annotation_idx["lincRnaRegion"]
                        )
                    else:
                        annotation_int += (
                            2 ** gencode_annotation_idx["OtherTranscriptRegion"]
                        )

            annotation_int_list.append(annotation_int)

        annotation_ints = np.asarray(annotation_int_list)
        annotation_ints += 2 ** gencode_annotation_idx["Any"]

        return annotation_ints

    def annotate_region(self, annotated_vcf: pd.DataFrame) -> list:
        region_annotation_idx = get_idx_dict(self._category_domain["functional_annotation"])
        #annotation_floats = np.zeros(len(annotated_vcf.index), dtype=float)
        annotation_ints = np.zeros(len(annotated_vcf.index), dtype=int)

        for region in region_annotation_idx:
            if region == "Any":
                continue

            region_vals = annotated_vcf[region].values.astype(np.int32)
            annotation_int_conv_func = (
                lambda x: 2 ** int(region_annotation_idx[region]) * int(x)
            )
            annotation_ints = annotation_ints + np.vectorize(annotation_int_conv_func)(region_vals)
            #annotation_floats += np.vectorize(annotation_int_conv_func, otypes=[float])(
            #    region_vals
            #)

        #annotation_ints = np.array([int(x) for x in annotation_floats])
        annotation_ints += 2 ** region_annotation_idx["Any"]

        return annotation_ints

    def parse_annotation_int(
        self, annotation_int: int, annotation_term_type: str
    ) -> list:
        """ Parse the annotation integer and
        choose the appropriate subset from the specific annotation terms.
        """
        return extract_sublist_by_int(
            self._category_domain[annotation_term_type], annotation_int
        )

    def parse_annotation_int_(
        self, annotation_int: int, annotation_term_type: str
    ) -> list:
        """ Parse the annotation integer and
        choose the appropriate subset from the specific annotation terms.
        """
        cat_list = [*self._category_domain['functional_score'], *self._category_domain['functional_annotation']]
        cat_list = [item for item in cat_list if item not in ['Any', 'All']]
        labels = extract_sublist_by_int(
            cat_list, int(annotation_int)
        )
        if annotation_term_type == 'functional_score':
            return ['All'] + [item for item in labels if item in self._category_domain['functional_score']]
        elif annotation_term_type == 'functional_annotation':
            return ['Any'] + [item for item in labels if item in self._category_domain['functional_annotation']]
