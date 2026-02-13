import argparse
import multiprocessing as mp
from functools import reduce
from itertools import product
from math import ceil
from pathlib import Path
from cwas.core.common import chunk_list
import re
import zarr
from collections import OrderedDict
import pandas as pd
import numpy as np
import yaml

from cwas.utils.log import print_arg, print_progress
from cwas.core.categorization.categorizer import Categorizer
from cwas.core.categorization.parser import (
    parse_annotated_vcf,
    parse_gene_matrix,
)
from cwas.runnable import Runnable
from cwas.utils.check import check_num_proc, check_is_file, check_is_dir

class Categorization(Runnable):
    def __init__(self, args: argparse.Namespace):
        super().__init__(args)
        self._annotated_vcf = None
        self._gene_matrix = None
        self._category_domain = None
        self._annotated_vcf_groupby_sample = None
        self._redundant_categories = None
        self._sample_ids = None
        self._categories = None
        self._result = None
        self._categorizer = None

    @staticmethod
    def _print_args(args: argparse.Namespace):
        print_arg(
            "No. worker processes for the categorization",
            f"{args.num_proc: ,d}",
        )
        print_arg("Annotated VCF file", args.input_path)

    @staticmethod
    def _check_args_validity(args: argparse.Namespace):
        check_num_proc(args.num_proc)
        check_is_file(args.input_path)
        check_is_dir(args.output_dir_path)

    @property
    def num_proc(self):
        return self.args.num_proc

    @property
    def input_path(self):
        return self.args.input_path.resolve()

    @property
    def output_dir_path(self):
        return self.args.output_dir_path.resolve()

    @property
    def mis_info_key(self) -> str:
        return self.get_env("VEP_MIS_INFO_KEY")

    @property
    def mis_thres(self) -> float:
        return float(self.get_env("VEP_MIS_THRES"))

    @property
    def result_path(self) -> Path:
        f_name = re.sub(r'annotated\.vcf\.gz|annotated\.vcf', 'categorization_result.zarr', self.input_path.name)
        return self.output_dir_path / f_name

    @property
    def annotated_vcf(self) -> pd.DataFrame:
        if self._annotated_vcf is None:
            print_progress("Parse the annotated VCF")
            self._annotated_vcf = parse_annotated_vcf(
                Path(self.input_path)
            )
        return self._annotated_vcf

    @property
    def gene_matrix(self) -> dict:
        if self._gene_matrix is None:
            self._gene_matrix = parse_gene_matrix(
                Path(self.get_env("GENE_MATRIX"))
            )
        return self._gene_matrix

    @property
    def category_domain(self) -> dict:
        if self._category_domain is None:
            with open(self.get_env('CATEGORY_DOMAIN'), 'r') as category_domain_file:
                self._category_domain = yaml.safe_load(category_domain_file)
        return self._category_domain

    @property
    def categorizer(self) -> Categorizer:
        if self._categorizer is None:
            self._categorizer = Categorizer(self.category_domain, self.gene_matrix, self.mis_info_key, self.mis_thres)
        return self._categorizer

    @property
    def annotated_vcf_groupby_sample(self):
        if self._annotated_vcf_groupby_sample is None:
            self._annotated_vcf_groupby_sample = self.annotated_vcf.groupby("SAMPLE")
        return self._annotated_vcf_groupby_sample

    @property 
    def sample_ids(self) -> list:
        if self._sample_ids is None:
            self._sample_ids = list(self.annotated_vcf_groupby_sample.groups)
        return self._sample_ids

    @property
    def annotated_vcf_split_by_sample(self) -> list:
        return [
            self.annotated_vcf_groupby_sample.get_group(sample_id)
            for sample_id in self.sample_ids
        ]

    @property
    def redundant_categories(self) -> set:
        if self._redundant_categories is None:
            redundant_category_table = pd.read_table(
                self.get_env("REDUNDANT_CATEGORY")
            )
            self._redundant_categories = reduce(
                lambda x, y: x.union(y),
                [
                    self._get_redundant_categories_from_row(row)
                    for _, row in redundant_category_table.iterrows()
                ],
                set()
            )
        return self._redundant_categories

    def _get_redundant_categories_from_row(self, row: pd.Series) -> list:
        annotation_term_lists = {}
        for k, v in row.items():
            if v == "*":
                annotation_term_lists[k] = self._category_domain[k]
            else:
                annotation_term_lists[k] = [v]

        return {
            "_".join(combination)
            for combination in product(
                annotation_term_lists["variant_type"],
                annotation_term_lists["gene_set"],
                annotation_term_lists["functional_score"],
                annotation_term_lists["gencode"],
                annotation_term_lists["functional_annotation"],
            )
        }

    def run(self):
        self.categorize_vcf()
        self.save_result()
        self.update_env()
        print_progress("Done")

    def categorize_vcf(self):
        from cwas.core.categorization.categorizer import _USE_RUST
        if _USE_RUST:
            self.categorize_vcf_bulk_rust()
            return
        results_each_sample = (
            self.categorize_vcf_for_each_sample()
            if self.num_proc == 1
            else self.categorize_vcf_for_each_sample_with_mp()
        )
        print_progress("Remove redundant categories from the result")
        all_categories = list(OrderedDict.fromkeys(k for d in results_each_sample for k in d.keys()))
        self._categories = [x for x in all_categories if not x in self.redundant_categories]
        print_progress(f"{len(self._categories):,d} categories have remained.")
        print_progress("Organize the results")
        # Build result matrix efficiently using numpy
        n_samples = len(self.sample_ids)
        n_cats = len(self._categories)
        result = np.zeros((n_samples, n_cats), dtype=np.int32)
        cat_to_idx = {cat: j for j, cat in enumerate(self._categories)}
        for i, d in enumerate(results_each_sample):
            for cat, count in d.items():
                j = cat_to_idx.get(cat)
                if j is not None:
                    result[i, j] = count
        self._result = result

    def categorize_vcf_bulk_rust(self):
        """Bulk Rust-accelerated categorization: prepare once, batch Rust calls."""
        import os

        # Set RAYON_NUM_THREADS before any Rust call (thread pool is lazily
        # initialized and immutable once created)
        os.environ['RAYON_NUM_THREADS'] = str(self.num_proc)

        from cwas_core import categorize_variants as _rust_categorize
        from cwas_core import build_category_names as _rust_build_names

        # Step 1: Prepare annotations once on the full VCF
        print_progress("Prepare annotations for the full VCF")
        annotations, group_sizes, group_terms = self.categorizer._prepare_annotations(
            self.annotated_vcf
        )
        n_variants_total = len(self.annotated_vcf)

        # Step 2: Build category names once
        print_progress("Build category names")
        category_names = _rust_build_names(group_terms)
        n_categories = len(category_names)
        print_progress(f"Total categories: {n_categories:,d}")

        # Step 3: Pre-compute non-redundant category mask
        print_progress("Compute non-redundant category mask")
        redundant = self.redundant_categories
        non_redundant_mask = np.array(
            [name not in redundant for name in category_names], dtype=bool
        )
        non_redundant_indices = np.where(non_redundant_mask)[0]
        non_redundant_names = [category_names[i] for i in non_redundant_indices]
        print_progress(f"Non-redundant categories: {len(non_redundant_names):,d}")

        # Step 4: Map variants to global sample indices
        sample_ids_list = self.sample_ids
        n_samples = len(sample_ids_list)
        sample_to_idx = {sid: i for i, sid in enumerate(sample_ids_list)}
        sample_col = self.annotated_vcf['SAMPLE'].values
        global_sample_indices = np.array(
            [sample_to_idx[s] for s in sample_col], dtype=np.intp
        )

        # Step 5: Compute batch size (target ~512 MB per Rayon thread)
        memory_budget_per_thread = 512 * 1024 * 1024  # 512 MB
        batch_size = max(1, memory_budget_per_thread // (n_categories * 4))
        n_batches = (n_samples + batch_size - 1) // batch_size
        print_progress(
            f"Batch size: {batch_size:,d} samples, {n_batches} batches "
            f"(num_proc={self.num_proc})"
        )

        # Pre-allocate result matrix (only non-redundant categories)
        n_non_redundant = len(non_redundant_indices)
        result = np.zeros((n_samples, n_non_redundant), dtype=np.int32)

        # Step 6: Process in batches
        for batch_idx in range(n_batches):
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, n_samples)
            batch_n = batch_end - batch_start

            print_progress(
                f"Categorizing batch {batch_idx + 1}/{n_batches} "
                f"(samples {batch_start + 1:,d}-{batch_end:,d})"
            )

            # Find variants belonging to this batch's samples
            variant_mask = (
                (global_sample_indices >= batch_start)
                & (global_sample_indices < batch_end)
            )
            variant_indices = np.where(variant_mask)[0]

            if len(variant_indices) == 0:
                continue

            # Remap sample IDs to batch-local (0..batch_n-1)
            batch_sample_ids_signed = (
                global_sample_indices[variant_indices] - batch_start
            )
            assert np.all(batch_sample_ids_signed >= 0), (
                f"Negative batch-local sample index in batch {batch_idx}"
            )
            batch_sample_ids = batch_sample_ids_signed.astype(np.uintp)

            # Slice annotations for this batch's variants
            batch_annotations = [
                [annotations[g][v] for v in variant_indices]
                for g in range(5)
            ]

            # Call Rust once for the entire batch
            count_matrix = _rust_categorize(
                batch_annotations, group_sizes, batch_sample_ids, batch_n
            )

            # Copy non-redundant columns to result
            result[batch_start:batch_end, :] = count_matrix[:, non_redundant_indices]

        # Step 7: Filter out all-zero categories
        print_progress("Filter out all-zero categories")
        nonzero_mask = np.any(result != 0, axis=0)
        self._categories = [
            non_redundant_names[i]
            for i in range(n_non_redundant)
            if nonzero_mask[i]
        ]
        self._result = result[:, nonzero_mask]

        print_progress(f"{len(self._categories):,d} categories have remained.")

    def categorize_vcf_for_each_sample(self):
        result = []
        for i, sample_vcf in enumerate(self.annotated_vcf_split_by_sample, 1):
            if i % 100 == 0:
                print_progress(f"Categorize variants of {i} samples")
            result.append(self.categorizer.categorize_variant(sample_vcf))
        return result

    def categorize_vcf_for_each_sample_with_mp(self):
        sample_vcfs = self.annotated_vcf_split_by_sample
        with mp.Pool(self.num_proc) as pool:
            print_progress("Categorize your input variants")
            return pool.map(
                self.categorizer.categorize_variant,
                sample_vcfs,
                chunksize=ceil(len(sample_vcfs) / self.num_proc),
            )

    def save_result(self):
        print_progress(f"Save the result to the file {self.result_path}")
        root = zarr.open(self.result_path, mode='w')
        root.create_group('metadata')
        root['metadata'].attrs['sample_id'] = self.sample_ids
        root['metadata'].attrs['category'] = self._categories
        root.create_dataset('data', data=self._result, chunks=(1000, 1000), dtype='i4')

    def update_env(self):
        self.set_env("CATEGORIZATION_RESULT", self.result_path)
        self.save_env()
        