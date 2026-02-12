import argparse
import multiprocessing as mp
from pathlib import Path
import parmap
from cwas.core.common import chunk_list
from tqdm import tqdm
import re
import zarr
import yaml

import pandas as pd
import numpy as np
from functools import partial

import cwas.utils.log as log
from cwas.core.categorization.categorizer import Categorizer
from cwas.core.common import DomainListMixin
from cwas.runnable import Runnable
from cwas.utils.check import check_num_proc, check_is_file, check_is_dir

from cwas.core.categorization.parser import (
    parse_annotated_vcf,
    parse_gene_matrix,
)

class Correlation(DomainListMixin, Runnable):
    def __init__(self, args: argparse.Namespace):
        super().__init__(args)
        self._annotated_vcf = None
        self._gene_matrix = None
        self._category_domain = None
        self._categorization_result = None
        self._categorization_root = None
        self._correlation_matrix = None
        self._intersection_matrix = None
        self._sample_ids = None
        self._categories = None
        self._adj_factor = None
        self._category_set_path = None
        self._category_set = None
        self._categorizer = None

    @staticmethod
    def _print_args(args: argparse.Namespace):
        if args.generate_corr_matrix == 'variant':
            log.print_arg("Annotated VCF file", args.annot_path)
        log.print_arg("Categorized file", args.cat_path)
        log.print_arg(
            "No. worker processes for the categorization",
            f"{args.num_proc: ,d}",
        )
        log.print_arg("Genereate an intersection matrix", (False if args.generate_inter_matrix is None else True))
        log.print_arg("Genereate a correlation matrix", (False if args.generate_corr_matrix is None else True))

    @staticmethod
    def _check_args_validity(args: argparse.Namespace):
        check_num_proc(args.num_proc)
        if args.generate_corr_matrix == 'variant':
            check_is_file(args.annot_path)
        check_is_dir(args.cat_path)
        check_is_dir(args.output_dir_path)

    @property
    def annot_path(self):
        return self.args.annot_path.resolve()

    @property
    def annotated_vcf(self) -> pd.DataFrame:
        if self._annotated_vcf is None:
            log.print_progress("Parse the annotated VCF")
            self._annotated_vcf = parse_annotated_vcf(
                Path(self.annot_path)
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
            with Path(self.get_env("CATEGORY_DOMAIN")).open(
                "r"
            ) as category_domain_file:
                self._category_domain = yaml.safe_load(category_domain_file)
        return self._category_domain

    @property
    def categorizer(self) -> Categorizer:
        if self._categorizer is None:
            self._categorizer = Categorizer(self.category_domain, self.gene_matrix, self.mis_info_key, self.mis_thres)
        return self._categorizer

    @property
    def mis_info_key(self) -> str:
        return self.get_env("VEP_MIS_INFO_KEY")

    @property
    def mis_thres(self) -> float:
        return float(self.get_env("VEP_MIS_THRES"))

    @property
    def num_proc(self):
        return self.args.num_proc

    @property
    def cat_path(self) -> Path:
        return self.args.cat_path.resolve()

    @property
    def category_info_path(self) -> Path:
        return self.args.category_info_path.resolve()

    @property
    def categorization_root(self):
        if self._categorization_root is None:
            self._categorization_root = zarr.open(self.cat_path, mode='r')
        return self._categorization_root

    @property
    def sample_ids(self):
        if self._sample_ids is None:
            self._sample_ids = self.categorization_root['metadata'].attrs['sample_id']
        return self._sample_ids

    @property
    def categories(self):
        if self._categories is None:
            self._categories = self.categorization_root['metadata'].attrs['category']
        return self._categories

    @property
    def category_set(self) -> pd.DataFrame:
        if self._category_set is None:
            self._category_set = pd.read_csv(self.category_info_path, sep='\t')
            self._category_set = self._category_set.loc[self._category_set["Category"].isin(self.categories)]
            self._category_set['Category'] = pd.Categorical(self._category_set['Category'],
                                                            categories=self.categories,
                                                            ordered=True)
            self._category_set.sort_values('Category', ignore_index=True, inplace=True)
        return self._category_set

    @property
    def output_dir_path(self):
        return self.args.output_dir_path.resolve()

    @property
    def generate_corr_matrix(self):
        return self.args.generate_corr_matrix

    @property
    def generate_inter_matrix(self):
        return self.args.generate_inter_matrix

    @property
    def matrix_path(self) -> Path:
        f_name = re.sub(r'categorization_result\.zarr\.gz|categorization_result\.zarr', 'correlation_matrix.zarr', self.cat_path.name)
        return self.output_dir_path / f_name

    @property
    def intersection_matrix_path(self) -> Path:
        f_name = re.sub(r'categorization_result\.zarr\.gz|categorization_result\.zarr', 'intersection_matrix.zarr', self.cat_path.name)
        return self.output_dir_path / f_name

    def run(self):
        for i in self.domain_list:
            self._domain = i
            log.print_progress(f"Generate correlation matrix for domain: {i}")
            self.generate_correlation_matrix()
            self.save_result()
        log.print_progress("Done")

    def generate_correlation_matrix(self):
        self.filtered_combs = self.category_set.loc[self.category_set['is_'+self._domain]==1]['Category'] if self._domain != 'all' else pd.Series(self.categories)
        if self._domain != 'all':
            column_indices = [self.categories.index(col) for col in self.filtered_combs]
            self.categorization_result = pd.DataFrame(self.categorization_root['data'][:, column_indices].astype(np.float64),
                                                      index=self.sample_ids,
                                                      columns=self.filtered_combs)
            self.categorization_result.index.name = 'SAMPLE'
        else:
            self.categorization_result = pd.DataFrame(self.categorization_root['data'].astype(np.float64),
                                                      index=self.sample_ids,
                                                      columns=self.categories)
            self.categorization_result.index.name = 'SAMPLE'
        if self.generate_corr_matrix == "sample":
            log.print_progress("Get an intersection matrix between categories using the number of samples")

            if self.num_proc == 1:
                intersection_matrix = self.process_columns_single(column_range = range(self.categorization_result.shape[1]), matrix=self.categorization_result)
            else:
                # Split the column range into evenly sized chunks based on the number of workers
                chunks = chunk_list(range(self.categorization_result.shape[1]), self.num_proc)
                result = parmap.map(self.process_columns, chunks, matrix=self.categorization_result, pm_pbar=True, pm_processes=self.num_proc)
                # Concatenate the count values
                intersection_matrix = pd.concat([pd.concat(chunk_results, axis=1) for chunk_results in result], axis=1)

        elif self.generate_corr_matrix == "variant":
            log.print_progress("Get an intersection matrix between categories using the number of variants")
            intersection_matrix = (
                self.get_intersection_matrix(self.annotated_vcf, self.categorizer, self.categorization_result.columns)
                if self.num_proc == 1
                else self.get_intersection_matrix_with_mp()
            )
        
        diag_sqrt = np.sqrt(np.diag(intersection_matrix))
        log.print_progress("Calculate a correlation matrix")
        self._intersection_matrix = intersection_matrix
        self._correlation_matrix = intersection_matrix/np.outer(diag_sqrt, diag_sqrt)

    @staticmethod
    def process_columns(column_range, matrix: pd.DataFrame) -> list:
        results = []
    
        # Iterate over the column range
        for i in column_range:
            # Multiply the i-th column with values in the matrix
            df_multiplied = matrix.mul(matrix.iloc[:, i], axis=0)
            
            # Count the number of values greater than 0 in each column
            count_values_gt_zero = (df_multiplied > 0).sum(axis=0)
            
            # Assign the column name to count_values_gt_zero
            count_values_gt_zero.name = matrix.columns[i]
            
            results.append(count_values_gt_zero)
        
        return results

    @staticmethod
    def process_columns_single(column_range, matrix: pd.DataFrame) -> pd.DataFrame:
        results = []

        pbar = tqdm(column_range, desc='Processing')

        for i in pbar:
            df_multiplied = matrix.mul(matrix.iloc[:, i], axis=0)
            count_values_gt_zero = (df_multiplied > 0).sum(axis=0)
            count_values_gt_zero.name = matrix.columns[i]
            results.append(count_values_gt_zero)

        pbar.close()

        return pd.concat(results, axis=1)

    def get_intersection_matrix_with_mp(self):
        from cwas.core.categorization.categorizer import _USE_RUST
        split_vcfs = np.array_split(self.annotated_vcf, self.num_proc)

        if _USE_RUST:
            # Sparse path: each worker returns a scipy.sparse.csr_matrix
            _get_sparse = partial(self._get_sparse_chunk, categorizer=self.categorizer)
            with mp.Pool(self.num_proc) as pool:
                sparse_chunks = pool.map(_get_sparse, split_vcfs)

            # Sum sparse matrices (scipy handles this efficiently)
            total_sparse = sparse_chunks[0]
            for sp in sparse_chunks[1:]:
                total_sparse = total_sparse + sp

            # Convert to dense DataFrame for only the requested categories
            categories = self.categorization_result.columns
            all_names = self.categorizer.get_all_category_names()
            name_to_idx = {name: i for i, name in enumerate(all_names)}
            keep_indices = [name_to_idx[cat] for cat in categories if cat in name_to_idx]
            keep = np.array(keep_indices)
            sub = total_sparse[keep][:, keep]
            matched_names = [all_names[i] for i in keep_indices]
            dense = sub.toarray().astype(np.int32)
            df = pd.DataFrame(dense, index=matched_names, columns=matched_names)
            return df.reindex(index=categories, columns=categories, fill_value=0).astype(int)
        else:
            # Dense fallback path
            _get_intersection_matrix = partial(self.get_intersection_matrix,
                                               categorizer=self.categorizer,
                                               categories=self.categorization_result.columns)
            with mp.Pool(self.num_proc) as pool:
                return sum(pool.map(_get_intersection_matrix, split_vcfs))

    @staticmethod
    def _get_sparse_chunk(annotated_vcf: pd.DataFrame, categorizer: Categorizer):
        """Worker function for sparse mp path. Returns scipy.sparse.csr_matrix."""
        return categorizer.get_intersection_as_sparse(annotated_vcf)

    @staticmethod
    def get_intersection_matrix(annotated_vcf: pd.DataFrame, categorizer: Categorizer, categories: pd.Index):
        return categorizer.get_intersection_as_dataframe(annotated_vcf, categories)

    def save_result(self):
        if self.generate_inter_matrix == True:
            log.print_progress("Save the intersection matrix to file")

            if self._domain == 'all':
                domain_int_path = Path(self.intersection_matrix_path)
            else:
                domain_int_path = Path(str(self.intersection_matrix_path).replace('.zarr', f'.{self._domain}.zarr'))
            root = zarr.open(domain_int_path, mode='w')
            root.create_group('metadata')
            root['metadata'].attrs['category'] = self._intersection_matrix.columns.tolist()
            root.create_dataset('data', data=self._intersection_matrix, chunks=(1000, 1000), dtype='i4')

        log.print_progress("Save the correlation matrix to file")
        if self._domain == 'all':
            domain_corr_path = Path(self.matrix_path)
        else:
            domain_corr_path = Path(str(self.matrix_path).replace('.zarr', f'.{self._domain}.zarr'))

        root = zarr.open(domain_corr_path, mode='w')
        root.create_group('metadata')
        root['metadata'].attrs['category'] = self._correlation_matrix.columns.tolist()
        root.create_dataset('data', data=self._correlation_matrix, chunks=(1000, 1000), dtype='float64')

