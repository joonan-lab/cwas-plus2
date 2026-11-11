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
from concurrent.futures import ProcessPoolExecutor
from collections import deque
from tempfile import TemporaryDirectory

import cwas.utils.log as log
from cwas.core.categorization.categorizer import Categorizer, _build_xtx_chunk
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
        log.print_arg("No. category-pair partitions", args.pair_partitions)
        log.print_arg("Genereate an intersection matrix", (False if args.generate_inter_matrix is None else True))
        log.print_arg("Genereate a correlation matrix", (False if args.generate_corr_matrix is None else True))

    @staticmethod
    def _check_args_validity(args: argparse.Namespace):
        check_num_proc(args.num_proc)
        if args.pair_partitions < 1:
            raise ValueError("pair_partitions must be a positive integer")
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
        from scipy import sparse

        self.filtered_combs = self.category_set.loc[self.category_set['is_'+self._domain]==1]['Category'] if self._domain != 'all' else pd.Series(self.categories)

        if self.generate_corr_matrix == "sample":
            # Sample mode: load zarr categorization result as before
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

            log.print_progress("Get an intersection matrix between categories using the number of samples")

            if self.num_proc == 1:
                intersection_matrix = self.process_columns_single(column_range = range(self.categorization_result.shape[1]), matrix=self.categorization_result)
            else:
                chunks = chunk_list(range(self.categorization_result.shape[1]), self.num_proc)
                result = parmap.map(self.process_columns, chunks, matrix=self.categorization_result, pm_pbar=True, pm_processes=self.num_proc)
                intersection_matrix = pd.concat([pd.concat(chunk_results, axis=1) for chunk_results in result], axis=1)

            diag_sqrt = np.sqrt(np.diag(intersection_matrix))
            log.print_progress("Calculate a correlation matrix")
            self._intersection_matrix = intersection_matrix
            self._correlation_matrix = intersection_matrix / np.outer(diag_sqrt, diag_sqrt)

        elif self.generate_corr_matrix == "variant":
            log.print_progress("Get an intersection matrix between categories using the number of variants")
            self._intersection_matrix = None
            self._correlation_matrix = None

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
        """Compute intersection matrix via sparse X.T @ X with multiprocessing."""
        from scipy import sparse

        split_vcfs = np.array_split(self.annotated_vcf, self.num_proc)
        categorizer = self.categorizer

        # Each worker builds X_i for its chunk and returns X_i.T @ X_i
        args_list = [(chunk, categorizer) for chunk in split_vcfs]
        with mp.Pool(self.num_proc) as pool:
            sparse_chunks = pool.map(_build_xtx_chunk, args_list)

        log.print_progress("Summing sparse chunk results")
        total_sparse = sparse_chunks[0]
        for sp in sparse_chunks[1:]:
            total_sparse = total_sparse + sp

        # Subset from full product space to filtered_combs categories
        categories = self.filtered_combs
        all_names = categorizer.get_all_category_names()
        name_to_idx = {name: i for i, name in enumerate(all_names)}
        keep_indices = [name_to_idx[cat] for cat in categories if cat in name_to_idx]

        if len(keep_indices) == total_sparse.shape[0]:
            # No subsetting needed — all categories match
            return total_sparse

        keep = np.array(keep_indices)
        sub = total_sparse[keep][:, keep].tocsr()
        return sub

    def _extract_domain_from_sparse(self, full_sparse):
        """Extract a domain subset from the full sparse intersection matrix."""
        all_names = self.categorizer.get_all_category_names()
        name_to_idx = {name: i for i, name in enumerate(all_names)}
        keep_indices = [name_to_idx[cat] for cat in self.filtered_combs if cat in name_to_idx]

        if len(keep_indices) == full_sparse.shape[0]:
            return full_sparse

        keep = np.array(keep_indices)
        return full_sparse[keep][:, keep].tocsr()

    @staticmethod
    def _get_sparse_chunk(annotated_vcf: pd.DataFrame, categorizer: Categorizer):
        """Worker function for sparse mp path. Returns scipy.sparse.csr_matrix."""
        return categorizer.get_intersection_as_sparse(annotated_vcf)

    @staticmethod
    def get_intersection_matrix(annotated_vcf: pd.DataFrame, categorizer: Categorizer, categories: pd.Index):
        return categorizer.get_intersection_as_dataframe(annotated_vcf, categories)

    def _get_category_names_for_matrix(self, matrix):
        """Get category names list from a matrix (sparse or DataFrame)."""
        from scipy import sparse
        if sparse.issparse(matrix):
            # For sparse matrices from variant mode, use filtered_combs
            return list(self.filtered_combs)
        else:
            return matrix.columns.tolist()

    def _save_sparse_to_zarr(self, sp_matrix, zarr_path, category_names, dtype, chunk_size=1000):
        """Write a sparse matrix to zarr chunk-by-chunk to avoid OOM."""
        from scipy import sparse
        n = sp_matrix.shape[0]
        root = zarr.open(zarr_path, mode='w')
        root.create_group('metadata')
        root['metadata'].attrs['category'] = category_names
        ds = root.create_dataset('data', shape=(n, n), chunks=(chunk_size, chunk_size), dtype=dtype)

        sp_csr = sp_matrix.tocsr()
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            ds[start:end, :] = sp_csr[start:end].toarray().astype(dtype)

    def save_result(self):
        from scipy import sparse

        if self.generate_corr_matrix == 'variant':
            self._save_variant_pair_results()
            return

        if self.generate_inter_matrix == True:
            log.print_progress("Save the intersection matrix to file")

            if self._domain == 'all':
                domain_int_path = Path(self.intersection_matrix_path)
            else:
                domain_int_path = Path(str(self.intersection_matrix_path).replace('.zarr', f'.{self._domain}.zarr'))

            category_names = self._get_category_names_for_matrix(self._intersection_matrix)

            if sparse.issparse(self._intersection_matrix):
                self._save_sparse_to_zarr(self._intersection_matrix, domain_int_path, category_names, dtype='i4')
            else:
                root = zarr.open(domain_int_path, mode='w')
                root.create_group('metadata')
                root['metadata'].attrs['category'] = category_names
                root.create_dataset('data', data=self._intersection_matrix, chunks=(1000, 1000), dtype='i4')

        log.print_progress("Save the correlation matrix to file")
        if self._domain == 'all':
            domain_corr_path = Path(self.matrix_path)
        else:
            domain_corr_path = Path(str(self.matrix_path).replace('.zarr', f'.{self._domain}.zarr'))

        category_names = self._get_category_names_for_matrix(self._correlation_matrix)

        if sparse.issparse(self._correlation_matrix):
            self._save_sparse_to_zarr(self._correlation_matrix, domain_corr_path, category_names, dtype='float64')
        else:
            root = zarr.open(domain_corr_path, mode='w')
            root.create_group('metadata')
            root['metadata'].attrs['category'] = category_names
            root.create_dataset('data', data=self._correlation_matrix, chunks=(1000, 1000), dtype='float64')

    def _save_variant_pair_results(self):
        names = list(self.filtered_combs)
        n = len(names)
        chunk_size = 1000
        suffix = '' if self._domain == 'all' else f'.{self._domain}'
        corr_path = Path(str(self.matrix_path).replace('.zarr', f'{suffix}.zarr'))
        int_path = Path(str(self.intersection_matrix_path).replace('.zarr', f'{suffix}.zarr'))
        with TemporaryDirectory(prefix='cwas_pairs_', dir=self.output_dir_path) as directory:
            log.print_progress('Build category membership indices in batches')
            counts = self.categorizer.write_category_memberships(self.annotated_vcf, names, directory)
            outputs = []
            for path, dtype in [(corr_path, 'float64')] + ([(int_path, 'int32')] if self.generate_inter_matrix else []):
                root = zarr.open(path, mode='w')
                root.create_group('metadata')
                root['metadata'].attrs['category'] = names
                outputs.append(root.create_dataset('data', shape=(n, n), chunks=(chunk_size, chunk_size), dtype=dtype, fill_value=0))
            corr = outputs[0]
            inter = outputs[1] if self.generate_inter_matrix else None
            writer = _PairChunkWriter(corr, inter, counts, chunk_size)
            for start in range(0, n, chunk_size):
                writer.select(start, start)
                end = min(n, start + chunk_size)
                idx = np.arange(end - start)
                writer.block[idx, idx] = counts[start:end]
                writer.flush()
            total = n * (n - 1) // 2
            partitions = self.args.pair_partitions
            log.print_progress(f'Calculate {total:,} unique non-diagonal pairs in {partitions} partitions with {self.num_proc} workers')
            tasks = iter(_category_pair_tasks(n, chunk_size))
            remaining = None
            executor = None
            try:
                if self.num_proc > 1:
                    executor = ProcessPoolExecutor(max_workers=self.num_proc, initializer=_init_pair_worker, initargs=(directory,))
                else:
                    _init_pair_worker(directory)
                for part in range(partitions):
                    size = total // partitions + (part < total % partitions)
                    def partition_tasks():
                        nonlocal remaining
                        left = size
                        while left:
                            i, start, end = remaining if remaining is not None else next(tasks)
                            stop = min(end, start + left)
                            remaining = (i, stop, end) if stop < end else None
                            left -= stop - start
                            yield i, start, stop
                    results = _bounded_pair_results(executor, partition_tasks(), self.num_proc)
                    for i, start, values in results:
                        writer.write(i, start, values)
                    writer.flush()
                    log.print_progress(f'Category-pair partition {part + 1}/{partitions} complete ({size:,} pairs)')
            finally:
                if executor is not None:
                    executor.shutdown(wait=True)
                _close_pair_worker()


_PAIR_INDICES = None
_PAIR_OFFSETS = None


def _init_pair_worker(directory):
    global _PAIR_INDICES, _PAIR_OFFSETS
    _PAIR_INDICES = np.load(Path(directory) / 'indices.npy', mmap_mode='r')
    _PAIR_OFFSETS = np.load(Path(directory) / 'offsets.npy', mmap_mode='r')


def _close_pair_worker():
    global _PAIR_INDICES, _PAIR_OFFSETS
    _PAIR_INDICES = None
    _PAIR_OFFSETS = None


def _count_pair_batch(task):
    i, start, end = task
    left = _PAIR_INDICES[_PAIR_OFFSETS[i]:_PAIR_OFFSETS[i + 1]]
    values = np.zeros(end - start, dtype=np.int32)
    if len(left):
        for j in range(start, end):
            right = _PAIR_INDICES[_PAIR_OFFSETS[j]:_PAIR_OFFSETS[j + 1]]
            if len(right) and left[0] <= right[-1] and right[0] <= left[-1]:
                values[j - start] = np.intersect1d(left, right, assume_unique=True).size
    return i, start, values


def _category_pair_tasks(n, chunk_size):
    for row in range(0, n, chunk_size):
        for col in range(row, n, chunk_size):
            for i in range(row, min(n, row + chunk_size)):
                start = max(i + 1, col)
                end = min(n, col + chunk_size)
                if start < end:
                    yield i, start, end


def _bounded_pair_results(executor, tasks, workers):
    if executor is None:
        for task in tasks:
            yield _count_pair_batch(task)
        return
    pending = deque()
    tasks = iter(tasks)
    for _ in range(2 * workers):
        task = next(tasks, None)
        if task is None:
            break
        pending.append(executor.submit(_count_pair_batch, task))
    while pending:
        yield pending.popleft().result()
        task = next(tasks, None)
        if task is not None:
            pending.append(executor.submit(_count_pair_batch, task))


class _PairChunkWriter:
    def __init__(self, corr, inter, counts, chunk_size):
        self.corr = corr
        self.inter = inter
        self.counts = counts
        self.chunk_size = chunk_size
        self.key = None
        self.block = None

    def select(self, row, col):
        key = row // self.chunk_size * self.chunk_size, col // self.chunk_size * self.chunk_size
        if key == self.key:
            return
        self.flush()
        self.key = key
        r, c = key
        n = len(self.counts)
        self.rows = slice(r, min(r + self.chunk_size, n))
        self.cols = slice(c, min(c + self.chunk_size, n))
        if self.inter is not None:
            self.block = np.asarray(self.inter[self.rows, self.cols], dtype=np.int32)
        else:
            denom = np.sqrt(self.counts[self.rows].astype(np.float64)[:, None] * self.counts[self.cols].astype(np.float64)[None, :])
            self.block = np.rint(self.corr[self.rows, self.cols] * denom).astype(np.int32)

    def write(self, i, start, values):
        self.select(i, start)
        r, c = self.key
        self.block[i - r, start - c:start - c + len(values)] = values
        if r == c:
            self.block[start - c:start - c + len(values), i - r] = values

    def flush(self):
        if self.key is None:
            return
        r, c = self.key
        if self.inter is not None:
            self.inter[self.rows, self.cols] = self.block
            if r != c:
                self.inter[self.cols, self.rows] = self.block.T
        denom = np.sqrt(self.counts[self.rows].astype(np.float64)[:, None] * self.counts[self.cols].astype(np.float64)[None, :])
        values = np.divide(self.block, denom, out=np.zeros_like(denom), where=denom != 0)
        if r == c:
            np.fill_diagonal(values, 1.0)
        self.corr[self.rows, self.cols] = values
        if r != c:
            self.corr[self.cols, self.rows] = values.T
        self.key = None
        self.block = None
