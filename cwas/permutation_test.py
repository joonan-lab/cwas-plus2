import argparse
import os
import sys
from pathlib import Path
from typing import Optional
from multiprocessing import Pool
import numpy as np
import pandas as pd
import re
from tqdm import tqdm
import warnings

from cwas.runnable import Runnable

from cwas.burden_test import BurdenTest
from cwas.utils.log import print_progress, print_arg
from cwas.utils.check import check_num_proc
from cwas.core.burden_test.binomial import binom_two_tail_vectorized

_DEVNULL = open(os.devnull, 'w')

# Module-level shared data for multiprocessing workers (inherited via fork, avoids pickling)
_worker_data = {}

class PermutationTest(BurdenTest):
    def __init__(self, args: Optional[argparse.Namespace] = None):
        super().__init__(args)
        self._perm_rrs = None
        self._binom_pvals = None
        self._perm_rrs_path = None
        self._binom_pvals_path = None

    @staticmethod
    def _print_args(args: argparse.Namespace):
        super(PermutationTest, PermutationTest)._print_args(args)
        print_arg(f"Number of permutations", args.num_perm)
        print_arg(f"Number of processes", args.num_proc)
        print_arg(f"Generate binomial p values for burden-shifted data", args.burden_shift)

    @staticmethod
    def _check_args_validity(args: argparse.Namespace):
        super(PermutationTest, PermutationTest)._check_args_validity(args)
        check_num_proc(args.num_proc)
    
    @property
    def cat_path(self) -> Path:
        return self.args.cat_path.resolve()

    @property
    def output_dir_path(self) -> Path:
        return self.args.output_dir_path.resolve()

    @property
    def result_path(self) -> Path:
        f_name = re.sub(r'categorization_result\.zarr\.gz|categorization_result\.zarr', 'permutation_test.txt.gz', self.cat_path.name)
        self._result_path = self.output_dir_path / f_name
        return self._result_path
    
    @property
    def perm_rrs_path(self) -> Path:
        if self._perm_rrs_path is None:
            f_name = re.sub(r'categorization_result\.zarr\.gz|categorization_result\.zarr', 'permutation_RRs.txt.gz', self.cat_path.name)
            self._perm_rrs_path = self.output_dir_path / f_name
        return self._perm_rrs_path
    
    @property
    def binom_pvals_path(self) -> Path:
        if self._binom_pvals_path is None:
            f_name = re.sub(r'categorization_result\.zarr\.gz|categorization_result\.zarr', 'binom_pvals.parquet', self.cat_path.name)
            self._binom_pvals_path = self.output_dir_path / f_name
        return self._binom_pvals_path
    
    @property
    def burden_shift(self) -> bool:
        return self.args.burden_shift

    @property
    def use_n_carrier(self) -> bool:
        return self.args.use_n_carrier

    def run_burden_test(self):
        print_progress("Run permutation test")

        vals = np.concatenate(
            self.cal_perm_rr(self.categorization_result,
                             self.args.num_perm,
                             burden_shift=self.burden_shift)
        )
        
        if self.burden_shift:
            perm_rrs = vals[range(0, len(vals), 2)]
            binom_pvals = vals[range(1, len(vals), 2)]
        else: 
            perm_rrs = vals
            
        self._result["P"] = self.get_perm_pval(
            perm_rrs,
            rr = self._result["Relative_Risk"].values
        )

        low_P_idx = self._result[self._result["P"] < 0.01].index
        print_progress(f"Run additional permutation test for {len(low_P_idx)} categories with P < 0.01")
        perm_rrs_x10 = np.concatenate(
            self.cal_perm_rr(
                self.categorization_result[low_P_idx],
                10*self.args.num_perm,
                burden_shift=False,
            )
        )

        self._result.loc[low_P_idx, "P"] = self.get_perm_pval(
            perm_rrs_x10,
            rr = self._result.loc[low_P_idx]["Relative_Risk"].values
        )
        ## Make a dataframe of binomial p values
        if self.burden_shift:
            self._binom_pvals = pd.DataFrame(binom_pvals, columns=self.categorization_result.columns)
            self._binom_pvals.index += 1
            self._binom_pvals.index.name = 'Trial'

    def cal_perm_rr(self, categorization_result: pd.DataFrame, num_perm: int, burden_shift: bool) -> np.ndarray:
        print_progress(f"Calculate permutation RRs (# of permutations: {num_perm})")

        var_counts = categorization_result[np.isin(self.phenotypes, ['case', 'ctrl'])].values

        # Store shared data in module-level dict (inherited via fork, avoids pickling)
        _worker_data['var_counts'] = var_counts
        _worker_data['case_cnt'] = self.case_cnt
        _worker_data['ctrl_cnt'] = self.ctrl_cnt
        _worker_data['use_n_carrier'] = self.use_n_carrier
        _worker_data['burden_shift'] = burden_shift

        if self.args.num_proc == 1:
            array_list = self._burden_test((0, num_perm))
        else:
            seed_range = []
            range_len = num_perm // self.args.num_proc
            if range_len == 0:
                raise ValueError(f'The number of processors ("{self.args.num_proc:,d}") are larger than '
                                f'the number of permutations ("{num_perm:,d}").')

            for i in range(self.args.num_proc - 1):
                r = (range_len * i, range_len * (i + 1))
                seed_range.append(r)
            seed_range.append((range_len * (self.args.num_proc - 1), num_perm))

            def _init_worker():
                sys.stdout = _DEVNULL

            # Ignore RuntimeWarnings only for the multiprocessing part
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                with Pool(self.args.num_proc, initializer=_init_worker) as pool:
                    sub_lists = pool.map(self._burden_test, seed_range)
                array_list = []
                for sub_list in sub_lists:
                    array_list.extend(sub_list)

        _worker_data.clear()
        return array_list
    
    @staticmethod
    def _burden_test(seed_range: tuple):
        # Read shared data from module-level global (inherited via fork, no pickling)
        var_counts = _worker_data['var_counts']
        case_cnt = _worker_data['case_cnt']
        ctrl_cnt = _worker_data['ctrl_cnt']
        use_n_carrier = _worker_data['use_n_carrier']
        burden_shift = _worker_data['burden_shift']

        total_cnt = case_cnt + ctrl_cnt
        num_perms = seed_range[1] - seed_range[0]
        num_cats = var_counts.shape[1]

        # Prepare data matrix
        if use_n_carrier:
            data = (var_counts > 0).astype(np.float64)
        else:
            data = var_counts if var_counts.dtype == np.float64 else var_counts.astype(np.float64)

        # Pre-compute total counts once: ctrl_counts = total - case_counts
        total_counts = data.sum(axis=0)  # (num_cats,)

        binom_p = case_cnt / total_cnt

        # Pre-allocate result arrays
        perm_rrs = np.empty((num_perms, num_cats), dtype=np.float64)
        if burden_shift:
            binom_pvals_arr = np.empty((num_perms, num_cats), dtype=np.float64)

        # Process in chunks to limit memory usage
        CHUNK_SIZE = 100
        for chunk_start in tqdm(range(0, num_perms, CHUNK_SIZE), desc="Processing", position=0, leave=True):
            chunk_end = min(chunk_start + CHUNK_SIZE, num_perms)
            chunk_size = chunk_end - chunk_start

            # Generate all case assignments for this chunk (same seeds for reproducibility)
            case_indicator = np.zeros((chunk_size, total_cnt), dtype=np.float32)
            for i in range(chunk_size):
                seed = 10001 + seed_range[0] + chunk_start + i
                np.random.seed(seed=seed)
                idx = np.random.choice(total_cnt, case_cnt, replace=False)
                case_indicator[i, idx] = 1.0

            # Batch matrix multiply: (chunk, total_cnt) @ (total_cnt, num_cats)
            n1 = case_indicator @ data   # case counts: (chunk, num_cats)
            n2 = total_counts - n1       # ctrl counts via subtraction

            if not use_n_carrier:
                np.round(n1, out=n1)
                np.round(n2, out=n2)

            with np.errstate(divide='ignore', invalid='ignore'):
                chunk_rrs = (n1 / case_cnt) / (n2 / ctrl_cnt)

            perm_rrs[chunk_start:chunk_end] = chunk_rrs

            if burden_shift:
                # Batch binom across all permutations in chunk (single vectorized call)
                binom_pvals_chunk = binom_two_tail_vectorized(
                    n1.ravel(), n2.ravel(), binom_p
                ).reshape(chunk_size, num_cats)
                binom_pvals_chunk[chunk_rrs < 1] *= -1
                binom_pvals_arr[chunk_start:chunk_end] = binom_pvals_chunk

        # Return in the expected interleaved format
        if burden_shift:
            result = np.empty((2 * num_perms, num_cats), dtype=np.float64)
            result[0::2] = perm_rrs
            result[1::2] = binom_pvals_arr
            return [result]
        else:
            return [perm_rrs]

    def get_perm_pval(self, perm_rrs, rr: np.ndarray):
        ## Permutation tests
        ## Check whether a permutation RR is more extreme than the original RR
        are_ext_rr = ((rr >= 1) & (perm_rrs >= rr)) | ((rr < 1) & (perm_rrs <= rr))
        ext_rr_cnt = are_ext_rr.sum(axis=0)
        return (ext_rr_cnt + 1) / (are_ext_rr.shape[0] + 1)
        
    def save_result(self):
        super().save_result()
        if self.burden_shift:
            print_progress(f"Save the binomial p values to the file {self.binom_pvals_path}")
            self._binom_pvals.to_parquet(self.binom_pvals_path, engine='pyarrow')