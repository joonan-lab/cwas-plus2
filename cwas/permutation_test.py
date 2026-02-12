import argparse
import os
import sys
from functools import partial
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
from cwas.core.burden_test.binomial import binom_two_tail

_DEVNULL = open(os.devnull, 'w')

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
            f_name = re.sub(r'categorization_result\.zarr\.gz|categorization_result\.zarr', 'binom_pvals.txt.gz', self.cat_path.name)
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
        
        _burden_test_partial = partial(self._burden_test, 
                                       case_cnt=self.case_cnt,
                                       ctrl_cnt=self.ctrl_cnt,
                                       var_counts=var_counts,
                                       use_n_carrier=self.use_n_carrier,
                                       burden_shift=burden_shift)

        if self.args.num_proc == 1:
            array_list = _burden_test_partial(
                (0, num_perm),
            )
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
            def mute():
                sys.stdout = _DEVNULL

            # Ignore RuntimeWarnings only for the multiprocessing part
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                with Pool(self.args.num_proc, initializer=mute) as pool:
                    sub_lists = pool.map(_burden_test_partial, seed_range)
                array_list = []
                for sub_list in sub_lists:
                    array_list.extend(sub_list)

        return array_list
    
    @staticmethod
    def _burden_test(seed_range: tuple, case_cnt: int, ctrl_cnt: int, var_counts: np.ndarray, use_n_carrier: bool, burden_shift: bool):
        array_list = []
        total_cnt = case_cnt + ctrl_cnt

        # Pre-compute carrier matrix once if needed (avoid recomputation per permutation)
        if use_n_carrier:
            is_carrier = (var_counts > 0).astype(np.int32)
        else:
            is_carrier = None

        for seed in tqdm(range(10001 + seed_range[0], 10001 + seed_range[1]), desc="Processing", position=0, leave=True):
            ## For reproducibility
            np.random.seed(seed=seed)
            ## Use boolean array directly instead of string comparison
            are_case = np.zeros(total_cnt, dtype=bool)
            idx = np.random.choice(total_cnt, case_cnt, replace=False)
            are_case[idx] = True

            if use_n_carrier:
                n1 = is_carrier[are_case, :].sum(axis=0)
                n2 = is_carrier[~are_case, :].sum(axis=0)
            else:
                n1 = var_counts[are_case, :].sum(axis=0).round()
                n2 = var_counts[~are_case, :].sum(axis=0).round()

            norm_n1 = n1 / case_cnt
            norm_n2 = n2 / ctrl_cnt
            perm_rr = norm_n1 / norm_n2

            binom_p = case_cnt / total_cnt

            ## Calculate binomial p values for burden-shifted data
            if burden_shift:
                binom_pval = np.vectorize(binom_two_tail)(
                    n1, n2, binom_p
                )
                ## If RR is below 1, the sign of binomial p value is reversed
                binom_pval[perm_rr < 1] *= -1
                array_list.append(np.concatenate((perm_rr[np.newaxis, :], binom_pval[np.newaxis, :])))
            else:
                array_list.append(perm_rr[np.newaxis, :])
        return array_list

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
            self._binom_pvals.to_csv(self.binom_pvals_path, sep='\t', compression='gzip')