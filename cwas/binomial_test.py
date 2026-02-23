import numpy as np
import pandas as pd
import zarr
from scipy.stats import norm
import matplotlib.pyplot as plt
import os

from cwas.burden_test import BurdenTest
from cwas.core.burden_test.binomial import (
    binom_one_tail, binom_two_tail,
    binom_one_tail_vectorized, binom_two_tail_vectorized,
)
from cwas.core.categorization.category import Category
from cwas.core.common import cmp_two_arr
from cwas.utils.log import print_progress
from scipy.stats import binomtest
from decimal import Decimal


class BinomialTest(BurdenTest):
    def __init__(self, args=None):
        super().__init__(args)
        self._categories = None

    @property
    def binom_p(self) -> float:
        return self.case_cnt / (self.case_cnt + self.ctrl_cnt)

    @property
    def count_thres(self) -> int:
        if self._count_thres is None:
            m = 1
            while True:
                p_value = binomtest(m-1, m, self.binom_p, alternative='greater').pvalue
                if p_value < 0.05:
                    return m
                m += 1

    @property
    def category_table(self) -> pd.DataFrame:
        if self._categories is not None:
            categories = [
                Category.from_str(cat_str).to_dict()
                for cat_str in self._categories
            ]
            return pd.DataFrame(categories, index=self._categories)
        return super().category_table

    def run(self):
        # If _categorization_result was pre-set (e.g. test fixtures), use the base class path
        if self._categorization_result is not None:
            return super().run()

        print_progress("Load the categorization result")
        root = zarr.open(self.cat_path, mode='r')
        sample_ids = list(root['metadata'].attrs['sample_id'])
        self._categories = list(root['metadata'].attrs['category'])

        # Validate sample IDs match sample_info
        if not cmp_two_arr(np.array(sample_ids), self.sample_info.index.values):
            raise ValueError(
                "The sample IDs from the sample information are "
                "not the same with the sample IDs "
                "from the categorization result."
            )

        # Set phenotypes in zarr sample order (enables binom_p, case_cnt, ctrl_cnt)
        self._phenotypes = np.array(self.sample_info.loc[sample_ids, 'PHENOTYPE'])
        is_case = self._phenotypes == "case"
        is_ctrl = self._phenotypes == "ctrl"

        # Read raw data as numpy
        print_progress("Read zarr data")
        data = root['data'][:]  # int32, shape (n_samples, n_categories)
        n_samples = len(sample_ids)

        if self.use_n_carrier:
            # Carrier mode: binary matrix, no adjustment needed
            binary = data > 0
            raw_counts = binary.sum(axis=0)
            self._raw_counts = pd.DataFrame(
                {'Raw_counts': raw_counts},
                index=pd.Index(self._categories, name='Category'),
            )

            case_carrier = binary[is_case].sum(axis=0)
            ctrl_carrier = binary[is_ctrl].sum(axis=0)
            del binary, data

            self._result = pd.DataFrame({
                'Case_Carrier_Count': case_carrier,
                'Ctrl_Carrier_Count': ctrl_carrier,
            }, index=self._categories)

            np.seterr(divide='ignore')
            self._result['Relative_Risk'] = (
                (case_carrier / self.case_cnt) / (ctrl_carrier / self.ctrl_cnt)
            )

            n1 = case_carrier.astype(np.int64)
            n2 = ctrl_carrier.astype(np.int64)
        else:
            # Variant count mode: matmul with adjustment weights
            weights = np.zeros((3, n_samples), dtype=np.float64)
            weights[0] = 1.0  # raw counts (unadjusted)

            if self.adj_factor is not None:
                if not cmp_two_arr(np.array(sample_ids), self.adj_factor.index.values):
                    raise ValueError(
                        "The sample IDs from the adjustment factor list are "
                        "not the same with the sample IDs "
                        "from the categorization result."
                    )
                adj = np.array(
                    [self.adj_factor.loc[sid, 'AdjustFactor'] for sid in sample_ids],
                    dtype=np.float64,
                )
            else:
                adj = np.ones(n_samples, dtype=np.float64)

            weights[1] = adj * is_case.astype(np.float64)
            weights[2] = adj * is_ctrl.astype(np.float64)

            print_progress("Compute category sums via matmul")
            data_f64 = data.astype(np.float64)
            del data
            sums = weights @ data_f64  # (3, n_categories)
            del data_f64

            raw_counts = sums[0]
            case_sum = sums[1]
            ctrl_sum = sums[2]

            self._raw_counts = pd.DataFrame(
                {'Raw_counts': raw_counts.astype(np.int64)},
                index=pd.Index(self._categories, name='Category'),
            )

            self._result = pd.DataFrame({
                'Case_DNV_Count': case_sum,
                'Ctrl_DNV_Count': ctrl_sum,
            }, index=self._categories)

            np.seterr(divide='ignore')
            self._result['Relative_Risk'] = (
                (case_sum / self.case_cnt) / (ctrl_sum / self.ctrl_cnt)
            )

            n1 = np.round(case_sum).astype(np.int64)
            n2 = np.round(ctrl_sum).astype(np.int64)

        # Vectorized binomial tests
        binom_p = self.binom_p
        print_progress("Run binomial test")
        self._result['P'] = binom_two_tail_vectorized(n1, n2, binom_p)
        self._result['P_1side'] = binom_one_tail_vectorized(n1 + 1, n2 + 1, binom_p)
        self._result['Z_1side'] = norm.ppf(1 - self._result['P_1side'].values)

        self._draw_volcano_plot()
        self.concat_category_info()
        self.save_result()
        self.save_counts_table(form='adj')
        self.save_category_info()
        self.update_env()

    def run_burden_test(self):
        print_progress("Run binomial test")
        if self.use_n_carrier:
            n1 = self.case_carrier_cnt
            n2 = self.ctrl_carrier_cnt
        else:
            n1 = self.case_variant_cnt.round()
            n2 = self.ctrl_variant_cnt.round()
        self._result["P"] = np.vectorize(binom_two_tail)(
            n1,
            n2,
            self.binom_p,
        )

        # Add the pseudocount(1) in order to avoid p-values of one
        self._result["P_1side"] = np.vectorize(binom_one_tail)(
            n1 + 1,
            n2 + 1,
            self.binom_p,
        )
        self._result["Z_1side"] = norm.ppf(1 - self._result["P_1side"].values)

        self._draw_volcano_plot()

    def _draw_volcano_plot(self):
        burden_res = self._result.copy()
        burden_res['log2_RR'] = burden_res['Relative_Risk'].apply(lambda x: np.log2(x))
        burden_res['-log_P'] = burden_res['P'].apply(lambda x: -np.log10(x))
        
        print_progress(f"Volcano plot will display categories with at least {self.count_thres} counts")
        selected_categories = self._raw_counts[self._raw_counts['Raw_counts'] >= self.count_thres].index.tolist()        
        burden_res = burden_res[burden_res.index.isin(selected_categories)]
        
        threshold = -np.log10(0.05)
        eff_threshold = None
        if self.eff_test:
            eff_threshold = -np.log10(0.05/self.eff_test)
        max_rr = max(burden_res.loc[burden_res.log2_RR!=np.inf, 'log2_RR'])
        min_rr = min(burden_res.loc[burden_res.log2_RR!=-np.inf, 'log2_RR'])
        max_val = max(abs(max_rr),abs(min_rr))
        max_x = np.trunc(max_val) + 2

        xticks = [int(x) for x in np.arange(-max_x, max_x+1, 2)]
        xlabels = xticks.copy()
        xlabels[0] = '-Inf'
        xlabels[-1] = 'Inf'
        max_logp = burden_res.loc[burden_res['-log_P'] != np.inf, '-log_P']
        max_y = np.trunc(max(max_logp)) + 2 if len(max_logp) > 0 else 2
        yticks = [int(x) for x in np.arange(0, max_y + 1, 2)]
        ylabels = yticks.copy()

        def replace_inf(x, v):
            if x == np.inf:
                return v
            elif x == -np.inf:
                return -v
            else:
                return x

        if self.tag != None:
            tags = self.tag.strip().split(",")
            for t in tags:
                self._draw_single_volcano_plot(burden_res, max_x, max_y, threshold, eff_threshold, xticks, xlabels, yticks, ylabels, replace_inf, tag_name=t)
        else:
            self._draw_single_volcano_plot(burden_res, max_x, max_y, threshold, eff_threshold, xticks, xlabels, yticks, ylabels, replace_inf)

    def _draw_single_volcano_plot(self, burden_res, max_x, max_y, threshold, eff_threshold, xticks, xlabels, yticks, ylabels, replace_inf, tag_name=None):
        fig, axes = plt.subplots(figsize=(self.plot_size, self.plot_size))

        plt.title(self.plot_title, fontsize=self.font_size, loc='left', pad=5)
        axes.vlines(x=0, ymin=-0.5, ymax=max_y+0.5, linestyles='-', color='lightgray', linewidth=1.25, zorder=1)
        axes.scatter(x=burden_res['log2_RR'].apply(lambda x: replace_inf(x, max_x)), y=burden_res['-log_P'].apply(lambda x: replace_inf(x, max_y)),
                    marker='o', color='silver', s=self.marker_size, label='Others' if tag_name else None, edgecolor='black', linewidth=0.5, zorder=2)
        if tag_name is not None:
            tag_mask = (burden_res.index.str.contains(tag_name)) & (burden_res['-log_P'] > threshold)
            axes.scatter(x=burden_res.loc[tag_mask, 'log2_RR'].apply(lambda x: replace_inf(x, max_x)),
                         y=burden_res.loc[tag_mask, '-log_P'].apply(lambda x: replace_inf(x, max_y)),
                         marker='o', label=tag_name, facecolor='#3d62a1', s=self.marker_size, alpha=.7, edgecolor='black', linewidth=0.5, zorder=3)
        axes.hlines(y=threshold, xmin=-(max_x+1), xmax=max_x+1, linestyles='--', linewidth=1.25, color='black')
        axes.text(-(max_x+1)+0.1, threshold+0.1, 'P=0.05', size=self.font_size*0.85, color='black')
        if self.eff_test:
            axes.hlines(y=eff_threshold, xmin=-(max_x+1), xmax=max_x+1, linestyles='--', linewidth=1.25, color='red')
            axes.text(-(max_x+1)+0.1, eff_threshold+0.1, ''.join(['P=', str('%.2E' % Decimal(0.05/self.eff_test)),', eff_num=', format(self.eff_test, ',')]), size=self.font_size*0.85, color='red')
        plt.ylim(-0.5, max_y+0.5)
        plt.xlim(-(max_x+1), (max_x+1))
        plt.xlabel('Relative Risk ($log_{2}$)', size=self.font_size)
        plt.ylabel("P ($-log_{10}$)", size=self.font_size)
        axes.set_xticks(xticks)
        axes.set_xticklabels(xlabels, fontsize=self.font_size)
        axes.set_yticks(yticks)
        axes.set_yticklabels(ylabels, fontsize=self.font_size)
        if tag_name is not None:
            axes.legend(markerscale=self.font_size*0.15, fontsize=self.font_size*0.85)
        axes.spines['bottom'].set_linewidth(1.25)
        axes.spines['left'].set_linewidth(1.25)
        axes.tick_params(width=1.25)
        axes.spines['top'].set_visible(False)
        axes.spines['right'].set_visible(False)

        suffix = f'.{tag_name}.volcano_plot.pdf' if tag_name else '.volcano_plot.pdf'
        output = self.result_path.name.replace(".txt", suffix)
        output_path = self.output_dir_path / output

        plt.tight_layout()
        plt.savefig(output_path, bbox_inches='tight')
        print_progress("Save the result to the volcano plot file {}".format(output_path))
