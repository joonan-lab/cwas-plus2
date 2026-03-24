.. _burdenshift:

*********************************
Burden shift analysis
*********************************

From burden test results, CWAS-Plus2 can find overrepresented domains across significant categories. By comparing the observed number of significant categories with the number of significant categories from samples with a randomly shuffled phenotype, the significance of the overrepresentation is determined.

The parameters of the command are as below:

- -i, --input_file: Path to the input file which is the result of binomial burden test (\*.burden_test.txt).
- -b, --burden_res: Path to the result of burden shift from permutation test (\*.binom_pvals.parquet or \*.binom_pvals.txt.gz).
- -c_info, --category_info: Path to a text file with category information (`*.category_info.txt`).
- -o_dir, --output_directory: Path to the directory where the output files will be saved. By default, outputs will be saved at ``$CWAS_WORKSPACE``.
- -c_set, --cat_set: Path to the category information file from binomial burden test (\*.category_info.txt).
- -c_count, --cat_count: Path of the categories counts file from binomial burden test (\*.category_counts.txt).
- -t, --tag: Tag used for the name of the output files. By default, None.
- -c_cutoff, --count_cutoff: The number of cutoff for category counts. It must be positive value. By default, 7.
- --pval: P-value threshold. By default, 0.05.
- -N, --n_cat_sets: The number of the category sets contained in the main output plot. Top N category sets will be displayed. By default, 10.
- -pt, --plot_title: Title of summarized plot of burden shift result. By default, "Burdenshift: Overrepresented terms".
- -fs, --fontsize: Font size of final main output plot. By default, 10.



.. code-block:: solidity
  
  cwas burden_shift -i INPUT.burden_test.txt \
  -b INPUT.binom_pvals.parquet \
  -o_dir OUTPUT_DIR \
  -c_info INPUT.category_info.txt \
  -c_count INPUT.category_counts.txt \
  -c_cutoff 7 \
  --pval 0.05


