.. _dawn:

*********************************
DAWN analysis
*********************************

The users can investigate the relationship between categories and identify the specific type of categories that are clustered within the network of categories of interest.

  - -e, --eig_vector: Eigen vector file. This is the output file from :ref:`calculation of effective number of tests <effnumtest>`. The file name must have pattern ``*eig_vecs*.zarr``.
  - -c, --corr_mat: Category correlation matrix file. This is the output file from :ref:`correlation <correlation>`. The file name must have pattern ``*correlation_matrix*.zarr``.
  - -P, --permut_test: Permutation test file. This is the output file from :ref:`burden test <permtest>`. The file name must have pattern ``*permutation_test*.txt.gz``.
  - -c_count, --cat_count: Path of the categories counts file from burden test.
  - -o_dir, --output_directory: Path to the directory where the output files will be saved. By default, outputs will be saved at ``$CWAS_WORKSPACE``.
  - --leiden: Perform Leiden clustering. Specify the input matrix type: ``eigen_vector`` or ``corr_mat``. By default, None.
  - -res, --resolution: Resolution for Leiden clustering. By default, 1.
  - -r, --range: Range (i.e., (start,end)) to find optimal K for k-means clustering. It must contain two integers that are comma-separated. The first integer refers to the start number and must be above 1. The second integer refers to the end. By default, 2,100.
  - -k, --k_val: K for K-means clustering. With this argument, users can determine K manually. ``-r`` and ``-k`` arguments are mutually exclusive. If ``-k`` is given, ``-r`` will be ignored.
  - -s, --seed: Seed value for t-SNE. Same seed will generate same results for the same inputs. By default, 42.
  - -T, --tsen_method: Gradient calculation algorithm for t-SNE, which is used in TSNE of sklearn. If the dataset is large, 'barnes_hut' is recommended. By default, exact.
  - -t, --tag: Tag used for the name of the output files. By default, None.
  - -l, --lambda: Lambda value for parameter tuning. By default, 5.25.
  - -C, --count_threshold: The threshold of variant (or sample) counts. The least amount of variants a category should have. By default, 20.
  - -R, --corr_threshold: The threshold of correlation values between clusters. Computed by the mean value of correlation values of categories within a cluster. By default, 0.12.
  - -S, --size_threshold: The threshold of the number of categories per cluster. The least amount of categories a cluster should have. By default, 2.
  - -p, --num_proc: Number of worker processes that will be used for the DAWN analysis. By default, 1.


.. code-block:: solidity

    cwas dawn -e INPUT_EIG_VEC \
    -c INPUT_CORR_MATRIX \
    -P INPUT_PERMUATION_RESULT \
    -o_dir OUTPUT_DIR \
    -r 2,100 \
    -s 42 \
    -t test \
    -c_count CATEGORY_COUNTS.txt \
    -C 20 \
    -R 0.12 \
    -S 2 \
    -p 8



