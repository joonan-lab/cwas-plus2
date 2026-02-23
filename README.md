# Category-wide association study (CWAS)

![CWAS CI Workflow](https://github.com/joonan30/cwas-plus2/actions/workflows/ci.yml/badge.svg)

**CWAS (Category-Wide Association Study)** is a data analytic tool to perform stringent association tests to find non-coding loci associated with autism spectrum disorder (ASD). CWAS runs category-based burden tests using de novo variants from whole genome sequencing data and diverse annotation data sets.

CWAS was used in the following papers.

- [An analytical framework for whole genome sequence association studies and its implications for autism spectrum disorder](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5961723/) (Werling _et al._, 2018)
- [Genome-wide de novo risk score implicates promoter variation in autism spectrum disorder](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6432922/) (An _et al._, 2018)
- [CWAS-Plus: Estimating category-wide association of rare noncoding variation from whole-genome sequencing data with cell-type-specific functional data](https://www.medrxiv.org/content/10.1101/2024.04.15.24305828v1) (Kim et al., in press)

Here is _the original CWAS repository: [sanderslab/cwas](https://github.com/sanderslab/cwas)_

## Quickstart

### Data requirements

Users must prepare following data for CWAS because it is very essential but cannot be generated automatically. Here are details.

#### 1. Input VCF data (De novo variant list)

```
#CHROM  POS ID  REF ALT QUAL    FILTER  INFO
chr1    3747728 .        T       C       .       .       SAMPLE=11000.p1
chr1    38338861        .       C       A       .       .       SAMPLE=11000.p1
chr1    117942118       .      T       G       .       .       SAMPLE=11000.p1
```

- The input VCF data must follow the [specification of VCF](https://samtools.github.io/hts-specs/VCFv4.2.pdf).
- The _INFO_ field must contain a sample ID of each variant with this format `SAMPLE={sample_id}`.

#### 2. List of samples

|  SAMPLE  | PHENOTYPE |
| :------: | :-------: |
| 11000.p1 |   case    |
| 11000.s1 |   ctrl    |
| 11002.p1 |   case    |
| 11002.s1 |   ctrl    |

- CWAS requires the file like above listing sample IDs with its family IDs and phenotypes (Case=_case_, Control=_ctrl_).
- Here are details of the required format.
  - Tab separated
  - 2 essential columns: _SAMPLE_ and _PHENOTYPE_
  - A value in the _PHENOTYPE_ must be _case_ or _ctrl_.
- The values in the _SAMPLE_ must be matched with the sample IDs of variants in the input VCF file.

#### 3. List of adjustment factors (Optional)

|  SAMPLE  | AdjustFactor |
| :------: | :----------: |
| 11000.p1 |    0.932     |
| 11000.s1 |    1.082     |
| 11002.p1 |    0.895     |
| 11002.s1 |    1.113     |

- The file like above is required if you want to adjust the number of variants for each sample in CWAS.
- Here are details of the required format.
  - Tab separated
  - 2 essential columns: _SAMPLE_ and _AdjustFactor_
  - A value in the _AdjustFactor_ must be a float.
- The values in the _SAMPLE_ must be matched with the sample IDs of variants in the input VCF file.

You can get the examples of the above data requirements from **[joonan-lab/cwas-input-example](https://github.com/joonan-lab/cwas-input-example)**

#### 4. CWAS annotation files

You can install those file from this repository: **[joonan-lab/cwas-dataset](https://github.com/joonan-lab/cwas-dataset)**

```bash
git clone https://github.com/joonan-lab/cwas-dataset.git
```

Due to the sizes of BigWig files for conservation scores, you must install them manually. [Please follow this instruction](https://github.com/joonan-lab/cwas-dataset/blob/main/bw_recipe.md).

### Installation

We recomment using _[conda virtual environment](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)_ to build environment for CWAS. Installing CWAS-Plus within a conda environment will prevent its installation in the global environment. When creating a conda environment, also install Python to enable local installations using pip. We recommend installing R too. Run the following statements in your shell.

```bash
conda create -n cwas python r-base
conda activate cwas
git clone https://github.com/joonan30/cwas-plus2.git
cd cwas-plus2
pip install .
```

- Python >=3.9, <3.13 is required.


In addition, you must install _[Variant Effect Predictor (VEP)](https://www.ensembl.org/vep)_ unless you plan to use the `--docker-mode` option in the annotation step.

### CWAS Execution

#### 1. Start

Run this command.

```bash
cwas start
```

As default, this command creates _CWAS workspace_ in your home directory. The path is `$HOME/.cwas`. `$HOME/.cwas/configuration.txt` has also generated.

Alternatively, _CWAS workspace_ can be specified with `-w` option. The `configuration.txt` will also be generated in the specified _CWAS workspace_.

```bash
.cwas
└── configuration.txt
```

#### 2. Configuration

Write the following information in the `$HOME/.cwas/configuration.txt`.

```bash
ANNOTATION_DATA_DIR=/path/to/your/dir
GENE_MATRIX=/path/to/your/file
ANNOTATION_KEY_CONFIG=/path/to/your/file
VEP=/path/to/your/vep
VEP_CACHE_DIR=/path/to/your/vep/cache/dir
VEP_CONSERVATION_FILE=/path/to/your/vep/resource
VEP_LOFTEE=/path/to/your/vep/resource
VEP_HUMAN_ANCESTOR_FA=/path/to/your/vep/resource
VEP_GERP_BIGWIG=/path/to/your/vep/resource
VEP_MIS_DB=/path/to/your/missense/database
VEP_MIS_INFO_KEY=
VEP_MIS_THRES=
```

The `ANNOTATION_DATA` is a directory that contains all the BED files from **[joonan-lab/cwas-dataset](https://github.com/joonan-lab/cwas-dataset)**.
The `VEP_MIS_DB` is a database that is used to define damaging missense variants. The `VEP_MIS_INFO_KEY` is an user-defined name of the database used to annotate variants. The `VEP_MIS_THRES` is a threshold for missense variants (missense variants with value>=threshold are defined as damaging missense variants).


After writing the above file, run this command.

```bash
cwas configuration [-f]
```

| Argument | Description |
| --- | --- |
| `-f`, `--force_overwrite` | Force to overwrite the result |

Following files will be generated in your home directory as default. If you specify _CWAS workspace_, the files will be located in the same directory as the `configuration.txt`.

```bash
.cwas
├── annotation-data
├── annotation_keys.yaml
├── category_domain.yaml
├── configuration.txt
├── gene_matrix.txt
└── redundant_category.txt
.cwas_env
```

#### 3. Preparation

This step merges the BED files to annotate variants. Run the following command.

```bash
cwas preparation [-p 4] [-f]
```

| Argument | Description | Default |
| --- | --- | --- |
| `-p`, `--num_proc` | Max number of worker processes | 1 |
| `-f`, `--force_overwrite` | Force to overwrite the result | - |

After running this, merged BED file and its index will be generated in your _CWAS workspace_.

```bash
.cwas
...
├── merged_annotation.bed.gz
├── merged_annotation.bed.gz.tbi
...
```

#### 4. Annotation

This step annotates your VCF file using _VEP_. Run this command.

```bash
cwas annotation -v /path/to/your/vcf [-p 4] [-o_dir /path/to/output] [--docker-mode] [--vep-version 115.0]
```

| Argument | Description | Default |
| --- | --- | --- |
| `-v`, `--vcf_file` | **(Required)** Target VCF file | - |
| `-p`, `--num_proc` | Number of processes for the annotation | 1 |
| `-o_dir`, `--output_directory` | Directory where output file will be saved | `$CWAS_WORKSPACE` |
| `--docker-mode` | Run VEP using Docker (`ensemblorg/ensembl-vep`) instead of a local binary | False |
| `--vep-version` | VEP version for Docker image tag (e.g. `115.0`). Only used with `--docker-mode` | latest |

Here is the result file.

```bash
.cwas
...
├── {Your VCF filename}.annotated.vcf.gz
...
```

#### 5. Categorization

This step categorizes your variants using the annotation datasets. Run this command.

```bash
cwas categorization -i /path/to/your/annotated/vcf [-p 4] [-o_dir /path/to/output]
```

| Argument | Description | Default |
| --- | --- | --- |
| `-i`, `--input_file` | **(Required)** Annotated VCF file | - |
| `-p`, `--num_proc` | Number of worker processes | 1 |
| `-o_dir`, `--output_directory` | Directory where output file will be saved | `$CWAS_WORKSPACE` |

After running this, you will get...

```bash
.cwas
...
├── {Your VCF filename}.categorization_result.zarr
...
```

Categorized results are generated in [zarr format](https://zarr.readthedocs.io/en/stable/index.html). Outputs are easily stored and loaded with zarr.

#### 6. Burden Test

This step is for calculation of relative risks and p-values for each category. As a default, these tests are based on variant-level analysis. The `--use_n_carrier` option can be used for sample-level analysis.

##### Binomial test

This step runs category-based burden test using the categorization result. The type of the test is binomial test. Run this command.

```bash
cwas binomial_test -i /path/to/your/categorization/result -s /path/to/your/samples \
    [-a /path/to/your/adj_factors] [-u] [-t TAG] [-num_ef N]
```

| Argument | Description | Default |
| --- | --- | --- |
| `-i`, `--input_file` | **(Required)** Categorized file (*.zarr) from categorization step | - |
| `-s`, `--sample_info` | **(Required)** File listing information of your samples | - |
| `-a`, `--adjustment_factor` | File listing adjustment factors of each sample | None |
| `-u`, `--use_n_carrier` | Use the number of samples with variants instead of variant counts | False |
| `-t`, `--tag` | Tags for highlighting points on the volcano plot (e.g. `CRE,CHD8`) | None |
| `-num_ef`, `--num_effective_test` | Number of effective tests | None |
| `-o_dir`, `--output_directory` | Directory where output file will be saved | `$CWAS_WORKSPACE` |
| `-ms`, `--marker_size` | Marker size of the volcano plot (pt) | 15 |
| `-fs`, `--font_size` | Font size of the volcano plot (pt) | 15 |
| `-ps`, `--plot_size` | Plot size of the volcano plot (inch, square) | 7 |
| `-pt`, `--plot_title` | Title of the volcano plot | "Binomial test result" |

After running this, you will get...

```bash
.cwas
...
├── {Your VCF filename}.burden_test.txt
├── {Your VCF filename}.burden_test.volcano_plot.pdf
├── {Your VCF filename}.category_counts.txt
├── {Your VCF filename}.category_info.txt
...
```

##### Permutation test

This step runs category-based permutations using the categorization result. Run this command.

```bash
cwas permutation_test -i /path/to/your/categorization/result -s /path/to/your/samples \
    [-a /path/to/your/adj_factors] [-n 10000] [-p 4] [-b] [-u]
```

| Argument | Description | Default |
| --- | --- | --- |
| `-i`, `--input_file` | **(Required)** Categorized file (*.zarr) from categorization step | - |
| `-s`, `--sample_info` | **(Required)** File listing information of your samples | - |
| `-a`, `--adjustment_factor` | File listing adjustment factors of each sample | None |
| `-n`, `--num_perm` | Number of label-swapping permutations | 10000 |
| `-p`, `--num_proc` | Number of worker processes | 1 |
| `-b`, `--burden_shift` | Generate binomial p-values for each permutation (for burden shift and DAWN) | False |
| `-u`, `--use_n_carrier` | Use the number of samples with variants instead of variant counts | False |
| `-o_dir`, `--output_directory` | Directory where output file will be saved | `$CWAS_WORKSPACE` |

After running this, you will get...

```bash
.cwas
...
├── {Your VCF filename}.permutation_test.txt
├── {Your VCF filename}.binom_pvals.parquet
...
```

#### 7. Correlation

This step generates a correlation matrix between categories.

```bash
cwas correlation -i /path/to/your/categorization/result -cm variant [-v /path/to/annotated/vcf] \
    [-im] [-p 4] [-d all]
```

| Argument | Description | Default |
| --- | --- | --- |
| `-i`, `--input_file` | **(Required)** Categorized file (*.zarr) from categorization step | - |
| `-cm`, `--corr_matrix` | **(Required)** Correlation matrix type: `variant` or `sample` | - |
| `-v`, `--annotated_vcf` | Annotated VCF file. Required for variant-level correlation (`-cm variant`) | None |
| `-im`, `--intersection_matrix` | Also generate a matrix of intersected variant/sample counts between categories | False |
| `-p`, `--num_proc` | Number of worker processes (recommend using 1/3 of available cores) | 1 |
| `-d`, `--domain_list` | Domain list to filter categories (see [Domain options](#domain-options)) | all |
| `-c_info`, `--category_info` | Path to category information file (*.category_info.txt) | None |
| `-o_dir`, `--output_directory` | Directory where output file will be saved | `$CWAS_WORKSPACE` |

#### 8. Effective Number of Tests

This step calculates the effective number of tests via eigen decomposition of the correlation or intersection matrix.

```bash
cwas effective_num_test -i /path/to/your/matrix -c_count /path/to/category_counts \
    [-if corr] [-n 10000] [-thr N] [-s /path/to/samples] [-d all] [-ef]
```

| Argument | Description | Default |
| --- | --- | --- |
| `-i`, `--input_file` | **(Required)** Correlation matrix or intersection matrix from categorization | - |
| `-c_count`, `--cat_count` | **(Required)** Category counts file from burden test or sign test | - |
| `-if`, `--input_format` | Input format: `corr` (correlation matrix) or `inter` (intersection matrix) | corr |
| `-n`, `--num_eig` | Number of eigenvalues to use | 10000 |
| `-thr`, `--threshold` | Minimum variant/sample count to filter categories | None |
| `-s`, `--sample_info` | Sample info file (required when `-thr` is not given) | None |
| `-c_set`, `--category_set` | Text file containing categories for eigen decomposition | None |
| `-d`, `--domain_list` | Domain list to filter categories (see [Domain options](#domain-options)) | all |
| `-ef`, `--eff_num_test` | Only calculate effective number of tests (eigenvalues only, skip eigenvectors) | False |
| `-t`, `--tag` | Tag used for the name of output files | None |
| `-o_dir`, `--output_directory` | Directory where output file will be saved | `$CWAS_WORKSPACE` |

#### 9. Burden Shift

This step performs burden shift analysis using permutation results.

```bash
cwas burden_shift -i /path/to/burden_test.txt -b /path/to/binom_pvals.parquet \
    -c_info /path/to/category_info.txt -c_count /path/to/category_counts.txt \
    [-c_cutoff 7] [-pval 0.05] [-N 10]
```

| Argument | Description | Default |
| --- | --- | --- |
| `-i`, `--input_file` | **(Required)** Burden test result (*.burden_test.txt) | - |
| `-b`, `--burden_res` | **(Required)** Burden shift result from permutation (*.binom_pvals.parquet or *.binom_pvals.txt.gz) | - |
| `-c_info`, `--category_info` | **(Required)** Category information file (*.category_info.txt) | - |
| `-c_count`, `--cat_count` | **(Required)** Category counts file (*.category_counts.txt) | - |
| `-c_cutoff`, `--count_cutoff` | Minimum count cutoff for categories (must be positive) | 7 |
| `-pval`, `--pval` | P-value threshold | 0.05 |
| `-c_set`, `--category_set` | List of interest category sets for the main output plot | None |
| `-N`, `--n_cat_sets` | Number of top category sets in the main output plot | 10 |
| `-t`, `--tag` | Tag used for the name of output files | None |
| `-pt`, `--plot_title` | Title of the summarized plot | "Burdenshift: Overrepresented terms" |
| `-fs`, `--fontsize` | Font size of the main output plot | 10 |
| `-o_dir`, `--output_directory` | Directory where output file will be saved | `$CWAS_WORKSPACE` |

#### 10. Risk Score

This step performs risk score analysis using categorization results.

```bash
cwas risk_score -i /path/to/categorization_result.zarr -s /path/to/samples \
    [-a /path/to/adj_factors] [-d all] [-thr 3] [-n 1000] [-p 4]
```

| Argument | Description | Default |
| --- | --- | --- |
| `-i`, `--input_file` | **(Required)** Categorization result file (*.zarr) | - |
| `-s`, `--sample_info` | **(Required)** File listing sample IDs with families and phenotypes | - |
| `-a`, `--adjustment_factor` | File listing adjustment factors of each sample | None |
| `-c_info`, `--category_info` | Path to category information file (*.category_info.txt) | None |
| `-d`, `--domain_list` | Domain list to filter categories (see [Domain options](#domain-options)) | all |
| `-t`, `--tag` | Tag used for the name of output files | None |
| `--do_each_one` | Use each annotation individually to calculate risk score | False |
| `--leave_one_out` | Calculate risk score excluding one annotation at a time | False |
| `-fs_group`, `--feature_selection_group` | Groups for feature selection | "gene_set,functional_score,functional_annotation" |
| `-u`, `--use_n_carrier` | Use number of samples with variants instead of variant counts | False |
| `-thr`, `--threshold` | Minimum variant count in controls to select rare categories | 3 |
| `-tf`, `--train_set_fraction` | Fraction of the training set | 0.7 |
| `-n_reg`, `--num_regression` | Number of regression trials for mean R-squared | 10 |
| `-f`, `--fold` | Number of folds in (Stratified)KFold | 5 |
| `-n`, `--n_permute` | Number of permutations for p-value calculation | 1000 |
| `--predict_only` | Only predict the risk score, skip permutation test | False |
| `-p`, `--num_proc` | Number of worker processes for permutation | 1 |
| `-S`, `--seed` | Seed of random state | 42 |
| `-pt`, `--plotsize` | Plot size as "width,height" in inches | "7,7" |
| `-fs`, `--fontsize` | Font size of the main plot | 10 |
| `-o_dir`, `--output_directory` | Directory where output file will be saved | `$CWAS_WORKSPACE` |

#### 11. DAWN Analysis

This step performs DAWN (Detecting Association With Networks) analysis.

```bash
cwas dawn -e /path/to/eig_vecs -c /path/to/corr_matrix -P /path/to/permutation_test \
    -c_count /path/to/category_counts [-k K] [-p 4]
```

| Argument | Description | Default |
| --- | --- | --- |
| `-e`, `--eig_vector` | **(Required)** Eigenvectors file from effective number test | - |
| `-c`, `--corr_mat` | **(Required)** Category correlation matrix file | - |
| `-P`, `--permut_test` | **(Required)** Permutation test result file | - |
| `-c_count`, `--cat_count` | **(Required)** Category counts file from burden test | - |
| `--leiden` | Perform Leiden clustering using: `eigen_vector` or `corr_mat` | None |
| `-res`, `--resolution` | Resolution for Leiden clustering | 1 |
| `-r`, `--range` | Range (start,end) to find optimal K for k-means (start > 1) | "2,100" |
| `-k`, `--k_val` | Fixed K for k-means clustering (mutually exclusive with `-r`) | None |
| `-s`, `--seed` | Seed value for t-SNE | 42 |
| `-T`, `--tsen_method` | t-SNE gradient algorithm: `barnes_hut` or `exact` | exact |
| `-t`, `--tag` | Tag used for the name of output files | None |
| `-l`, `--lambda` | Lambda value for parameter tuning | 5.25 |
| `-C`, `--count_threshold` | Minimum variant/sample count per category | 20 |
| `-R`, `--corr_threshold` | Correlation threshold between clusters | 0.12 |
| `-S`, `--size_threshold` | Minimum number of categories per cluster | 2 |
| `--no-parsimonious` | Disable parsimonious K selection (use absolute silhouette max) | Enabled |
| `-p`, `--num_proc` | Number of worker processes | 1 |
| `-o_dir`, `--output_directory` | Directory where output file will be saved | `$CWAS_WORKSPACE` |

#### 12. Extract Variant

This step extracts variants of interest from the annotated VCF file.

```bash
cwas extract_variant -i /path/to/annotated/vcf [-c_set /path/to/category_set] [-t TAG] [-ai]
```

| Argument | Description | Default |
| --- | --- | --- |
| `-i`, `--input_file` | **(Required)** Annotated VCF file | - |
| `-c_set`, `--category_set` | Text file containing categories for extracting variants | None |
| `-t`, `--tag` | Tag used for output file name (output.\<tag\>.extracted_variants.txt.gz) | None |
| `-ai`, `--annotation_info` | Save with annotation information attached (gene list, functional annotations, etc) | False |
| `-o_dir`, `--output_directory` | Directory where output file will be saved | `$CWAS_WORKSPACE` |

### Domain options

Several steps (`correlation`, `effective_num_test`, `risk_score`, `dawn`) support the `-d` / `--domain_list` argument to filter categories by GENCODE domain. Available options:

| Option | Description |
| --- | --- |
| `run_all` | Test all available domain options |
| `all` | Include all domains (default) |
| `coding` | Coding regions |
| `noncoding` | Non-coding regions |
| `ptv` | Protein-truncating variants |
| `missense` | Missense variants |
| `damaging_missense` | Damaging missense variants |
| `promoter` | Promoter regions |
| `noncoding_wo_promoter` | Non-coding regions without promoter |
| `splice` | Splice site regions |
| `intron` | Intronic regions |
| `intergenic` | Intergenic regions |
| `3primeutr` | 3' UTR |
| `5primeutr` | 5' UTR |
| `utr` | UTR regions |
| `lincRNA` | Long intergenic non-coding RNA |
