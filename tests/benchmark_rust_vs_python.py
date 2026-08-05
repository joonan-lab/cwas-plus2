"""
End-to-end benchmark: Python fallback vs Rust acceleration.
Uses the test VCF data for realistic comparison.
"""
import pathlib
import time
import sys
import numpy as np

TEST_VCF = pathlib.Path(__file__).parent / "test_file" / "test_annotated.vcf.gz"


def benchmark_vcf_parser():
    """Compare VCF parsing speed."""
    from cwas.core.categorization.parser import (
        _parse_annotated_vcf_python,
        _parse_annotated_vcf_rust,
    )

    # Warmup
    _parse_annotated_vcf_python(TEST_VCF)
    _parse_annotated_vcf_rust(TEST_VCF)

    N = 50
    start = time.perf_counter()
    for _ in range(N):
        df_py = _parse_annotated_vcf_python(TEST_VCF)
    py_time = (time.perf_counter() - start) / N

    start = time.perf_counter()
    for _ in range(N):
        df_rs = _parse_annotated_vcf_rust(TEST_VCF)
    rs_time = (time.perf_counter() - start) / N

    n_variants = len(df_py)
    speedup = py_time / rs_time
    print(f"{'='*60}")
    print(f"[1] VCF Parser ({n_variants} variants, avg of {N} runs)")
    print(f"    Python:  {py_time*1000:8.2f} ms")
    print(f"    Rust:    {rs_time*1000:8.2f} ms")
    print(f"    Speedup: {speedup:8.1f}x")
    print()
    return speedup


def benchmark_categorizer():
    """Compare categorization speed using Rust bitmask ops vs Python itertools.product."""
    from cwas.core.categorization.categorizer import Categorizer
    from cwas.core.categorization.parser import _parse_annotated_vcf_python

    df = _parse_annotated_vcf_python(TEST_VCF)

    # Build a minimal category_domain and gene_matrix for the test data
    # Extract annotation field names from VCF header
    annot_terms = _get_annot_terms()
    category_domain = _build_category_domain(annot_terms)
    gene_matrix = _build_gene_matrix(df)

    categorizer = Categorizer(category_domain, gene_matrix, "MPC", 2.0)

    # Warmup
    categorizer._categorize_variant_python(df)

    N = 20

    # Python
    start = time.perf_counter()
    for _ in range(N):
        result_py = categorizer._categorize_variant_python(df)
    py_time = (time.perf_counter() - start) / N

    # Rust (if available)
    try:
        categorizer._categorize_variant_rust(df)  # warmup
        start = time.perf_counter()
        for _ in range(N):
            result_rs = categorizer._categorize_variant_rust(df)
        rs_time = (time.perf_counter() - start) / N

        speedup = py_time / rs_time
        print(f"[2] Categorizer ({len(df)} variants, {len(result_py)} categories, avg of {N} runs)")
        print(f"    Python:  {py_time*1000:8.2f} ms")
        print(f"    Rust:    {rs_time*1000:8.2f} ms")
        print(f"    Speedup: {speedup:8.1f}x")

        # Verify correctness: Rust results should be a superset of Python
        # (Rust produces all possible categories including zero counts; Python only non-zero)
        mismatch = 0
        for key, val in result_py.items():
            if key in result_rs:
                if result_rs[key] != val:
                    mismatch += 1
            else:
                mismatch += 1
        if mismatch == 0:
            print(f"    Correctness: MATCH (all {len(result_py)} Python categories found in Rust)")
        else:
            print(f"    Correctness: {mismatch} MISMATCHES out of {len(result_py)}")
    except Exception as e:
        speedup = 1.0
        print(f"[2] Categorizer: Rust failed ({e}), skipping comparison")
        print(f"    Python:  {py_time*1000:8.2f} ms")

    print()
    return speedup


def benchmark_intersection():
    """Compare intersection matrix computation speed."""
    import pandas as pd
    from cwas.core.categorization.categorizer import Categorizer
    from cwas.core.categorization.parser import _parse_annotated_vcf_python

    df = _parse_annotated_vcf_python(TEST_VCF)
    annot_terms = _get_annot_terms()
    category_domain = _build_category_domain(annot_terms)
    gene_matrix = _build_gene_matrix(df)

    categorizer = Categorizer(category_domain, gene_matrix, "MPC", 2.0)

    N = 5

    # Python (dict → DataFrame)
    start = time.perf_counter()
    for _ in range(N):
        result_py = categorizer._get_intersection_python(df)
    py_time = (time.perf_counter() - start) / N

    # Rust via dict (old path)
    try:
        categorizer._get_intersection_rust(df)  # warmup
        start = time.perf_counter()
        for _ in range(N):
            result_rs = categorizer._get_intersection_rust(df)
        rs_time = (time.perf_counter() - start) / N

        speedup_dict = py_time / rs_time
        n_cats = len(result_py)
        print(f"[3a] Intersection Matrix via dict ({len(df)} variants, {n_cats} categories, avg of {N} runs)")
        print(f"    Python:       {py_time*1000:8.2f} ms")
        print(f"    Rust (dict):  {rs_time*1000:8.2f} ms")
        print(f"    Speedup:      {speedup_dict:8.1f}x")
    except Exception as e:
        speedup_dict = 1.0
        rs_time = py_time
        print(f"[3a] Intersection Matrix via dict: Rust failed ({e})")
        print(f"    Python:  {py_time*1000:8.2f} ms")

    # Rust via direct DataFrame (new path, no dict)
    try:
        # Build categories list from Python result for alignment
        all_cats = sorted(set(
            list(result_py.keys()) +
            [k2 for v in result_py.values() for k2 in v.keys()]
        ))
        categories = pd.Index(all_cats)

        categorizer._get_intersection_rust_df(df, categories)  # warmup
        start = time.perf_counter()
        for _ in range(N):
            result_df = categorizer._get_intersection_rust_df(df, categories)
        df_time = (time.perf_counter() - start) / N

        speedup_df = py_time / df_time
        print(f"[3b] Intersection Matrix direct DataFrame ({len(df)} variants, avg of {N} runs)")
        print(f"    Python:         {py_time*1000:8.2f} ms")
        print(f"    Rust (dict):    {rs_time*1000:8.2f} ms")
        print(f"    Rust (direct):  {df_time*1000:8.2f} ms")
        print(f"    Speedup vs Py:  {speedup_df:8.1f}x")
        print(f"    Dict overhead:  {(rs_time - df_time)/rs_time*100:5.1f}% eliminated")
        speedup = speedup_df
    except Exception as e:
        speedup = speedup_dict
        print(f"[3b] Intersection direct DataFrame: failed ({e})")

    print()
    return speedup


def benchmark_permutation():
    """Compare string-based vs boolean-based permutation test."""
    total_cnt = 5000
    case_cnt = 2000
    n_categories = 500

    np.random.seed(42)
    var_counts = np.random.poisson(2, size=(total_cnt, n_categories)).astype(np.float64)

    N = 20

    # String approach (original)
    start = time.perf_counter()
    for seed in range(N):
        np.random.seed(seed + 10001)
        swap_labels = np.full(total_cnt, 'ctrl')
        idx = np.random.choice(total_cnt, case_cnt, replace=False)
        for k in idx:
            swap_labels[k] = 'case'
        are_case = swap_labels == 'case'
        n1 = var_counts[are_case, :].sum(axis=0)
        n2 = var_counts[~are_case, :].sum(axis=0)
        _ = (n1 / case_cnt) / (n2 / (total_cnt - case_cnt))
    str_time = (time.perf_counter() - start) / N

    # Boolean approach (optimized)
    start = time.perf_counter()
    for seed in range(N):
        np.random.seed(seed + 10001)
        are_case = np.zeros(total_cnt, dtype=bool)
        idx = np.random.choice(total_cnt, case_cnt, replace=False)
        are_case[idx] = True
        n1 = var_counts[are_case, :].sum(axis=0)
        n2 = var_counts[~are_case, :].sum(axis=0)
        _ = (n1 / case_cnt) / (n2 / (total_cnt - case_cnt))
    bool_time = (time.perf_counter() - start) / N

    speedup = str_time / bool_time
    print(f"[4] Permutation Test ({total_cnt} samples, {n_categories} categories, avg of {N} runs)")
    print(f"    String (original): {str_time*1000:8.2f} ms/perm")
    print(f"    Boolean (optimized): {bool_time*1000:8.2f} ms/perm")
    print(f"    Speedup: {speedup:8.1f}x")
    print()
    return speedup


def benchmark_categorizer_scaled():
    """Simulate larger dataset by repeating test data."""
    from cwas_core import categorize_variants, build_category_names
    from itertools import product as itertools_product
    from collections import defaultdict

    # Simulate: 5 groups with realistic sizes
    # variant_type: 3, gene_set: 20, functional_score: 5, gencode: 15, functional_annotation: 10
    # Total categories: 3*20*5*15*10 = 45,000
    group_sizes_small = [3, 5, 3, 4, 3]  # 540 categories (manageable for Python)
    n_variants = 5000

    np.random.seed(42)
    # Build random index lists: each variant gets 1-3 random indices per group
    annotations = []
    for g in range(5):
        group_annots = []
        for v in range(n_variants):
            n_active = np.random.randint(1, min(4, group_sizes_small[g] + 1))
            indices = sorted(np.random.choice(group_sizes_small[g], n_active, replace=False).tolist())
            group_annots.append(indices)
        annotations.append(group_annots)

    sample_ids = np.zeros(n_variants, dtype=np.uintp)

    # Build term lists
    group_terms = []
    for g, size in enumerate(group_sizes_small):
        group_terms.append([f"G{g}T{i}" for i in range(size)])

    N = 10

    # Rust
    categorize_variants(annotations, group_sizes_small, sample_ids, 1)  # warmup
    start = time.perf_counter()
    for _ in range(N):
        result_rs = categorize_variants(annotations, group_sizes_small, sample_ids, 1)
    rs_time = (time.perf_counter() - start) / N

    # Python equivalent
    def python_categorize():
        result = defaultdict(int)
        for v in range(n_variants):
            term_lists = []
            for g in range(5):
                terms = [group_terms[g][i] for i in annotations[g][v]]
                term_lists.append(terms)
            for combo in itertools_product(*term_lists):
                result["_".join(combo)] += 1
        return result

    python_categorize()  # warmup
    start = time.perf_counter()
    for _ in range(N):
        result_py = python_categorize()
    py_time = (time.perf_counter() - start) / N

    n_cats = int(np.prod(group_sizes_small))
    avg_combos = np.mean([np.prod([len(annotations[g][v]) for g in range(5)]) for v in range(min(100, n_variants))])
    speedup = py_time / rs_time

    print(f"[5] Scaled Categorizer ({n_variants} variants, {n_cats} possible categories)")
    print(f"    Avg combos/variant: ~{avg_combos:.0f}")
    print(f"    Python:  {py_time*1000:8.2f} ms")
    print(f"    Rust:    {rs_time*1000:8.2f} ms")
    print(f"    Speedup: {speedup:8.1f}x")
    print()
    return speedup


def _get_annot_terms():
    """Extract ANNOT field names from VCF header."""
    import gzip
    with gzip.open(TEST_VCF, 'rt') as f:
        for line in f:
            if line.startswith("##INFO=<ID=ANNOT"):
                import re
                m = re.search(r'Key=([^"]+)', line)
                if m:
                    return m.group(1).rstrip('">\n').split('|')
    return []


def _build_category_domain(annot_terms):
    """Build a minimal category_domain dict for testing."""
    return {
        'variant_type': ['SNV', 'Indel', 'All'],
        'gene_set': ['Any'],
        'functional_score': ['All'] + annot_terms[:2],  # first 2 as functional_score
        'gencode': ['Any', 'CodingRegion', 'NoncodingRegion', 'PromoterRegion',
                     'IntronRegion', 'IntergenicRegion'],
        'functional_annotation': ['Any'] + annot_terms[2:],  # rest as functional_annotation
    }


def _build_gene_matrix(df):
    """Build a minimal gene_matrix for testing, keyed by gene ID."""
    gene_matrix = {}
    if 'Gene' in df.columns:
        for gene in df['Gene'].unique():
            gene_matrix[gene] = {'ProteinCoding'}
    if 'NEAREST' in df.columns:
        for gene in df['NEAREST'].unique():
            if gene not in gene_matrix:
                gene_matrix[gene] = set()
    return gene_matrix


if __name__ == "__main__":
    print()
    print("=" * 60)
    print("  CWAS-Plus Rust Acceleration Benchmark")
    print("=" * 60)
    print()

    speedups = {}
    speedups['VCF Parser'] = benchmark_vcf_parser()
    speedups['Categorizer (test VCF)'] = benchmark_categorizer()
    speedups['Intersection Matrix'] = benchmark_intersection()
    speedups['Permutation Test'] = benchmark_permutation()
    speedups['Categorizer (5k scaled)'] = benchmark_categorizer_scaled()

    print("=" * 60)
    print("  Summary")
    print("=" * 60)
    for name, spd in speedups.items():
        bar = "#" * min(int(spd), 80)
        print(f"  {name:30s} {spd:6.1f}x  {bar}")
    print()
    print("  Note: Test VCF has only 18 variants. Real-world speedups")
    print("  scale with data size (10k-1M+ variants).")
    print("=" * 60)
