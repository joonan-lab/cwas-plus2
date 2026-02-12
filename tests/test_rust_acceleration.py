"""
Tests to verify Rust acceleration produces identical results to Python fallback.
"""
import pathlib
import time

import numpy as np
import pandas as pd
import pytest

TEST_VCF = pathlib.Path(__file__).parent / "test_file" / "test_annotated.vcf.gz"


def _can_import_rust():
    try:
        import cwas_core
        return True
    except ImportError:
        return False


# ============================================================
# 1. VCF Parser: Rust vs Python
# ============================================================

class TestVCFParser:
    def test_python_parser(self):
        """Verify the Python VCF parser works on test data."""
        from cwas.core.categorization.parser import _parse_annotated_vcf_python
        df = _parse_annotated_vcf_python(TEST_VCF)
        assert len(df) > 0, "No variants parsed"
        assert "SAMPLE" in df.columns
        assert "ANNOT" in df.columns
        print(f"  Python parsed {len(df)} variants, {len(df.columns)} columns")

    @pytest.mark.skipif(
        not _can_import_rust(),
        reason="cwas_core not installed"
    )
    def test_rust_parser(self):
        """Verify the Rust VCF parser works on test data."""
        from cwas.core.categorization.parser import _parse_annotated_vcf_rust
        df = _parse_annotated_vcf_rust(TEST_VCF)
        assert len(df) > 0, "No variants parsed"
        print(f"  Rust parsed {len(df)} variants, {len(df.columns)} columns")

    @pytest.mark.skipif(
        not _can_import_rust(),
        reason="cwas_core not installed"
    )
    def test_parser_rust_vs_python_row_count(self):
        """Rust and Python parsers should parse the same number of rows."""
        from cwas.core.categorization.parser import (
            _parse_annotated_vcf_python,
            _parse_annotated_vcf_rust,
        )
        df_py = _parse_annotated_vcf_python(TEST_VCF)
        df_rs = _parse_annotated_vcf_rust(TEST_VCF)
        assert len(df_py) == len(df_rs), (
            f"Row count mismatch: Python={len(df_py)}, Rust={len(df_rs)}"
        )
        print(f"  Both parsed {len(df_py)} rows")

    @pytest.mark.skipif(
        not _can_import_rust(),
        reason="cwas_core not installed"
    )
    def test_parser_rust_vs_python_key_columns(self):
        """Key columns (CHROM, POS, REF, ALT, SAMPLE, ANNOT) should match."""
        from cwas.core.categorization.parser import (
            _parse_annotated_vcf_python,
            _parse_annotated_vcf_rust,
        )
        df_py = _parse_annotated_vcf_python(TEST_VCF)
        df_rs = _parse_annotated_vcf_rust(TEST_VCF)

        key_cols = ["CHROM", "POS", "REF", "ALT"]
        for col in key_cols:
            if col in df_py.columns and col in df_rs.columns:
                py_vals = df_py[col].tolist()
                rs_vals = df_rs[col].tolist()
                assert py_vals == rs_vals, f"Column {col} mismatch"
                print(f"  Column '{col}' matches")

        # SAMPLE is parsed from INFO in Python but should be present in both
        if "SAMPLE" in df_py.columns and "SAMPLE" in df_rs.columns:
            assert df_py["SAMPLE"].tolist() == df_rs["SAMPLE"].tolist()
            print(f"  Column 'SAMPLE' matches")


# ============================================================
# 2. Categorizer: Rust functions unit tests
# ============================================================

class TestRustCategorizer:
    @pytest.mark.skipif(
        not _can_import_rust(),
        reason="cwas_core not installed"
    )
    def test_categorize_single_variant(self):
        """Single variant with known bitmasks should produce correct counts."""
        from cwas_core import categorize_variants, build_category_names

        # One variant with bit 0 set in all groups → only category [0][0][0][0][0]
        annotations = np.array([[1, 1, 1, 1, 1]], dtype=np.uint64)
        group_sizes = np.array([3, 2, 2, 2, 2], dtype=np.uintp)
        sample_ids = np.array([0], dtype=np.uintp)

        result = categorize_variants(annotations, group_sizes, sample_ids, 1)
        names = build_category_names([
            ['A', 'B', 'C'], ['X', 'Y'], ['P', 'Q'], ['M', 'N'], ['S', 'T']
        ])

        # Only one category should be non-zero
        assert result[0, 0] == 1  # A_X_P_M_S
        assert np.sum(result) == 1
        assert names[0] == 'A_X_P_M_S'
        print("  Single variant categorization correct")

    @pytest.mark.skipif(
        not _can_import_rust(),
        reason="cwas_core not installed"
    )
    def test_categorize_multi_bit(self):
        """Variant with multiple bits set should generate product combinations."""
        from cwas_core import categorize_variants

        # bits 0,1 set in group 0 → 2 terms; bit 0 in others → 1 term each
        # Total combos: 2*1*1*1*1 = 2
        annotations = np.array([[0b11, 1, 1, 1, 1]], dtype=np.uint64)
        group_sizes = np.array([3, 2, 2, 2, 2], dtype=np.uintp)
        sample_ids = np.array([0], dtype=np.uintp)

        result = categorize_variants(annotations, group_sizes, sample_ids, 1)
        assert np.sum(result) == 2
        print("  Multi-bit categorization correct")

    @pytest.mark.skipif(
        not _can_import_rust(),
        reason="cwas_core not installed"
    )
    def test_categorize_multi_sample(self):
        """Multiple samples should accumulate independently."""
        from cwas_core import categorize_variants

        annotations = np.array([
            [1, 1, 1, 1, 1],  # sample 0
            [1, 1, 1, 1, 1],  # sample 1
            [1, 1, 1, 1, 1],  # sample 0 again
        ], dtype=np.uint64)
        group_sizes = np.array([3, 2, 2, 2, 2], dtype=np.uintp)
        sample_ids = np.array([0, 1, 0], dtype=np.uintp)

        result = categorize_variants(annotations, group_sizes, sample_ids, 2)
        assert result.shape == (2, 48)  # 3*2*2*2*2 = 48
        assert result[0, 0] == 2  # sample 0 got 2 variants
        assert result[1, 0] == 1  # sample 1 got 1 variant
        print("  Multi-sample categorization correct")


# ============================================================
# 3. Intersection Matrix: Rust unit tests
# ============================================================

class TestRustIntersection:
    @pytest.mark.skipif(
        not _can_import_rust(),
        reason="cwas_core not installed"
    )
    def test_intersection_symmetric(self):
        """Intersection matrix should be symmetric."""
        from cwas_core import compute_intersection_matrix

        annotations = np.array([
            [0b11, 0b01, 0b01, 0b01, 0b01],
            [0b10, 0b10, 0b10, 0b10, 0b10],
        ], dtype=np.uint64)
        group_sizes = np.array([3, 2, 2, 2, 2], dtype=np.uintp)

        matrix = compute_intersection_matrix(annotations, group_sizes)
        assert np.allclose(matrix, matrix.T), "Matrix not symmetric"
        print("  Intersection matrix is symmetric")

    @pytest.mark.skipif(
        not _can_import_rust(),
        reason="cwas_core not installed"
    )
    def test_intersection_diagonal(self):
        """Diagonal should equal category counts (self-intersection)."""
        from cwas_core import categorize_variants, compute_intersection_matrix

        annotations = np.array([
            [0b101, 0b01, 0b01, 0b01, 0b01],
            [0b010, 0b10, 0b10, 0b10, 0b10],
            [0b111, 0b11, 0b11, 0b11, 0b11],
        ], dtype=np.uint64)
        group_sizes = np.array([3, 2, 2, 2, 2], dtype=np.uintp)
        sample_ids = np.array([0, 0, 0], dtype=np.uintp)

        counts = categorize_variants(annotations, group_sizes, sample_ids, 1)[0]
        matrix = compute_intersection_matrix(annotations, group_sizes)

        for i in range(len(counts)):
            assert matrix[i, i] == counts[i], (
                f"Diagonal mismatch at {i}: matrix={matrix[i,i]}, count={counts[i]}"
            )
        print("  Diagonal matches category counts")


# ============================================================
# 3b. Sparse Intersection Matrix: Rust unit tests
# ============================================================

class TestRustIntersectionSparse:
    @pytest.mark.skipif(
        not _can_import_rust(),
        reason="cwas_core not installed"
    )
    def test_sparse_matches_dense(self):
        """Sparse intersection result should match the dense version exactly."""
        from cwas_core import compute_intersection_matrix, compute_intersection_matrix_sparse
        from scipy import sparse

        annotations = np.array([
            [0b11, 0b01, 0b01, 0b01, 0b01],
            [0b10, 0b10, 0b10, 0b10, 0b10],
            [0b111, 0b11, 0b11, 0b11, 0b11],
        ], dtype=np.uint64)
        group_sizes = np.array([3, 2, 2, 2, 2], dtype=np.uintp)

        # Dense reference
        dense = compute_intersection_matrix(annotations, group_sizes)

        # Sparse result → reconstruct full matrix
        coo_dict = compute_intersection_matrix_sparse(annotations, group_sizes)
        row = coo_dict['row']
        col = coo_dict['col']
        data = coo_dict['data']
        shape = coo_dict['shape']

        upper = sparse.coo_matrix((data, (row, col)), shape=shape)
        upper_csr = upper.tocsr()
        diag = sparse.diags(upper_csr.diagonal())
        full = (upper_csr + upper_csr.T - diag).toarray()

        assert np.allclose(dense, full), (
            f"Sparse vs dense mismatch. Max diff: {np.max(np.abs(dense - full))}"
        )
        print("  Sparse matches dense exactly")

    @pytest.mark.skipif(
        not _can_import_rust(),
        reason="cwas_core not installed"
    )
    def test_sparse_symmetric(self):
        """Reconstructed sparse matrix should be symmetric."""
        from cwas_core import compute_intersection_matrix_sparse
        from scipy import sparse

        annotations = np.array([
            [0b101, 0b01, 0b01, 0b01, 0b01],
            [0b010, 0b10, 0b10, 0b10, 0b10],
            [0b111, 0b11, 0b11, 0b11, 0b11],
        ], dtype=np.uint64)
        group_sizes = np.array([3, 2, 2, 2, 2], dtype=np.uintp)

        coo_dict = compute_intersection_matrix_sparse(annotations, group_sizes)
        row = coo_dict['row']
        col = coo_dict['col']
        data = coo_dict['data']
        shape = coo_dict['shape']

        upper = sparse.coo_matrix((data, (row, col)), shape=shape)
        upper_csr = upper.tocsr()
        diag = sparse.diags(upper_csr.diagonal())
        full = (upper_csr + upper_csr.T - diag).toarray()

        assert np.allclose(full, full.T), "Reconstructed sparse matrix not symmetric"
        print("  Sparse matrix is symmetric")

    @pytest.mark.skipif(
        not _can_import_rust(),
        reason="cwas_core not installed"
    )
    def test_sparse_empty_input(self):
        """Empty input should produce an empty sparse result."""
        from cwas_core import compute_intersection_matrix_sparse

        annotations = np.zeros((0, 5), dtype=np.uint64)
        group_sizes = np.array([3, 2, 2, 2, 2], dtype=np.uintp)

        coo_dict = compute_intersection_matrix_sparse(annotations, group_sizes)
        assert len(coo_dict['row']) == 0
        assert len(coo_dict['col']) == 0
        assert len(coo_dict['data']) == 0
        assert coo_dict['shape'] == (48, 48)  # 3*2*2*2*2
        print("  Empty input handled correctly")

    @pytest.mark.skipif(
        not _can_import_rust(),
        reason="cwas_core not installed"
    )
    def test_sparse_diagonal_matches_counts(self):
        """Diagonal of sparse matrix should equal category counts."""
        from cwas_core import (
            categorize_variants,
            compute_intersection_matrix_sparse,
        )
        from scipy import sparse

        annotations = np.array([
            [0b101, 0b01, 0b01, 0b01, 0b01],
            [0b010, 0b10, 0b10, 0b10, 0b10],
            [0b111, 0b11, 0b11, 0b11, 0b11],
        ], dtype=np.uint64)
        group_sizes = np.array([3, 2, 2, 2, 2], dtype=np.uintp)
        sample_ids = np.array([0, 0, 0], dtype=np.uintp)

        counts = categorize_variants(annotations, group_sizes, sample_ids, 1)[0]

        coo_dict = compute_intersection_matrix_sparse(annotations, group_sizes)
        upper = sparse.coo_matrix(
            (coo_dict['data'], (coo_dict['row'], coo_dict['col'])),
            shape=coo_dict['shape'],
        )
        diag_vals = upper.tocsr().diagonal()

        for i in range(len(counts)):
            assert diag_vals[i] == counts[i], (
                f"Diagonal mismatch at {i}: sparse={diag_vals[i]}, count={counts[i]}"
            )
        print("  Sparse diagonal matches category counts")


# ============================================================
# 4. Permutation Test: Boolean array optimization
# ============================================================

class TestPermutationOptimization:
    def test_boolean_permutation(self):
        """Verify the boolean-based permutation produces valid results."""
        total_cnt = 100
        case_cnt = 40

        np.random.seed(42)
        are_case = np.zeros(total_cnt, dtype=bool)
        idx = np.random.choice(total_cnt, case_cnt, replace=False)
        are_case[idx] = True

        assert np.sum(are_case) == case_cnt
        assert np.sum(~are_case) == total_cnt - case_cnt
        print(f"  Boolean permutation: {case_cnt} cases, {total_cnt - case_cnt} controls")

    def test_boolean_vs_string_equivalence(self):
        """Boolean array approach should give same results as string approach."""
        total_cnt = 50
        case_cnt = 20

        # String approach (original)
        np.random.seed(12345)
        swap_labels = np.full(total_cnt, 'ctrl')
        idx_str = np.random.choice(total_cnt, case_cnt, replace=False)
        for k in idx_str:
            swap_labels[k] = 'case'
        are_case_str = swap_labels == 'case'

        # Boolean approach (optimized)
        np.random.seed(12345)
        are_case_bool = np.zeros(total_cnt, dtype=bool)
        idx_bool = np.random.choice(total_cnt, case_cnt, replace=False)
        are_case_bool[idx_bool] = True

        assert np.array_equal(are_case_str, are_case_bool), (
            "Boolean and string approaches produce different results"
        )
        print("  Boolean and string approaches are equivalent")


# ============================================================
# 5. Performance comparison (informational)
# ============================================================

class TestPerformance:
    @pytest.mark.skipif(
        not _can_import_rust(),
        reason="cwas_core not installed"
    )
    def test_categorizer_performance(self):
        """Compare Rust vs Python categorization speed."""
        from cwas_core import categorize_variants

        # Generate synthetic data: 10k variants, small group sizes
        n_variants = 10000
        np.random.seed(42)
        annotations = np.random.randint(1, 8, size=(n_variants, 5)).astype(np.uint64)
        group_sizes = np.array([3, 3, 3, 3, 3], dtype=np.uintp)
        sample_ids = np.zeros(n_variants, dtype=np.uintp)

        # Rust timing
        start = time.time()
        for _ in range(10):
            categorize_variants(annotations, group_sizes, sample_ids, 1)
        rust_time = (time.time() - start) / 10

        print(f"\n  Rust categorization: {rust_time*1000:.1f}ms for {n_variants} variants")
        print(f"  ({n_variants / rust_time:.0f} variants/sec)")

    def test_permutation_performance(self):
        """Compare boolean vs string permutation speed."""
        total_cnt = 5000
        case_cnt = 2000
        n_iters = 100

        # String approach
        start = time.time()
        for seed in range(n_iters):
            np.random.seed(seed)
            swap_labels = np.full(total_cnt, 'ctrl')
            idx = np.random.choice(total_cnt, case_cnt, replace=False)
            for k in idx:
                swap_labels[k] = 'case'
            _ = swap_labels == 'case'
        str_time = time.time() - start

        # Boolean approach
        start = time.time()
        for seed in range(n_iters):
            np.random.seed(seed)
            are_case = np.zeros(total_cnt, dtype=bool)
            idx = np.random.choice(total_cnt, case_cnt, replace=False)
            are_case[idx] = True
        bool_time = time.time() - start

        speedup = str_time / bool_time
        print(f"\n  String approach: {str_time*1000:.1f}ms ({n_iters} iters)")
        print(f"  Boolean approach: {bool_time*1000:.1f}ms ({n_iters} iters)")
        print(f"  Speedup: {speedup:.1f}x")


