"""
Test cwas.dawn

Note: Dawn.__init__ calls importr('stats').kmeans, which requires R.
Tests are skipped if R is unavailable.
"""
import argparse

import numpy as np
import pytest
from pathlib import Path


def _can_import_dawn():
    """Check if cwas.dawn can be imported (requires rpy2, igraph, scanpy)."""
    try:
        import cwas.dawn  # noqa: F401
        return True
    except Exception:
        return False


requires_dawn = pytest.mark.skipif(
    not _can_import_dawn(),
    reason="cwas.dawn dependencies not available (rpy2/igraph/scanpy)",
)


@requires_dawn
def test_import():
    """Module can be imported when rpy2 is available."""
    import cwas.dawn  # noqa: F401


@requires_dawn
class TestDawnInstance:
    """Tests that require R to instantiate Dawn."""

    @staticmethod
    def _make_args(**overrides):
        defaults = dict(
            num_proc=1,
            eig_vector_file=Path("/tmp/test.eig_vecs.zarr"),
            corr_mat_file=Path("/tmp/test.correlation_matrix.zarr"),
            permut_test_file=Path("/tmp/test.permutation_test.txt.gz"),
            category_count_file=Path("/tmp/catcount.txt"),
            output_dir_path=Path("/tmp/output"),
            input_dir_path=Path("/tmp"),
            leiden_clustering=None,
            lambda_val=0.5,
            k_range="2:10",
            k_val=5,
            seed=42,
            resolution=1.0,
            tsne_method="barnes_hut",
            tag="test",
            count_threshold=5,
            corr_threshold=0.3,
            size_threshold=10,
        )
        defaults.update(overrides)
        return argparse.Namespace(**defaults)

    @staticmethod
    def _make_inst(**overrides):
        from cwas.dawn import Dawn

        class DawnMock(Dawn):
            @staticmethod
            def _print_args(args):
                pass

            @staticmethod
            def _check_args_validity(args):
                pass

        return DawnMock(TestDawnInstance._make_args(**overrides))

    # --- Simple property tests ---

    def test_num_proc(self):
        inst = self._make_inst(num_proc=4)
        assert inst.num_proc == 4

    def test_lambda_val(self):
        inst = self._make_inst(lambda_val=0.7)
        assert inst.lambda_val == 0.7

    def test_seed(self):
        inst = self._make_inst(seed=99)
        assert inst.seed == 99

    def test_tag(self):
        inst = self._make_inst(tag="dawn_v1")
        assert inst.tag == "dawn_v1"

    def test_resolution(self):
        inst = self._make_inst(resolution=0.5)
        assert inst.resolution == 0.5

    def test_count_threshold(self):
        inst = self._make_inst(count_threshold=10)
        assert inst.count_threshold == 10

    def test_corr_threshold(self):
        inst = self._make_inst(corr_threshold=0.5)
        assert inst.corr_threshold == 0.5

    def test_size_threshold(self):
        inst = self._make_inst(size_threshold=20)
        assert inst.size_threshold == 20

    def test_leiden_clustering_none(self):
        inst = self._make_inst(leiden_clustering=None)
        assert inst.leiden_clustering is None

    def test_leiden_clustering_set(self):
        inst = self._make_inst(leiden_clustering="eigen_vector")
        assert inst.leiden_clustering == "eigen_vector"

    def test_tsne_method(self):
        inst = self._make_inst(tsne_method="exact")
        assert inst.tsne_method == "exact"

    def test_k_val(self):
        inst = self._make_inst(k_val=8)
        assert inst.k_val == 8

    def test_k_range(self):
        inst = self._make_inst(k_range="3:15")
        assert inst.k_range == "3:15"
