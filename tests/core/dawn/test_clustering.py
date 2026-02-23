"""
Test cwas.core.dawn.clustering — kmeans_cluster class.
"""
import numpy as np
import pandas as pd
import pytest
import os
import tempfile

from cwas.core.dawn.clustering import kmeans_cluster


def test_import():
    from cwas.core.dawn.clustering import kmeans_cluster  # noqa: F401


def test_kmeans_cluster_properties():
    df = pd.DataFrame({"t-SNE1": [1.0, 2.0, 3.0], "t-SNE2": [4.0, 5.0, 6.0]})
    kc = kmeans_cluster(df, seed=42)
    assert kc.seed == 42
    assert kc.tsne_out is df


# --- Synthetic data helpers ---

def _make_blobs(n_per_cluster=50, k=3, seed=42):
    """Create well-separated 2D blobs for testing."""
    rng = np.random.RandomState(seed)
    centers = np.array([[i * 10, 0] for i in range(k)])
    points = []
    for c in centers:
        points.append(rng.randn(n_per_cluster, 2) * 0.5 + c)
    data = np.vstack(points)
    return pd.DataFrame(data, columns=["t-SNE1", "t-SNE2"])


# --- Parsimonious K selection tests ---

class TestParsimoniousK:
    """Tests for parsimonious K selection (smallest K in plateau)."""

    def _make_kc(self):
        df = _make_blobs(n_per_cluster=20, k=3, seed=0)
        return kmeans_cluster(df, seed=42)

    def test_parsimonious_false_returns_argmax(self):
        """With parsimonious=False, optimal_k returns the silhouette argmax."""
        kc = self._make_kc()
        with tempfile.TemporaryDirectory() as tmpdir:
            out = os.path.join(tmpdir, "sil_plot.pdf")
            k = kc.optimal_k("2,5", out, parsimonious=False)
            assert 2 <= k <= 5

    def test_parsimonious_true_returns_valid_k(self):
        """With parsimonious=True, optimal_k returns a K within the range."""
        kc = self._make_kc()
        with tempfile.TemporaryDirectory() as tmpdir:
            out = os.path.join(tmpdir, "sil_plot.pdf")
            k = kc.optimal_k("2,5", out, parsimonious=True)
            assert 2 <= k <= 5

    def test_parsimonious_selects_smallest_in_plateau(self):
        """Parsimonious mode should select K <= silhouette-argmax K."""
        kc = self._make_kc()
        with tempfile.TemporaryDirectory() as tmpdir:
            kc2 = kmeans_cluster(kc.tsne_out.copy(), seed=42)
            k_argmax = kc.optimal_k("2,5", os.path.join(tmpdir, "p1.pdf"), parsimonious=False)
            k_pars = kc2.optimal_k("2,5", os.path.join(tmpdir, "p2.pdf"), parsimonious=True)
            assert k_pars <= k_argmax

    def test_parsimonious_reproducible(self):
        """Same seed should produce the same parsimonious K."""
        df = _make_blobs(n_per_cluster=30, k=3, seed=0)
        with tempfile.TemporaryDirectory() as tmpdir:
            kc1 = kmeans_cluster(df.copy(), seed=42)
            k1 = kc1.optimal_k("2,5", os.path.join(tmpdir, "p1.pdf"), parsimonious=True)
            kc2 = kmeans_cluster(df.copy(), seed=42)
            k2 = kc2.optimal_k("2,5", os.path.join(tmpdir, "p2.pdf"), parsimonious=True)
            assert k1 == k2

    def test_parsimonious_flat_silhouette(self):
        """When all silhouette scores are equal, parsimonious picks smallest K."""
        kc = self._make_kc()
        flat_val = 0.5
        kc._avg_sil = lambda k: flat_val
        with tempfile.TemporaryDirectory() as tmpdir:
            k = kc.optimal_k("2,5", os.path.join(tmpdir, "p.pdf"), parsimonious=True)
            assert k == 2

    def test_parsimonious_single_k(self):
        """Range with a single K value should return that K."""
        kc = self._make_kc()
        with tempfile.TemporaryDirectory() as tmpdir:
            k = kc.optimal_k("3,3", os.path.join(tmpdir, "p.pdf"), parsimonious=True)
            assert k == 3


class TestOptimalKValidation:
    """Tests for k_range input validation."""

    def test_start_greater_than_end_raises(self):
        df = _make_blobs(n_per_cluster=20, k=3, seed=0)
        kc = kmeans_cluster(df, seed=42)
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="Invalid k_range"):
                kc.optimal_k("10,5", os.path.join(tmpdir, "p.pdf"))

    def test_start_less_than_2_raises(self):
        df = _make_blobs(n_per_cluster=20, k=3, seed=0)
        kc = kmeans_cluster(df, seed=42)
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="Invalid k_range"):
                kc.optimal_k("1,5", os.path.join(tmpdir, "p.pdf"))
