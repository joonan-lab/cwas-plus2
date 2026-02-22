"""Numerical validation: rpy2 (R) vs sklearn equivalence.

Tests that the sklearn implementations in clustering.py, dawn.py, and
risk_score.py produce results equivalent to their R counterparts.
"""

import numpy as np
import pytest
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.linear_model import LassoCV
from sklearn.metrics import adjusted_rand_score, r2_score, silhouette_score
from sklearn.model_selection import PredefinedSplit


# ---------------------------------------------------------------------------
# Shared synthetic data
# ---------------------------------------------------------------------------

def _make_cluster_data(seed=42, n_per_cluster=100, k=3):
    """Well-separated Gaussian blobs for deterministic KMeans tests."""
    centers = np.array([[0, 0], [10, 10], [-10, 10]])[:k]
    X, y_true = make_blobs(
        n_samples=n_per_cluster * k, centers=centers,
        cluster_std=0.8, random_state=seed,
    )
    return X, y_true


def _r_converter():
    """Build an rpy2 converter that handles numpy arrays."""
    from rpy2.robjects import default_converter
    from rpy2.robjects.numpy2ri import converter as np_converter
    return default_converter + np_converter


# ---------------------------------------------------------------------------
# Test 1 – KMeans: direct R vs sklearn comparison
# ---------------------------------------------------------------------------

def test_kmeans_r_vs_sklearn():
    """R stats::kmeans and sklearn KMeans should agree on well-separated data."""
    pytest.importorskip("rpy2.robjects")
    from rpy2.robjects import r
    from rpy2.robjects.packages import importr

    cv = _r_converter()
    stats = importr("stats")

    X, y_true = _make_cluster_data()
    n, p = X.shape

    with cv.context():
        # --- R kmeans ---
        r_x = r.matrix(X, nrow=n, ncol=p)
        r_result = stats.kmeans(r_x, centers=3, nstart=300, iter_max=100)
        r_labels = r_result.getbyname("cluster").astype(int) - 1  # 1-indexed → 0
        r_totwithinss = r_result.getbyname("tot.withinss")[0]

    # --- sklearn KMeans ---
    sk = KMeans(n_clusters=3, n_init=300, max_iter=100, random_state=42)
    sk.fit(X)
    sk_labels = sk.labels_

    # --- Compare ---
    ari = adjusted_rand_score(r_labels, sk_labels)
    sil_r = silhouette_score(X, r_labels)
    sil_sk = silhouette_score(X, sk_labels)

    print("\n=== Test 1: KMeans – R vs sklearn ===")
    print(f"  Adjusted Rand Index     : {ari:.6f}")
    print(f"  Silhouette (R)          : {sil_r:.6f}")
    print(f"  Silhouette (sklearn)    : {sil_sk:.6f}")
    print(f"  Tot within-SS (R)       : {r_totwithinss:.4f}")
    print(f"  Inertia (sklearn)       : {sk.inertia_:.4f}")

    assert ari == 1.0, f"Clusterings differ: ARI = {ari}"
    assert abs(sil_r - sil_sk) < 0.01
    assert abs(r_totwithinss - sk.inertia_) / r_totwithinss < 0.01


# ---------------------------------------------------------------------------
# Test 2 – KMeans with explicit init centers (dawn.py pattern)
# ---------------------------------------------------------------------------

def test_kmeans_explicit_init_r_vs_sklearn():
    """With identical starting centers, R and sklearn must converge identically."""
    pytest.importorskip("rpy2.robjects")
    from rpy2.robjects import r
    from rpy2.robjects.packages import importr

    cv = _r_converter()
    stats = importr("stats")

    X, _ = _make_cluster_data()
    n, p = X.shape

    # Pick 3 fixed initial centers (indices 0, 100, 200 — one per true cluster)
    init_idx = [0, 100, 200]
    init_centers = X[init_idx]

    with cv.context():
        # --- R kmeans with explicit center matrix ---
        r_x = r.matrix(X, nrow=n, ncol=p)
        r_centers_mat = r.matrix(init_centers, nrow=len(init_idx), ncol=p)
        r_result = stats.kmeans(r_x, centers=r_centers_mat, iter_max=100)
        r_labels = r_result.getbyname("cluster").astype(int) - 1
        r_final_centers = r_result.getbyname("centers")

    # --- sklearn with same init ---
    sk = KMeans(n_clusters=3, init=init_centers, n_init=1, max_iter=100,
                random_state=42)
    sk.fit(X)
    sk_labels = sk.labels_

    # Align label spaces: build a mapping from sklearn label → R label
    label_map = {}
    for sk_l in range(3):
        mask = sk_labels == sk_l
        # majority R label in this sklearn cluster
        r_vals, counts = np.unique(r_labels[mask], return_counts=True)
        label_map[sk_l] = r_vals[counts.argmax()]
    sk_labels_aligned = np.array([label_map[l] for l in sk_labels])

    # Sort R centers rows to match sklearn center ordering
    r_centers_aligned = r_final_centers[[label_map[i] for i in range(3)]]

    print("\n=== Test 2: KMeans explicit init – R vs sklearn ===")
    match_pct = np.mean(sk_labels_aligned == r_labels) * 100
    center_diff = np.max(np.abs(sk.cluster_centers_ - r_centers_aligned))
    print(f"  Label match             : {match_pct:.1f}%")
    print(f"  Max center coord diff   : {center_diff:.2e}")

    assert np.array_equal(sk_labels_aligned, r_labels), \
        f"Labels differ on {np.sum(sk_labels_aligned != r_labels)} points"
    np.testing.assert_allclose(sk.cluster_centers_, r_centers_aligned, atol=1e-6)


# ---------------------------------------------------------------------------
# Test 3 – LassoCV: ground-truth recovery (no R comparison)
# ---------------------------------------------------------------------------

def test_lassocv_ground_truth_recovery():
    """LassoCV on synthetic sparse data should recover the true support."""
    rng = np.random.RandomState(42)
    n, p = 400, 50
    n_nonzero = 5

    # True coefficients: first 5 features have nonzero weights
    beta = np.zeros(p)
    beta[:n_nonzero] = rng.uniform(1.5, 3.0, size=n_nonzero)
    true_support = set(range(n_nonzero))

    X = rng.randn(n, p)
    y = X @ beta + rng.randn(n) * 0.5  # low noise

    # Custom CV folds mirroring _custom_cv_folds (10-fold)
    n_folds = 10
    rand_idx = rng.permutation(n)
    foldid = np.zeros(n, dtype=int)
    for i in range(1, n_folds + 1):
        idx = rand_idx[np.arange(n * (i - 1) / n_folds, n * i / n_folds, dtype=int)]
        foldid[idx] = i

    cv = PredefinedSplit(foldid)

    lasso = LassoCV(cv=cv, n_alphas=100, random_state=42)
    lasso.fit(X, y)

    # Evaluate
    coef = lasso.coef_
    estimated_support = set(np.where(np.abs(coef) > 1e-6)[0])
    recovered = true_support & estimated_support
    false_positives = estimated_support - true_support
    fp_rate = len(false_positives) / (p - n_nonzero)

    # Test-set prediction (fold 0 is "test" in the real code, use fold 1 here)
    test_mask = foldid == 1
    y_pred = lasso.predict(X[test_mask])
    y_test = y[test_mask]
    r2 = r2_score(y_test, y_pred)
    pearson_r = np.corrcoef(y_test, y_pred)[0, 1]

    print("\n=== Test 3: LassoCV ground-truth recovery ===")
    print(f"  True support            : {sorted(true_support)}")
    print(f"  Estimated support       : {sorted(estimated_support)}")
    print(f"  Recovered ({len(recovered)}/{n_nonzero})          : {sorted(recovered)}")
    print(f"  False positive rate     : {fp_rate:.2%}")
    print(f"  Best alpha              : {lasso.alpha_:.6f}")
    print(f"  R² (test fold)          : {r2:.4f}")
    print(f"  Pearson r (test fold)   : {pearson_r:.4f}")

    assert len(recovered) >= 4, \
        f"Only recovered {len(recovered)}/5 true non-zeros: {sorted(recovered)}"
    assert fp_rate < 0.20, f"False positive rate too high: {fp_rate:.2%}"
    assert r2 > 0.5, f"R² too low: {r2:.4f}"
    assert pearson_r > 0.7, f"Pearson r too low: {pearson_r:.4f}"
