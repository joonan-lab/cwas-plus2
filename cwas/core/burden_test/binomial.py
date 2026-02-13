import numpy as np
from scipy.stats import binom, binomtest


def binom_two_tail(n1: int, n2: int, p: float):
    n1 = int(n1)
    n2 = int(n2)
    if n1 + n2 == 0:
        return 1.0
    return binomtest(k=n1, n=n1 + n2, p=p, alternative="two-sided").pvalue


def binom_one_tail(n1: int, n2: int, p: float):
    n1 = int(n1)
    n2 = int(n2)
    if n1 + n2 == 0:
        return 1.0
    return binomtest(k=n1, n=n1 + n2, p=p, alternative="greater").pvalue


def binom_one_tail_vectorized(n1: np.ndarray, n2: np.ndarray, p: float) -> np.ndarray:
    """Vectorized one-sided (greater) binomial test. P(X >= n1) = sf(n1-1, n, p)."""
    n1 = np.asarray(n1, dtype=np.int64)
    n2 = np.asarray(n2, dtype=np.int64)
    n = n1 + n2
    result = np.ones(len(n1), dtype=np.float64)
    mask = n > 0
    result[mask] = binom.sf(n1[mask] - 1, n[mask], p)
    return result


def binom_two_tail_vectorized(n1: np.ndarray, n2: np.ndarray, p: float) -> np.ndarray:
    """Vectorized two-sided binomial test using the minlike method (matches scipy binomtest)."""
    n1 = np.asarray(n1, dtype=np.int64)
    n2 = np.asarray(n2, dtype=np.int64)
    n = n1 + n2
    result = np.ones(len(n1), dtype=np.float64)
    mask = n > 0
    if not np.any(mask):
        return result

    k = n1[mask]
    n_m = n[mask]

    d = binom.pmf(k, n_m, p)
    mode = np.floor((n_m + 1) * p).astype(np.int64)
    above = k >= mode

    # Binary search for opposite-tail boundary where pmf(y) <= d
    # For k above mode, search lower tail [0, mode]; for k below mode, search upper tail [mode, n]
    lo = np.where(above, np.int64(0), mode)
    hi = np.where(above, mode, n_m)
    tol = d * (1 + 1e-10)

    for _ in range(55):
        mid = (lo + hi) // 2
        pmf_mid = binom.pmf(mid, n_m, p)
        go_left = np.where(above, pmf_mid > tol, pmf_mid <= tol)
        hi = np.where(go_left, mid, hi)
        lo = np.where(go_left, lo, mid + 1)

    # For both cases, lo converges to the first position where the condition flips.
    # above: lo = first x in [0, mode] with pmf > d → cdf(lo-1) gives the lower tail
    # below: lo = first x in [mode, n] with pmf <= d → sf(lo-1) gives the upper tail
    # When no boundary found, lo goes past hi: cdf(-1)=0 or sf(n)=0, correctly yielding 0.
    y = lo

    # Sum both tails
    pval = np.where(
        above,
        binom.cdf(y - 1, n_m, p) + binom.sf(k - 1, n_m, p),
        binom.cdf(k, n_m, p) + binom.sf(y - 1, n_m, p),
    )
    pval = np.clip(pval, 0.0, 1.0)
    result[mask] = pval
    return result
