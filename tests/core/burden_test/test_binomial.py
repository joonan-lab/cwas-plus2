"""
Test cwas.core.burden_test.binomial
"""
import numpy as np
import pytest
from scipy.stats import binomtest

from cwas.core.burden_test.binomial import (
    binom_one_tail,
    binom_one_tail_vectorized,
    binom_two_tail,
    binom_two_tail_vectorized,
)


# --- binom_two_tail tests ---

def test_binom_two_tail_zero_total():
    """n1 + n2 == 0 should return 1.0."""
    assert binom_two_tail(0, 0, 0.5) == 1.0


def test_binom_two_tail_equal_split():
    """Equal split at p=0.5 should give p-value of 1.0."""
    result = binom_two_tail(5, 5, 0.5)
    expected = binomtest(5, 10, 0.5, alternative="two-sided").pvalue
    assert result == pytest.approx(expected)


def test_binom_two_tail_extreme():
    """Very extreme split should give small p-value."""
    result = binom_two_tail(10, 0, 0.5)
    assert result < 0.01


def test_binom_two_tail_matches_scipy():
    """Verify consistency with scipy.stats.binomtest."""
    for n1, n2, p in [(3, 7, 0.5), (8, 2, 0.3), (1, 9, 0.5)]:
        result = binom_two_tail(n1, n2, p)
        expected = binomtest(n1, n1 + n2, p, alternative="two-sided").pvalue
        assert result == pytest.approx(expected), f"Failed for n1={n1}, n2={n2}, p={p}"


def test_binom_two_tail_float_inputs():
    """Should accept and truncate float inputs to int."""
    result = binom_two_tail(3.7, 6.2, 0.5)
    expected = binom_two_tail(3, 6, 0.5)
    assert result == pytest.approx(expected)


# --- binom_one_tail tests ---

def test_binom_one_tail_zero_total():
    assert binom_one_tail(0, 0, 0.5) == 1.0


def test_binom_one_tail_all_success():
    result = binom_one_tail(10, 0, 0.5)
    expected = binomtest(10, 10, 0.5, alternative="greater").pvalue
    assert result == pytest.approx(expected)


def test_binom_one_tail_no_success():
    result = binom_one_tail(0, 10, 0.5)
    expected = binomtest(0, 10, 0.5, alternative="greater").pvalue
    assert result == pytest.approx(expected)


def test_binom_one_tail_matches_scipy():
    for n1, n2, p in [(3, 7, 0.5), (8, 2, 0.3), (5, 5, 0.5)]:
        result = binom_one_tail(n1, n2, p)
        expected = binomtest(n1, n1 + n2, p, alternative="greater").pvalue
        assert result == pytest.approx(expected)


# --- binom_one_tail_vectorized tests ---

def test_binom_one_tail_vectorized_matches_scalar():
    n1 = np.array([3, 8, 0, 5])
    n2 = np.array([7, 2, 10, 5])
    p = 0.5
    result = binom_one_tail_vectorized(n1, n2, p)
    for i in range(len(n1)):
        expected = binom_one_tail(n1[i], n2[i], p)
        assert result[i] == pytest.approx(expected, rel=1e-6), f"Mismatch at index {i}"


def test_binom_one_tail_vectorized_zero_total():
    n1 = np.array([0, 5])
    n2 = np.array([0, 5])
    result = binom_one_tail_vectorized(n1, n2, 0.5)
    assert result[0] == 1.0


def test_binom_one_tail_vectorized_shape():
    n1 = np.array([1, 2, 3])
    n2 = np.array([9, 8, 7])
    result = binom_one_tail_vectorized(n1, n2, 0.5)
    assert result.shape == (3,)


# --- binom_two_tail_vectorized tests ---

def test_binom_two_tail_vectorized_matches_scalar():
    n1 = np.array([3, 8, 0, 5, 10])
    n2 = np.array([7, 2, 10, 5, 0])
    p = 0.5
    result = binom_two_tail_vectorized(n1, n2, p)
    for i in range(len(n1)):
        expected = binom_two_tail(n1[i], n2[i], p)
        assert result[i] == pytest.approx(expected, rel=1e-4), f"Mismatch at index {i}"


def test_binom_two_tail_vectorized_zero_total():
    n1 = np.array([0, 5])
    n2 = np.array([0, 5])
    result = binom_two_tail_vectorized(n1, n2, 0.5)
    assert result[0] == 1.0


def test_binom_two_tail_vectorized_all_zeros():
    n1 = np.array([0, 0, 0])
    n2 = np.array([0, 0, 0])
    result = binom_two_tail_vectorized(n1, n2, 0.5)
    np.testing.assert_array_equal(result, [1.0, 1.0, 1.0])


def test_binom_two_tail_vectorized_shape():
    n1 = np.array([1, 2, 3])
    n2 = np.array([9, 8, 7])
    result = binom_two_tail_vectorized(n1, n2, 0.5)
    assert result.shape == (3,)


def test_binom_two_tail_vectorized_unequal_p():
    """Test with p != 0.5."""
    n1 = np.array([7, 3])
    n2 = np.array([3, 7])
    p = 0.3
    result = binom_two_tail_vectorized(n1, n2, p)
    for i in range(len(n1)):
        expected = binom_two_tail(n1[i], n2[i], p)
        assert result[i] == pytest.approx(expected, rel=1e-4)


def test_binom_two_tail_vectorized_pvalue_range():
    """All p-values should be in [0, 1]."""
    np.random.seed(42)
    n1 = np.random.randint(0, 20, size=50)
    n2 = np.random.randint(0, 20, size=50)
    result = binom_two_tail_vectorized(n1, n2, 0.5)
    assert np.all(result >= 0.0)
    assert np.all(result <= 1.0)
