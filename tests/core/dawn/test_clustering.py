"""
Test cwas.core.dawn.clustering — kmeans_cluster class.
"""
import pytest


def test_import():
    from cwas.core.dawn.clustering import kmeans_cluster  # noqa: F401


def test_kmeans_cluster_properties():
    import pandas as pd
    from cwas.core.dawn.clustering import kmeans_cluster

    df = pd.DataFrame({"t-SNE1": [1.0, 2.0, 3.0], "t-SNE2": [4.0, 5.0, 6.0]})
    kc = kmeans_cluster(df, seed=42)
    assert kc.seed == 42
    assert kc.tsne_out is df
