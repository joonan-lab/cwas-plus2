"""
Test cwas.core.dawn.supernodeWGS — pure utility methods.
"""
import numpy as np
import pandas as pd
import pytest


def _has_igraph():
    try:
        import igraph  # noqa: F401
        return True
    except ImportError:
        return False


requires_igraph = pytest.mark.skipif(
    not _has_igraph(),
    reason="igraph not available",
)


def _make_supernodeWGS(**kwargs):
    from cwas.core.dawn.supernodeWGS import supernodeWGS_func

    defaults = dict(
        corr_mat=np.eye(3),
        fit_res=pd.DataFrame({"annotation": ["A", "B", "C"], "cluster": [1, 1, 2]}),
        max_cluster=3,
        cores=1,
        output_dir_path="/tmp",
        tag="test",
        seed=42,
    )
    defaults.update(kwargs)
    return supernodeWGS_func(**defaults)


def _make_data_collection(**kwargs):
    from cwas.core.dawn.supernodeWGS import data_collection

    defaults = dict(
        path="/tmp",
        cores=1,
        max_cluster=5,
        seed=42,
    )
    defaults.update(kwargs)
    return data_collection(**defaults)


# --- supernodeWGS_func properties ---


@requires_igraph
def test_properties():
    inst = _make_supernodeWGS()
    assert inst.max_cluster == 3
    assert inst.cores == 1
    assert inst.tag == "test"
    assert inst.seed == 42
    assert inst.verbose is True


@requires_igraph
def test_clusters_from_fit_res():
    inst = _make_supernodeWGS()
    assert inst.clusters == [1, 1, 2]


# --- _index_to_pair ---


@requires_igraph
def test_index_to_pair_first():
    inst = _make_supernodeWGS(max_cluster=5)
    assert inst._index_to_pair(1, 5) == (1, 1)


@requires_igraph
def test_index_to_pair_second():
    inst = _make_supernodeWGS(max_cluster=5)
    assert inst._index_to_pair(2, 5) == (1, 2)


@requires_igraph
def test_index_to_pair_diagonal():
    inst = _make_supernodeWGS(max_cluster=5)
    # val=6 corresponds to (2,2) for max_val=5
    assert inst._index_to_pair(6, 5) == (2, 2)


@requires_igraph
def test_index_to_pair_last():
    inst = _make_supernodeWGS(max_cluster=5)
    # max index = 5*(5-1)/2 + 5 = 15
    assert inst._index_to_pair(15, 5) == (5, 5)


@requires_igraph
def test_index_to_pair_invalid():
    inst = _make_supernodeWGS(max_cluster=5)
    with pytest.raises(AssertionError):
        inst._index_to_pair(0, 5)
    with pytest.raises(AssertionError):
        inst._index_to_pair(16, 5)


# --- _matching_ ---


@requires_igraph
def test_matching_identity():
    inst = _make_supernodeWGS()
    name1 = np.array(["A", "B", "C"])
    name2 = np.array(["A", "B", "C"])
    result = inst._matching_(name1, name2)
    np.testing.assert_array_equal(result, [1, 2, 3])


@requires_igraph
def test_matching_reordered():
    inst = _make_supernodeWGS()
    name1 = np.array(["C", "A"])
    name2 = np.array(["A", "B", "C"])
    result = inst._matching_(name1, name2)
    np.testing.assert_array_equal(result, [3, 1])


# --- _cluster_size_ ---


@requires_igraph
def test_cluster_size_no_flags():
    inst = _make_supernodeWGS()
    vec = np.array([0.1, 0.2, 0.3])
    clustering = [1, 1, 2]
    flag_vec = np.array([False, False, False])
    result = inst._cluster_size_(vec, clustering, flag_vec)
    assert result == [2, 1]


@requires_igraph
def test_cluster_size_with_flags():
    inst = _make_supernodeWGS()
    vec = np.array([0.1, 0.2, 0.3])
    clustering = [1, 1, 2]
    flag_vec = np.array([True, False, False])
    result = inst._cluster_size_(vec, clustering, flag_vec)
    assert result == [1, 1]


# --- _screen_name_ ---


@requires_igraph
def test_screen_name_noncoding():
    inst = _make_supernodeWGS()
    names = ["SNV_GeneSet1_All_NoncodingRegion_FuncAnnot"]
    result = inst._screen_name_(names)
    assert result == [True]


@requires_igraph
def test_screen_name_coding():
    inst = _make_supernodeWGS()
    names = ["SNV_GeneSet1_All_CodingRegion_FuncAnnot"]
    result = inst._screen_name_(names)
    assert result == [False]


# --- _node_color_ ---


@requires_igraph
def test_node_color_at_max():
    inst = _make_supernodeWGS()
    color = inst._node_color_(maxz=5.0, minz=0.0, x=5.0)
    assert isinstance(color, str)
    assert color.startswith("#")


@requires_igraph
def test_node_color_at_min():
    inst = _make_supernodeWGS()
    color = inst._node_color_(maxz=5.0, minz=0.0, x=0.0)
    assert isinstance(color, str)


# --- _value_to_color ---


@requires_igraph
def test_value_to_color():
    from matplotlib import cm
    inst = _make_supernodeWGS()
    cmap = cm.get_cmap("Reds", 256)(np.linspace(0, 1, 256))
    color = inst._value_to_color(cmap=cmap, maxz=10.0, minz=0.0, x=5.0)
    assert len(color) == 4  # RGBA


# --- report_results ---


@requires_igraph
def test_report_results():
    inst = _make_supernodeWGS()
    result = inst.report_results(
        vec=["cat1", "cat2", "cat3"],
        posterior=np.array([0.8, 0.5, 0.2]),
        pvalue=np.array([0.01, 0.05, 0.5]),
        Iupdate=np.array([1, 1, 0]),
    )
    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["Name", "p.value", "FDR", "indicator"]
    assert len(result) == 3


# --- _term_freq ---


@requires_igraph
def test_term_freq_basic():
    inst = _make_supernodeWGS(tag="test")
    cats = [["SNV_GeneA_All_Coding_Func1", "SNV_GeneB_All_Coding_Func2"]]
    result = inst._term_freq(cats[0])
    # "Coding" appears 2 times (most frequent non-excluded term)
    assert result == "Coding"


# --- data_collection ---


@requires_igraph
def test_data_collection_properties():
    dc = _make_data_collection()
    assert dc.path == "/tmp"
    assert dc.cores == 1
    assert dc.max_cluster == 5
    assert dc.seed == 42


@requires_igraph
def test_pair_to_index_identity():
    dc = _make_data_collection(max_cluster=5)
    assert dc._pair_to_index_(1, 1, 5) == 1


@requires_igraph
def test_pair_to_index_symmetry():
    dc = _make_data_collection(max_cluster=5)
    assert dc._pair_to_index_(1, 2, 5) == dc._pair_to_index_(2, 1, 5)


@requires_igraph
def test_pair_to_index_boundary():
    dc = _make_data_collection(max_cluster=5)
    assert dc._pair_to_index_(5, 5, 5) == 15


@requires_igraph
def test_pair_to_index_invalid():
    dc = _make_data_collection(max_cluster=5)
    with pytest.raises(AssertionError):
        dc._pair_to_index_(0, 1, 5)
    with pytest.raises(AssertionError):
        dc._pair_to_index_(6, 1, 5)


# --- soft and l2n ---


@requires_igraph
def test_soft_basic():
    dc = _make_data_collection()
    x = np.array([3.0, -2.0, 0.5])
    result = dc.soft(x, 1.0)
    np.testing.assert_array_almost_equal(result, [2.0, -1.0, 0.0])


@requires_igraph
def test_soft_zero_threshold():
    dc = _make_data_collection()
    x = np.array([1.0, -1.0])
    result = dc.soft(x, 0.0)
    np.testing.assert_array_almost_equal(result, [1.0, -1.0])


@requires_igraph
def test_l2n_basic():
    dc = _make_data_collection()
    assert dc.l2n(np.array([3.0, 4.0])) == pytest.approx(5.0)


@requires_igraph
def test_l2n_zero():
    dc = _make_data_collection()
    assert dc.l2n(np.array([0.0, 0.0])) == pytest.approx(0.05)


# --- _determine_sign_ ---


@requires_igraph
def test_determine_sign_positive():
    dc = _make_data_collection()
    vec = np.array([1.0, 2.0, 0.5, -0.1])
    assert dc._determine_sign_(vec) == 1


@requires_igraph
def test_determine_sign_negative():
    dc = _make_data_collection()
    vec = np.array([-1.0, -2.0, -0.5, 0.1])
    assert dc._determine_sign_(vec) == -1


# --- safesvd ---


@requires_igraph
def test_safesvd_basic():
    dc = _make_data_collection()
    x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    U, s, Vt = dc.safesvd(x)
    assert U.shape[0] == 3
    assert len(s) == 2
