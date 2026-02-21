import cwas.core.categorization.utils as utils


def test_get_idx_dict():
    assert utils.get_idx_dict(["a", "b", "c"]) == {"a": 0, "b": 1, "c": 2}


def test_extract_sublist_by_int():
    test_list = ["a", "b", "c", "d"]
    assert utils.extract_sublist_by_int(test_list, 0) == []
    assert utils.extract_sublist_by_int(test_list, 1) == ["a"]
    assert utils.extract_sublist_by_int(test_list, 2) == ["b"]
    assert utils.extract_sublist_by_int(test_list, 4) == ["c"]
    assert utils.extract_sublist_by_int(test_list, 8) == ["d"]
    assert utils.extract_sublist_by_int(test_list, 3) == ["a", "b"]
    assert utils.extract_sublist_by_int(test_list, 11) == ["a", "b", "d"]
    assert utils.extract_sublist_by_int(test_list, 12) == ["c", "d"]
    assert utils.extract_sublist_by_int(test_list, 13) == ["a", "c", "d"]


def test_bitmask_to_indices_zero():
    assert utils.bitmask_to_indices(0) == []


def test_bitmask_to_indices_powers_of_two():
    assert utils.bitmask_to_indices(1) == [0]
    assert utils.bitmask_to_indices(2) == [1]
    assert utils.bitmask_to_indices(4) == [2]
    assert utils.bitmask_to_indices(8) == [3]


def test_bitmask_to_indices_multiple_bits():
    assert utils.bitmask_to_indices(3) == [0, 1]
    assert utils.bitmask_to_indices(5) == [0, 2]
    assert utils.bitmask_to_indices(7) == [0, 1, 2]
    assert utils.bitmask_to_indices(10) == [1, 3]


def test_bitmask_to_indices_all_bits():
    assert utils.bitmask_to_indices(15) == [0, 1, 2, 3]
    assert utils.bitmask_to_indices(255) == list(range(8))
