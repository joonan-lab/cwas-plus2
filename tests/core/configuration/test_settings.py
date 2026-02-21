"""
Test cwas.core.configuration.settings
"""
from cwas.core.configuration.settings import (
    get_default_domains,
    get_domain_types,
    get_redundant_domain_pairs,
)


def test_get_default_domains_returns_dict():
    domains = get_default_domains()
    assert isinstance(domains, dict)


def test_get_default_domains_has_required_keys():
    domains = get_default_domains()
    expected_keys = {"variant_type", "functional_score", "gene_set", "gencode", "functional_annotation"}
    assert set(domains.keys()) == expected_keys


def test_get_default_domains_variant_types():
    domains = get_default_domains()
    assert "All" in domains["variant_type"]
    assert "SNV" in domains["variant_type"]
    assert "Indel" in domains["variant_type"]


def test_get_default_domains_gencode_has_coding():
    domains = get_default_domains()
    assert "CodingRegion" in domains["gencode"]
    assert "NoncodingRegion" in domains["gencode"]
    assert "PTVRegion" in domains["gencode"]


def test_get_default_domains_is_deep_copy():
    """Modifying returned dict should not affect internal state."""
    d1 = get_default_domains()
    d1["variant_type"].append("Test")
    d2 = get_default_domains()
    assert "Test" not in d2["variant_type"]


def test_get_domain_types_returns_list():
    types = get_domain_types()
    assert isinstance(types, list)


def test_get_domain_types_content():
    types = get_domain_types()
    assert "variant_type" in types
    assert "gencode" in types
    assert "gene_set" in types
    assert len(types) == 5


def test_get_domain_types_is_deep_copy():
    t1 = get_domain_types()
    t1.append("extra")
    t2 = get_domain_types()
    assert "extra" not in t2


def test_get_redundant_domain_pairs_returns_dict():
    pairs = get_redundant_domain_pairs()
    assert isinstance(pairs, dict)


def test_get_redundant_domain_pairs_has_entries():
    pairs = get_redundant_domain_pairs()
    assert len(pairs) > 0
    # Each value should be a set of tuples
    for key, value in pairs.items():
        assert isinstance(key, tuple)
        assert len(key) == 2
        assert isinstance(value, set)
        for pair in value:
            assert isinstance(pair, tuple)
            assert len(pair) == 2


def test_get_redundant_domain_pairs_known_entry():
    pairs = get_redundant_domain_pairs()
    vt_gc = pairs.get(("variant_type", "gencode"))
    assert vt_gc is not None
    # SNV + FrameshiftRegion is a known redundant pair
    assert ("SNV", "FrameshiftRegion") in vt_gc


def test_get_redundant_domain_pairs_is_deep_copy():
    p1 = get_redundant_domain_pairs()
    p1[("variant_type", "gencode")].add(("Test", "Test"))
    p2 = get_redundant_domain_pairs()
    assert ("Test", "Test") not in p2[("variant_type", "gencode")]
