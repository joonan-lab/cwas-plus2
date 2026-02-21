import argparse
import pytest
from pathlib import Path
import cwas.factory
from cwas.runnable import Runnable


def test_make_class_name():
    assert cwas.factory.make_class_name("categorization") == "Categorization"
    assert cwas.factory.make_class_name("annotation") == "Annotation"
    assert cwas.factory.make_class_name("burden_test") == "BurdenTest"
    assert cwas.factory.make_class_name("permutation_test") == "PermutationTest"
    assert cwas.factory.make_class_name("burden_shift") == "BurdenShift"
    assert cwas.factory.make_class_name("risk_score") == "RiskScore"
    assert cwas.factory.make_class_name("dawn") == "Dawn"


def test_create_factory():
    input_path = Path(__file__).parent / "test_file/test_annotated.vcf.gz"
    output_dir_path = Path(__file__).parent / "test_file"
    cat_path = Path(__file__).parent / "test_file/test.categorization_result.zarr"
    sample_info_path = Path(__file__).parent / "test_file/sample.txt"
    adj_factor_path = Path(__file__).parent / "test_file/adj_factors.txt"

    factory_inst = cwas.factory.create("categorization")
    args = argparse.Namespace(num_proc=1, input_path=str(input_path),
                              output_dir_path=str(output_dir_path))
    assert isinstance(factory_inst.argparser(), argparse.ArgumentParser)
    assert isinstance(factory_inst.runnable(args), Runnable)

    factory_inst = cwas.factory.create("binomial_test")
    args = argparse.Namespace(num_proc=1, cat_path=str(cat_path),
                              output_dir_path=str(output_dir_path),
                              sample_info_path = str(sample_info_path),
                              adj_factor_path = str(adj_factor_path),
                              use_n_carrier = False)
    assert isinstance(factory_inst.argparser(), argparse.ArgumentParser)
    assert isinstance(factory_inst.runnable(args), Runnable)

def test_create_factory_with_invalid_step():
    with pytest.raises(ValueError):
        cwas.factory.create("burden_test")


def test_make_class_name_single_word():
    assert cwas.factory.make_class_name("start") == "Start"
    assert cwas.factory.make_class_name("configuration") == "Configuration"
    assert cwas.factory.make_class_name("preparation") == "Preparation"


def test_make_class_name_multi_word():
    assert cwas.factory.make_class_name("effective_num_test") == "EffectiveNumTest"
    assert cwas.factory.make_class_name("extract_variant") == "ExtractVariant"


def test_cwas_factory_properties():
    factory_inst = cwas.factory.create("categorization")
    assert callable(factory_inst.argparser)
    assert factory_inst.runnable is not None


def test_create_nonexistent_module():
    with pytest.raises(ValueError, match="does not support"):
        cwas.factory.create("this_does_not_exist")


def test_create_all_valid_steps():
    # These steps should all resolve without error
    for step in ["start", "configuration", "preparation", "annotation",
                  "categorization", "binomial_test", "permutation_test",
                  "extract_variant", "effective_num_test", "correlation"]:
        factory_inst = cwas.factory.create(step)
        assert factory_inst.argparser is not None
        assert factory_inst.runnable is not None