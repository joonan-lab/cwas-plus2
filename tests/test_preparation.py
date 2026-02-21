"""
Test cwas.preparation
"""
import random
from multiprocessing import cpu_count
from pathlib import Path
import pytest
from cwas.env import Env
import cwas.cli
import sys
import yaml



@pytest.fixture(scope="module", autouse=True)
def setup(cwas_workspace: Path, annotation_dir: Path):
    cwas_workspace.mkdir()
    create_annotation_keys_file(cwas_workspace)  # Add this line
    set_env(cwas_workspace, annotation_dir)

@pytest.fixture(scope="module", autouse=True)
def teardown(cwas_workspace: Path):
    yield
    reset_env()
    remove_workspace(cwas_workspace)

def create_annotation_keys_file(cwas_workspace: Path):
    path = cwas_workspace / "annotation_keys.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write("""
        functional_score:
          - example1
          - example2
        functional_annotation:
          - example3
          - example4
        """)

def set_env(cwas_workspace: Path, annotation_dir: Path):
    env = Env()
    env.set_env("CWAS_WORKSPACE", cwas_workspace)
    env.set_env("ANNOTATION_DATA", annotation_dir)
    env.set_env("ANNOTATION_BED_KEY", cwas_workspace / "annotation_keys.yaml")
    env.save()

def reset_env():
    env = Env()
    env.reset()
    env.remove_file()

def remove_workspace(cwas_workspace: Path):
    for f in cwas_workspace.glob("*"):
        f.unlink()
    cwas_workspace.rmdir()

def test_default_args():
    sys.argv = ['cwas', 'preparation']
    inst = cwas.cli.main()
    assert getattr(inst, "num_proc") == 1
    assert getattr(inst, "force_overwrite") == 0

def test_parse_args():
    cpu = random.choice(range(1, cpu_count() + 1))
    sys.argv = ['cwas', 'preparation', "-p", str(cpu), "--force_overwrite"]
    inst = cwas.cli.main()
    assert getattr(inst, "num_proc") == cpu
    assert getattr(inst, "force_overwrite") == 1

    sys.argv = ['cwas', 'preparation', "-p", str(cpu)]
    inst = cwas.cli.main()
    assert getattr(inst, "num_proc") == cpu
    assert getattr(inst, "force_overwrite") == 0

    sys.argv = ['cwas', 'preparation', "--force_overwrite"]
    inst = cwas.cli.main()
    assert getattr(inst, "num_proc") == 1
    assert getattr(inst, "force_overwrite") == 1

def test_parse_args_value_error():
    cpu = cpu_count() + 1
    args = ["-p", str(cpu)]
    with pytest.raises(ValueError):
        sys.argv = ['cwas', 'preparation', *args]
        cwas.cli.main()

    args = ["-p", "0"]
    with pytest.raises(ValueError):
        sys.argv = ['cwas', 'preparation', *args]
        cwas.cli.main()
