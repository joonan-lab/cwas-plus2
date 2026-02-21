"""
Test the methods in cwas.utils.check
"""
import multiprocessing as mp
from pathlib import Path

import cwas.utils.check as check
import pytest


def test_check_is_file(tmpdir):
    # None cannot be accepted.
    with pytest.raises(ValueError):
        check.check_is_file(None)

    # If a input path is valid, there will be no error,
    real_path = tmpdir.join("hello.txt")
    real_path.write("Hello")
    check.check_is_file(real_path)
    assert True
    check.check_is_file(Path(real_path))
    assert True

    # If a input path is invalid, FileNotFoundError will be raised.
    fake_path = tmpdir.join("nofile.txt")
    with pytest.raises(FileNotFoundError):
        check.check_is_file(fake_path)


def test_check_is_dir(tmp_path):
    # None cannot be accepted.
    with pytest.raises(ValueError):
        check.check_is_dir(None)

    # If a directory of the input path exists, there will be no error.
    real_dir = tmp_path / "real"
    real_dir.mkdir()
    check.check_is_dir(real_dir)
    assert True
    check.check_is_dir(str(real_dir))
    assert True

    # If a directory of the input path does not exist,
    # NotADirectoryError is raised.
    fake_dir = tmp_path / "fake"
    with pytest.raises(NotADirectoryError):
        check.check_is_dir(fake_dir)


def test_check_num_proc():
    max_num_proc = mp.cpu_count()

    # The number of worker processes must be in this range.
    for num_proc in range(1, max_num_proc + 1):
        check.check_num_proc(num_proc)
        assert True

    # Otherwise, ValueError will be raised.
    with pytest.raises(ValueError):
        check.check_num_proc(-1)
    with pytest.raises(ValueError):
        check.check_num_proc(0)
    with pytest.raises(ValueError):
        check.check_num_proc(max_num_proc + 1)


def test_check_same_n_lines_same(tmp_path):
    f1 = tmp_path / "a.txt"
    f2 = tmp_path / "b.txt"
    f1.write_text("line1\nline2\nline3\n")
    f2.write_text("lineA\nlineB\nlineC\n")
    # Should not raise
    check.check_same_n_lines([f1, f2])


def test_check_same_n_lines_different(tmp_path):
    f1 = tmp_path / "a.txt"
    f2 = tmp_path / "b.txt"
    f1.write_text("line1\nline2\n")
    f2.write_text("lineA\nlineB\nlineC\n")
    with pytest.raises(ValueError, match="different number of lines"):
        check.check_same_n_lines([f1, f2])


def test_check_same_n_lines_single_file(tmp_path):
    f1 = tmp_path / "a.txt"
    f1.write_text("line1\n")
    # Single file should always pass
    check.check_same_n_lines([f1])
