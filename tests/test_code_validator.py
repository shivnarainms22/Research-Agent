"""Tests for experiments/code_validator.py"""
from __future__ import annotations

import pytest

from experiments.code_validator import validate, ValidationError, _check_ast


def test_valid_simple_code():
    code = "import numpy as np\nresult = np.array([1, 2, 3])\nprint(result)"
    validate(code)  # should not raise


def test_forbidden_subprocess_import():
    code = "import subprocess\nsubprocess.run(['ls'])"
    with pytest.raises(ValidationError, match="subprocess"):
        _check_ast(code)


def test_forbidden_pty_import():
    code = "import pty\npty.spawn('/bin/sh')"
    with pytest.raises(ValidationError, match="pty"):
        _check_ast(code)


def test_forbidden_ctypes_import():
    code = "import ctypes\nctypes.CDLL('libc.so.6')"
    with pytest.raises(ValidationError, match="ctypes"):
        _check_ast(code)


def test_forbidden_eval_call():
    code = "x = eval('1 + 1')"
    with pytest.raises(ValidationError, match="eval"):
        _check_ast(code)


def test_forbidden_exec_call():
    code = "exec('import os')"
    with pytest.raises(ValidationError, match="exec"):
        _check_ast(code)


def test_forbidden_rmtree_attribute():
    code = "import shutil\nshutil.rmtree('/tmp/workspace')"
    with pytest.raises(ValidationError, match="rmtree"):
        _check_ast(code)
