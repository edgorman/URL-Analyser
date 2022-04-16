import os
import sys
import pytest
from io import StringIO


@pytest.fixture(autouse=True)
def tmp_out():
    out = StringIO()
    sys.stdout = out
    yield out
    sys.stdout = sys.__stdout__


@pytest.fixture(autouse=True)
def test_directory():
    return os.path.dirname(os.path.realpath(__file__))
