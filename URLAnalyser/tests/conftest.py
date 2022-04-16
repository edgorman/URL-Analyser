import os
import pytest


@pytest.fixture(autouse=True)
def test_directory():
    return os.path.dirname(os.path.realpath(__file__))
