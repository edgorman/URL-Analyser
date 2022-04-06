import os
import pytest


@pytest.fixture(autouse=True)
def test_path():
    return os.path.dirname(os.path.realpath(__file__))

@pytest.fixture(autouse=True)
def results_dict_path(test_path):
    return os.path.join(test_path, "data", "models", "results-dict.json")
