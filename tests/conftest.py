import os
import pytest

from URLAnalyser.data.data import load_url_data


@pytest.fixture(autouse=True)
def test_path():
    return os.path.dirname(os.path.realpath(__file__))

@pytest.fixture(autouse=True)
def results_dict_path(test_path):
    return os.path.join(test_path, "data", "models", "results-dict.json")

@pytest.fixture(autouse=True)
def url_dataframe(test_path):
    return load_url_data("lexical", sample_rate=1, path=os.path.join(test_path, "data", "urls"))
