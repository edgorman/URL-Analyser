import os
import pytest

from URLAnalyser.data.data import load_url_names


@pytest.fixture(autouse=True)
def results_dict_path():
    return os.path.join(os.path.dirname(os.path.realpath(__file__)), "data", "models", "results-dict.json")

@pytest.fixture(autouse=True)
def url_dataframe():
    return load_url_names(sample_rate=1, path=os.path.join(os.path.dirname(os.path.realpath(__file__)), "data", "urls"))
