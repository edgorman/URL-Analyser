import os
import pytest

from URLAnalyser.utils import get_urls


@pytest.fixture(autouse=True)
def results_dict_path():
    return os.path.join(os.path.dirname(os.path.realpath(__file__)), "data", "models", "results-dict.json")

@pytest.fixture(autouse=True)
def url_df():
    return get_urls(sample_rate=1, path=os.path.join(os.path.dirname(os.path.realpath(__file__)), "data", "urls"))
