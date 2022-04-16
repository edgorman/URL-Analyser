import os
import sys
import pytest
from io import StringIO


# NOTE: If you want to print anything while testing
#       You will need to comment out this fixture below
#       Which captures all text automatically
@pytest.fixture(autouse=True)
def tmp_out():
    out = StringIO()
    sys.stdout = out
    yield out
    sys.stdout = sys.__stdout__


@pytest.fixture(autouse=True)
def parent_directory():
    return os.path.dirname(os.path.realpath(__file__))


@pytest.fixture(autouse=True)
def data_directory(parent_directory):
    return os.path.join(parent_directory, "data")


@pytest.fixture(autouse=True)
def url_data_directory(data_directory):
    return os.path.join(data_directory, "urls")


@pytest.fixture(autouse=True)
def model_data_directory(data_directory):
    return os.path.join(data_directory, "models")


@pytest.fixture(autouse=True)
def features_data_directory(data_directory):
    return os.path.join(data_directory, "features")


@pytest.fixture(autouse=True)
def features_directory(parent_directory):
    return os.path.join(parent_directory, "features")


@pytest.fixture(autouse=True)
def models_directory(parent_directory):
    return os.path.join(parent_directory, "models")
