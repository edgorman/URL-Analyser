import os
import sys
import pytest
from io import StringIO

import pandas as pd
import tensorflow as tf
from sklearn.svm import SVC

from URLAnalyser.features.features import get_train_features, scale_train_features


# NOTE: If you want to print anything while testing
#       You will need to comment out this fixture below
#       Which captures all text automatically
@pytest.fixture(autouse=True)
def tmp_out():
    out = StringIO()
    sys.stdout = out
    yield out
    sys.stdout = sys.__stdout__


@pytest.fixture(autouse=True, scope='module')
def parent_directory():
    return os.path.dirname(os.path.realpath(__file__))


@pytest.fixture(autouse=True, scope='module')
def data_directory(parent_directory):
    return os.path.join(parent_directory, "data")


@pytest.fixture(autouse=True, scope='module')
def url_data_directory(data_directory):
    return os.path.join(data_directory, "urls")


@pytest.fixture(autouse=True, scope='module')
def model_data_directory(data_directory):
    return os.path.join(data_directory, "models")


@pytest.fixture(autouse=True, scope='module')
def features_data_directory(data_directory):
    return os.path.join(data_directory, "features")


@pytest.fixture(autouse=True, scope='module')
def features_directory(parent_directory):
    return os.path.join(parent_directory, "features")


@pytest.fixture(autouse=True, scope='module')
def models_directory(parent_directory):
    return os.path.join(parent_directory, "models")


@pytest.fixture(autouse=True)
def train_test_data():
    x_train, x_test = get_train_features(
        pd.DataFrame(data={"name": ["example.com", "another.com", "athird.com", "afourth.com"]}),
        pd.DataFrame(data={"name": ["google.com", "another.com", "athird.com", "afourth.com"]}),
        "lexical",
        1)
    x_train, x_test = scale_train_features(x_train, x_test)
    y_train, y_test = pd.Series(data=[1, 0, 1, 0]), pd.Series(data=[0, 1, 0, 1])
    return x_train, x_test, y_train, y_test


@pytest.fixture
def keras_model(train_test_data):
    x_train, _, y_train, _ = train_test_data

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(1, activation='sigmoid', input_shape=(1,)))
    model.compile(optimizer='adam', loss='binary_crossentropy')
    model.fit(x_train, y_train)

    return model


@pytest.fixture
def sklearn_model(train_test_data):
    x_train, _, y_train, _ = train_test_data

    model = SVC()
    model.fit(x_train, y_train)

    return model
