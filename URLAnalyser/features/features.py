import os
import pandas as pd
from sklearn.preprocessing import StandardScaler

from URLAnalyser import app
from URLAnalyser.utils import load_json_as_dict
from URLAnalyser.features.lexical import get_lexical
from URLAnalyser.features.host import get_host
from URLAnalyser.features.content import get_content


def _get_method(dataset_name):
    if dataset_name == 'lexical':
        return get_lexical
    if dataset_name == 'host':
        return get_host
    if dataset_name == 'content':
        return get_content


def get_url_features(url_name, dataset_name, feature_index):
    vocab_path = os.path.join(app.DATA_DIRECTORY, "features", "vocab-dict.json")
    url_set = pd.DataFrame(data={"name": [url_name]})
    get_method = _get_method(dataset_name)
    return get_method(url_set, feature_index, load_json_as_dict(vocab_path))


def get_train_test_features(train_set, test_set, dataset_name, feature_index):
    get_method = _get_method(dataset_name)
    vocab = load_json_as_dict(os.path.join(app.DATA_DIRECTORY, "features", "vocab-dict.json"))

    train_feats = get_method(train_set, feature_index)
    test_feats = get_method(test_set, feature_index, vocab)

    return train_feats, test_feats


def scale_features(train, test):
    scaler = StandardScaler().fit(train)
    train = scaler.transform(train)
    test = scaler.transform(test)

    return train, test
