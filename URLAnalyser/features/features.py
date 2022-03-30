import os
from ..utils import load_json_as_dict
from URLAnalyser.features.lexical import get_lexical
from URLAnalyser.features.host import get_host
from URLAnalyser.features.content import get_content


def get_method(dataset_name):
    if dataset_name == 'lexical': return get_lexical
    if dataset_name == 'host': return get_host
    if dataset_name == 'content': return get_content

def get_vocab_dict():
    vocab_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "data", "vocab-dict.json")
    return load_json_as_dict(vocab_path)

def get_train_test_features(dataset_name, train_set, test_set, feature_index):
    extract_method = get_method(dataset_name)
    train_feats = extract_method(train_set, feature_index)
    test_feats = extract_method(test_set, feature_index, get_vocab_dict())
    
    return train_feats, test_feats

def get_url_features(dataset_name, url_name, feature_index):
    extract_method = get_method(dataset_name)
    feats = extract_method(url_name, feature_index, get_vocab_dict())

    return feats