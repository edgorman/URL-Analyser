import os
import json
from lexical import get_lexical
from host import get_host
from content import get_content


def get_method(dataset_name):
    if dataset_name == 'lexical': return get_lexical
    if dataset_name == 'host': return get_host
    if dataset_name == 'content': return get_content

def get_vocab_dict():
    vocab_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "data", "vocab-dict.json")

    with open(vocab_path) as vocab_json:
        return json.load(vocab_json)

def get_train_test_features(dataset_name, train_set, test_set, feature_index):
    extract_method = get_method(dataset_name)
    train_feats = extract_method(train_set, feature_index)
    test_feats = extract_method(test_set, feature_index, get_vocab_dict())
    
    return train_feats, test_feats

def get_url_features(dataset_name, url_name, feature_index):
    extract_method = get_method(dataset_name)
    feats = extract_method(url_name, feature_index, get_vocab_dict())

    return feats