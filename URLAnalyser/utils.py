import os
import json
from importlib import import_module

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

PARENT_FOLDER = os.path.dirname(os.path.realpath(__file__))


def load_json_as_dict(filename):
    with open(filename, 'r') as f:
        return json.load(f)


def save_json_as_dict(dict, filename):
    with open(filename, 'w') as f:
        json.dump(dict, f)


def is_url_valid(url):
    # TODO: testing once this is done
    return True


def is_model_valid(models_dict, model_name, dataset_name, feature_index):
    if model_name in models_dict.keys():
        if dataset_name in models_dict[model_name]['featuresets'].keys():
            if int(
                    feature_index) in models_dict[model_name]['featuresets'][dataset_name]['indexes']:
                return True
    return False


def generate_model_filename(model_name, dataset_name, feature_index):
    return f"{model_name}-{dataset_name}-{feature_index}"


def is_model_stored(model_name, dataset_name, feature_index, path=os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "data", "models")):
    filename = generate_model_filename(model_name, dataset_name, feature_index)
    model_filenames = [s.split(".")[0] for s in os.listdir(path)]

    return filename in model_filenames


def get_class(class_name):
    paths = class_name.split('.')
    path = ".".join(paths[:-1])
    name = paths[-1]

    return getattr(import_module(path), name)


def bag_of_words(features, series, key, path=os.path.join(PARENT_FOLDER, "data", "features", "vocab-dict.json")):
    vocab = load_json_as_dict(path)

    if len(vocab) == 0 or key not in vocab:
        vectorizer = CountVectorizer(decode_error="ignore")
        vectorizer.fit_transform(series)
        vocab[key] = {k: v.tolist() for k, v in vectorizer.vocabulary_.items()}
        save_json_as_dict(vocab, path)
    else:
        vectorizer = CountVectorizer(vocabulary=vocab[key], decode_error='ignore')

    bow_df = pd.DataFrame(vectorizer.transform(series).todense(), columns=vectorizer.get_feature_names())
    features = features.reset_index()
    features = pd.concat([features, bow_df], axis=1)
    return features.drop(['index'], axis=1)


def safe_division(a, b):
    return 0 if b == 0 else a / b
