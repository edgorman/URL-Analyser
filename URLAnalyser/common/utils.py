import os
import json
from importlib import import_module
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer


def load_json_as_dict(filename):
    with open(filename, 'r') as f:
        return json.load(f)

def is_valid_url(url):
    return True

def is_valid_model(models_dictionary, model_name, dataset_name, feature_index):
    if model_name in models_dictionary.keys():
        if dataset_name in models_dictionary[model_name]['featuresets'].keys():
            if int(feature_index) in models_dictionary[model_name]['featuresets'][dataset_name]['indexes']:
                return True
    return False

def generate_model_filename(model_name, dataset_name, feature_index):
    return f"{model_name}-{dataset_name}-{feature_index}"

def is_model_stored(model_name, dataset_name, feature_index):
    filename = generate_model_filename(model_name, dataset_name, feature_index)
    saved_model_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "data", "models")
    saved_models = [s.split(".")[0] for s in os.listdir(saved_model_path)]

    return filename in saved_models

def get_class(class_name):
    paths = class_name.split('.')
    path = ".".join(paths[:-1])
    name = paths[-1]

    return getattr(import_module(path), name)

def bag_of_words(features, series, vocab):
    series = series.fillna("")
    vectorizer = CountVectorizer(vocabulary=vocab, decode_error='ignore')
    bow_df = pd.DataFrame(vectorizer.transform(series).todense(), columns=vectorizer.get_feature_names())
    features = features.reset_index()
    features = pd.concat([features, bow_df], axis=1)
    return features.drop(['index'], axis=1)

def safe_division(a, b):
    return 0 if b == 0 else a / b