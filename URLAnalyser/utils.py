import os
import json
import pickle
from importlib import import_module
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer


def load_json_as_dict(filename):
    with open(filename, 'r') as f:
        return json.load(f)

def is_valid_url(url):
    # TODO: testing once this is done
    return True

def is_valid_model(models_dictionary, model_name, dataset_name, feature_index):
    if model_name in models_dictionary.keys():
        if dataset_name in models_dictionary[model_name]['featuresets'].keys():
            if int(feature_index) in models_dictionary[model_name]['featuresets'][dataset_name]['indexes']:
                return True
    return False

def generate_model_filename(model_name, dataset_name, feature_index):
    return f"{model_name}-{dataset_name}-{feature_index}"

def is_model_stored(model_name, dataset_name, feature_index, path=os.path.join(os.path.dirname(os.path.realpath(__file__)), "data", "models")):
    filename = generate_model_filename(model_name, dataset_name, feature_index)
    model_filenames = [s.split(".")[0] for s in os.listdir(path)]

    return filename in model_filenames

def get_class(class_name):
    paths = class_name.split('.')
    path = ".".join(paths[:-1])
    name = paths[-1]

    return getattr(import_module(path), name)

def get_urls(sample_rate=1, path=os.path.join(os.path.dirname(os.path.realpath(__file__)), "data", "urls")):
    if "whitelist.txt" in os.listdir(path) and "blacklist.txt" in os.listdir(path): 
        benign = pd.read_csv(os.path.join(path, "whitelist.txt"), header=None, names=["name"])
        benign['class'] = 0
        malicious = pd.read_csv(os.path.join(path, "blacklist.txt"), header=None, names=["name"])
        malicious['class'] = 1
        
        urls = pd.concat([benign, malicious]).dropna()
        return urls.sample(frac=sample_rate)
    return pd.DataFrame()

def split_urls(url_df):
    y = url_df['class']
    x = url_df.drop(['class'], axis=1)
    return train_test_split(x, y, test_size=0.2)

def bag_of_words(features, series, vocab):
    if len(vocab) == 0:
        vectorizer = CountVectorizer(decode_error="ignore")
        vectorizer.fit_transform(series)
    else:
        vectorizer = CountVectorizer(vocabulary=vocab, decode_error='ignore')
    
    bow_df = pd.DataFrame(vectorizer.transform(series).todense(), columns=vectorizer.get_feature_names())
    features = features.reset_index()
    features = pd.concat([features, bow_df], axis=1)
    return features.drop(['index'], axis=1)

def safe_division(a, b):
    return 0 if b == 0 else a / b

def save_model(model, filename, path=os.path.join(os.path.dirname(os.path.realpath(__file__)), "data", "models")):
    pickle.dump(model, open(os.path.join(path, filename + ".pkl"), 'wb'))

def load_model(filename, path=os.path.join(os.path.dirname(os.path.realpath(__file__)), "data", "models")):
    return pickle.load(open(os.path.join(path, filename + ".pkl"), 'rb'))
