import os
import pandas as pd
from sklearn.model_selection import train_test_split


def _load_file(filename, path=os.path.join(os.path.dirname(os.path.realpath(__file__)))):
    if filename in os.listdir(path):
        df = pd.read_csv(os.path.join(path, filename), header=None, names=["name"])
        df.insert(1, 'class', 1 if filename == "blacklist.txt" else 0)
        df = df.dropna()
        return df

def _load_lexical(sample_rate, path=os.path.join(os.path.dirname(os.path.realpath(__file__)), "urls")):
    if "whitelist.txt" in os.listdir(path) and "blacklist.txt" in os.listdir(path): 
        benign = _load_file("whitelist.txt", path)
        malicious = _load_file("blacklist.txt", path)
        
        return pd.concat([benign, malicious]).sample(frac=sample_rate)
    return pd.DataFrame()

def _load_host(sample_rate):
    return pd.DataFrame()

def _load_content(sample_rate):
    return pd.DataFrame()

def _load_method(dataset_name):
    if dataset_name == 'lexical': return _load_lexical
    if dataset_name == 'host': return _load_host
    if dataset_name == 'content': return _load_content

def load_url_data(dataset_name, sample_rate=1):
    load_method = _load_method(dataset_name)
    return load_method(sample_rate)

def get_train_test_data(url_dataframe):
    y = url_dataframe['class']
    x = url_dataframe.drop(['class'], axis=1)
    return train_test_split(x, y, test_size=0.2)
