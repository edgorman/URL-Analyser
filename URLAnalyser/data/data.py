import os
import re
import pandas as pd
from sklearn.model_selection import train_test_split

from URLAnalyser.utils import is_url_valid
from URLAnalyser.data.host import get_host
from URLAnalyser.data.content import get_content


def _load_file(filename, path):
    if filename in os.listdir(path):
        df = pd.read_csv(os.path.join(path, filename), header=None, names=["name"])
        df = df.dropna()
        df.insert(1, 'class', 1 if filename == "blacklist.txt" else 0)
        df['name'] = df['name'].apply(lambda x: re.sub(r"https?://(www\.)?", "", x))
        return df

def _load_lexical(sample_rate, path):
    if "whitelist.txt" in os.listdir(path) and "blacklist.txt" in os.listdir(path): 
        benign = _load_file("whitelist.txt", path)
        malicious = _load_file("blacklist.txt", path)
        return pd.concat([benign, malicious]).sample(frac=sample_rate)
    return None

def _load_host(sample_rate, path):
    # Initialise host df with lexical values
    host = _load_lexical(sample_rate, path)
    if host is None:
        return None
    
    # Remove urls that are not valid
    host = host[is_url_valid(host.name)]

    # Extract host based information from sites
    host["info"] = host['name'].apply(lambda x: get_host(x))
    # TODO: apply extraction methods

    # Remove extra info column at the end
    host.drop(["info"], axis=1)
    return host

def _load_content(sample_rate, path):
    # Initialise content df with lexical values
    content = _load_lexical(sample_rate, path)
    if content is None:
        return None
    
    # Remove urls that are not valid
    content = content[is_url_valid(content.name)]

    # Extract content based information from sites
    content["info"] = content['name'].apply(lambda x: get_content(x))
    # TODO: apply extraction methods

    # Remove extra info column at the end
    content.drop(["info"], axis=1)
    return content

def _load_method(dataset_name):
    if dataset_name == 'lexical': return _load_lexical
    if dataset_name == 'host': return _load_host
    if dataset_name == 'content': return _load_content

def load_url_data(dataset_name, sample_rate=1, path=os.path.join(os.path.dirname(os.path.realpath(__file__)))):
    load_method = _load_method(dataset_name)
    return load_method(sample_rate, path)

def get_train_test_data(url_dataframe):
    y = url_dataframe['class']
    x = url_dataframe.drop(['class'], axis=1)
    return train_test_split(x, y, test_size=0.2)
