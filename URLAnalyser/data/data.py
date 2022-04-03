import os
import re
import time
import datetime
import pandas as pd
import whois
from pythonping import ping
from sklearn.model_selection import train_test_split

from URLAnalyser.utils import is_url_valid


# LEXICAL ----------
def _load_lexical(sample_rate, path=os.path.join(os.path.dirname(os.path.realpath(__file__)), "urls")):
    if "whitelist.txt" in os.listdir(path) and "blacklist.txt" in os.listdir(path): 
        benign = _load_file("whitelist.txt", path)
        malicious = _load_file("blacklist.txt", path)
        return pd.concat([benign, malicious]).sample(frac=sample_rate)
    return None

# HOST ----------
def _get_host(url):
    try:
        return whois.whois(url)
    except:
        return None

def _host_registrar(obj, typ):
    try:
        return obj[typ]
    except:
        return None

def _host_country(obj, typ):
    try:
        return obj[typ]
    except:
        return None

def _host_server(obj, typ):
    try:
        return len(obj[typ])
    except:
        return 0

def _host_date(obj, typ, attr_str):
    try:
        d = obj[typ]
    except:
        return None

    if isinstance(obj[typ], list):
        obj[typ] = obj[typ][0]

    if not isinstance(obj[typ], datetime):
        return None

    return getattr(obj[typ], attr_str)

def _host_speed(url):
    t1 = time.perf_counter()
    host = _get_host(url)
    return time.perf_counter() - t1

def _host_latency(url):
    response = ping("google.com", size=40, count=10, timeout=1)
    return response.rtt_avg_ms

def _load_host(sample_rate):
    url_host = _load_lexical(sample_rate)
    if url_host is not None:
        url_host = url_host[is_url_valid(url_host.name)]

        # TODO: Get host related data for remaining urls
        return url_host
    return None

# CONTENT ----------
def _content_type(response):
    try:
        return response.headers['Content-Type']
    except:
        return "NA"

def _content_redirect(obj):
    try:
        return obj.is_redirect
    except:
        return None

def _content_content(obj):
    try:
        return obj.content
    except:
        return "NA"

def _load_content(sample_rate):
    url_cont = _load_lexical(sample_rate)
    if url_cont is not None:
        url_cont = url_cont[is_url_valid(url_cont.name)]

        # TODO: Get content related data for remaining urls
        return url_cont
    return None

# GENERAL METHODS ----------
def _load_file(filename, path=os.path.join(os.path.dirname(os.path.realpath(__file__)))):
    if filename in os.listdir(path):
        df = pd.read_csv(os.path.join(path, filename), header=None, names=["name"])
        df = df.dropna()
        df.insert(1, 'class', 1 if filename == "blacklist.txt" else 0)
        df['name'] = df['name'].apply(lambda x: re.sub(r"https?://(www\.)?", "", x))
        return df

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
