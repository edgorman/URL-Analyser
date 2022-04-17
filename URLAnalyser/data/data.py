import os
import re
import pandas as pd
from sklearn.model_selection import train_test_split

from URLAnalyser.log import Log
from URLAnalyser.constants import URL_DATA_DIRECTORY
from URLAnalyser.utils import url_is_valid
from URLAnalyser.utils import load_csv_as_dataframe
from URLAnalyser.utils import save_dataframe_to_csv
from URLAnalyser.data.host import get_host
from URLAnalyser.data.host import host_registrar
from URLAnalyser.data.host import host_country
from URLAnalyser.data.host import host_server_count
from URLAnalyser.data.host import host_date
from URLAnalyser.data.host import host_speed
from URLAnalyser.data.host import host_latency
from URLAnalyser.data.content import get_content
from URLAnalyser.data.content import content_type
from URLAnalyser.data.content import content_redirect
from URLAnalyser.data.content import content_content


def _load_lexical(sample_rate: float, use_cache: bool, is_keras: bool):
    '''
        Load the ground truth lexical data from the local text files

        Parameters:
            sample_rate: Percentage of urls to sample
            use_cache: Whether to use stored version (ignored)
            is_keras: True if data will be used for a keras model

        Returns:
            dataframe: Dataframe object containing url names and classes
    '''
    if "whitelist.txt" in os.listdir(URL_DATA_DIRECTORY) and "blacklist.txt" in os.listdir(URL_DATA_DIRECTORY):
        benign = load_csv_as_dataframe(os.path.join(URL_DATA_DIRECTORY, "whitelist.txt"))
        malicious = load_csv_as_dataframe(os.path.join(URL_DATA_DIRECTORY, "blacklist.txt"))

        benign.insert(1, 'class', 0)
        malicious.insert(1, 'class', 1)

        urls = pd.concat([benign, malicious])
        urls = urls.sample(frac=sample_rate) if is_keras else urls.sample(n=min(len(urls), 20000))

        urls["name"] = urls["name"].apply(lambda x: re.sub(r"https?://(www\.)?", "", x))
        urls["is_valid"] = False
        return urls
    return None


def _load_host(sample_rate: float, use_cache: bool, is_keras: bool):
    '''
        Load the host data using the url names

        Parameters:
            sample_rate: Percentage of urls to sample
            use_cache: Whether to use stored version
            is_keras: True if data will be used for a keras model

        Returns:
            dataframe: Dataframe object containing host data
    '''
    # If cache enabled, try load from their first
    if use_cache and "host.csv" in os.listdir(URL_DATA_DIRECTORY):
        return load_csv_as_dataframe(os.path.join(URL_DATA_DIRECTORY, "host.csv"))

    # Initialise host df with lexical values
    host = _load_lexical(sample_rate, use_cache, is_keras)
    if host is None:
        return None

    # Remove urls that are not valid
    host["is_valid"] = host["name"].apply(lambda x: url_is_valid(x))
    host = host[host.is_valid]

    # Extract host based information from sites
    host["info"] = host['name'].apply(lambda x: get_host(x))
    host["registrar"] = host['info'].apply(lambda x: host_registrar(x))
    host["location"] = host['info'].apply(lambda x: host_country(x))
    host["server_count"] = host['info'].apply(lambda x: host_server_count(x))
    host["creation_date"] = host['info'].apply(lambda x: host_date(x, "creation_date"))
    host["updated_date"] = host['info'].apply(lambda x: host_date(x, "updated_date"))
    host["expiration_date"] = host['info'].apply(lambda x: host_date(x, "expiration_date"))
    host["speed"] = host['name'].apply(lambda x: host_speed(x))
    host["latency"] = host['name'].apply(lambda x: host_latency(x))

    # Remove extra info column at the end
    host.drop(["info"], axis=1, inplace=True)
    save_dataframe_to_csv(host, os.path.join(URL_DATA_DIRECTORY, "host.csv"))
    return host


def _load_content(sample_rate: float, use_cache: bool, is_keras: bool):
    '''
        Load the content data using the url names

        Parameters:
            sample_rate: Percentage of urls to sample
            use_cache: Whether to use stored version
            is_keras: True if data will be used for a keras model

        Returns:
            dataframe: Dataframe object containing content data
    '''
    # If cache enabled, try load from their first
    if use_cache and "content.csv" in os.listdir(URL_DATA_DIRECTORY):
        return load_csv_as_dataframe(os.path.join(URL_DATA_DIRECTORY, "content.csv"))

    # Initialise content df with lexical values
    content = _load_lexical(sample_rate, use_cache, is_keras)
    if content is None:
        return None

    # Remove urls that are not valid
    content["is_valid"] = content["name"].apply(lambda x: url_is_valid(x))
    content = content[content.is_valid]

    # Extract content based information from sites
    content["info"] = content['name'].apply(lambda x: get_content(x))
    content["type"] = content['info'].apply(lambda x: content_type(x))
    content["is_redirect"] = content['info'].apply(lambda x: content_redirect(x))
    content["content"] = content['info'].apply(lambda x: content_content(x))

    # Remove extra info column at the end
    content.drop(["info"], axis=1, inplace=True)
    save_dataframe_to_csv(content, os.path.join(URL_DATA_DIRECTORY, "content.csv"))
    return content


def _load_method(dataset_name: str):
    '''
        Load the dataframe for the corresponding dataset

        Parameters:
            dataset_name: Name of the chosen dataset

        Returns:
            method: Method for loading dataframe for dataset
    '''
    if dataset_name == 'lexical':
        return _load_lexical
    elif dataset_name == 'host':
        return _load_host
    elif dataset_name == 'content':
        return _load_content
    else:
        return None


def get_url_data(dataset_name: str, sample_rate: float = 1, use_cache: bool = True, is_keras: bool = False):
    '''
        Load the url dataframe for the corresponding dataset

        Parameters:
            dataset_name: Name of the chosen dataset
            sample_rate: Percentage of urls to sample
            use_cache: Whether to use stored version
            is_keras: True if data will be used for a keras model

        Returns:
            dataframe: Dataframe object containing url data
    '''
    load_method = _load_method(dataset_name)
    if load_method is None:
        Log.error(f"Failed to find load method for dataset '{dataset_name}'.")

    url_data = load_method(sample_rate, use_cache, is_keras)
    if url_data is None:
        Log.error(f"Failed to load data for '{dataset_name}'.")

    url_data.drop(["is_valid"], inplace=True, axis=1)
    url_data.reset_index(inplace=True, drop=True)
    return url_data


def get_train_test_data(url_data: pd.DataFrame):
    '''
        Split the url dataframe into a train and test set

        Parameters:
            url_data: Dataframe object containing url data

        Returns:
            x_train: Training features
            x_test: Testing features
            y_train: Training labels
            y_test: Testing labels
    '''
    y = url_data['class']
    x = url_data.drop(['class'], axis=1)
    return train_test_split(x, y, test_size=0.2)
