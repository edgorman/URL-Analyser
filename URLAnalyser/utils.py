import os
import json
import pandas as pd
from importlib import import_module
from sklearn.feature_extraction.text import CountVectorizer

from URLAnalyser.constants import FEATURES_DATA_DIRECTORY
from URLAnalyser.constants import MODEL_DATA_DIRECTORY
from URLAnalyser.data.host import get_host
from URLAnalyser.data.content import get_content


def load_json_as_dict(filename: str) -> dict:
    '''
        Load a JSON file into a dictionary

        Parameters:
            filename: Name of the JSON file

        Returns:
            dict: JSON file as a dictionary
    '''
    with open(filename, 'r') as f:
        return json.load(f)


def save_dict_as_json(dict: dict, filename: str) -> None:
    '''
        Save a dictionary into a JSON file

        Parameters:
            dict: Dictionary object
            filename: Name of the JSON file

        Returns:
            None
    '''
    with open(filename, 'w') as f:
        json.dump(dict, f)


def load_csv_as_dataframe(filename: str) -> pd.DataFrame:
    '''
        Load a CSV file into a dataframe

        Parameters:
            filename: Name of the CSV file

        Returns:
            dataframe: CSV file as a dataframe
    '''
    return pd.read_csv(filename, header=0)


def save_dataframe_to_csv(df: pd.DataFrame, filename: str) -> None:
    '''
        Save a dataframe into a CSV file

        Parameters:
            df: Dataframe object
            filename: Name of the CSV file

        Returns:
            None
    '''
    df.to_csv(filename, index=False)


def url_is_valid(url: str) -> bool:
    '''
        Check if the URL passed is valid (i.e. its host and content properties can be requested)

        Parameters:
            url: Name of the URL

        Returns:
            result: True if URL is valid, False if not
    '''
    return get_host(url) is not None and get_content(url) is not None


def config_is_valid(model_name: str, dataset_name: str, feature_index: int) -> bool:
    '''
        Check if the model requested is valid including the dataset and features

        Parameters:
            model_name: Name of the model
            dataset_name: Name of the dataset
            feature_index: Index of the features

        Returns:
            result: True if configuration is valid, False if not
    '''
    models_dict = load_json_as_dict(os.path.join(MODEL_DATA_DIRECTORY, "results-dict.json"))

    if model_name in models_dict.keys():
        if dataset_name in models_dict[model_name]['featuresets'].keys():
            if feature_index in models_dict[model_name]['featuresets'][dataset_name]['indexes']:
                return True

    return False


def model_is_stored(filename: str) -> bool:
    '''
        Check if the model filename is located in the default models directory

        Parameters:
            filename: Name of the model file

        Returns:
            result: True if file name is present, False if not
    '''
    model_filenames = [s.split(".")[0] for s in os.listdir(MODEL_DATA_DIRECTORY)]
    return filename in model_filenames


def generate_config_filename(model_name: str, dataset_name: str, feature_index: str) -> str:
    '''
        Generate the filename of a configuration

        Parameters:
            model_name: Name of the model
            dataset_name: Name of the dataset
            feature_index: Index of the features

        Returns:
            result: File name of the configuration
    '''
    return f"{model_name}-{dataset_name}-{feature_index}"


def get_class(class_name: str) -> object:
    '''
        Get the class object corresponding to the string given

        For example, given a class_name of sklearn.svm.SVM, this will return the SVM object for instantiation

        Parameters:
            class_name: Name of the class

        Returns:
            class_object: Object version of class
    '''
    paths = class_name.split('.')
    path = ".".join(paths[:-1])
    name = paths[-1]

    return getattr(import_module(path), name)


def bag_of_words(features: pd.DataFrame, series: pd.Series, key: str, use_cache: bool = True) -> pd.DataFrame:
    '''
        Append the bag of words feature and return the original dataframe

        Parameters:
            features: Existing features dataframe
            series: Column of data to apply bag of words to
            key: Name of the vocab to use
            use_cache: Whether to use the cahced version

        Returns:
            features: Features dataframe with bag of words columns
    '''
    vocab = load_json_as_dict(os.path.join(FEATURES_DATA_DIRECTORY, "vocab-dict.json"))

    if use_cache and key in vocab:
        vectorizer = CountVectorizer(vocabulary=vocab[key], decode_error='ignore')
    else:
        vectorizer = CountVectorizer(decode_error="ignore")
        vectorizer.fit_transform(series)
        vocab[key] = {k: v.tolist() for k, v in vectorizer.vocabulary_.items()}
        save_dict_as_json(vocab, os.path.join(FEATURES_DATA_DIRECTORY, "vocab-dict.json"))

    bow_df = pd.DataFrame(vectorizer.transform(series).todense(), columns=vectorizer.get_feature_names())
    features = features.reset_index()
    features = pd.concat([features, bow_df], axis=1)
    return features.drop(['index'], axis=1)


def safe_division(a: int, b: int) -> float:
    '''
        Safely divide a by b, returning 0 in the event that b is 0

        Parameters:
            a: numerator
            b: denominator

        Returns:
            result: safely divided result
    '''
    return 0 if b == 0 else a / b
