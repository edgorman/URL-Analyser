import pandas as pd
from sklearn.preprocessing import StandardScaler

from URLAnalyser.features.lexical import get_lexical
from URLAnalyser.features.host import get_host
from URLAnalyser.features.content import get_content


def _get_method(dataset_name: str) -> object:
    '''
        Load the feature set for the corresponding dataset

        Parameters:
            dataset_name: Name of the chosen dataset

        Returns:
            method: Method for loading featureset for dataset
    '''
    if dataset_name == 'lexical':
        return get_lexical
    elif dataset_name == 'host':
        return get_host
    elif dataset_name == 'content':
        return get_content
    return None


def get_url_features(url_data: pd.DataFrame, dataset_name: str, feature_index: int) -> pd.DataFrame:
    '''
        Get the URL features for the dataset and feature index passed

        Parameters:
            url_data: Dataframe of URL
            dataset_name: Name of dataset for features
            feature_index: Index of features to extract

        Returns:
            features: Dataframe of URL features
    '''
    get_method = _get_method(dataset_name)
    url_features = get_method(url_data, feature_index, True)

    return url_features


def get_train_features(train: pd.DataFrame, test: pd.DataFrame, dataset_name: str, feature_index: int) -> pd.DataFrame:
    '''
        Get the train and test features for the dataset and feature index passed

        Parameters:
            train: Dataframe of training URLs
            test: Dataframe of testing URLs
            dataset_name: Name of dataset for features
            feature_index: Index of features to extract

        Returns:
            train_feats: Dataframe of training features
            test_feats: Dataframe of testing features
    '''
    get_method = _get_method(dataset_name)
    train_feats = get_method(train, feature_index, False)
    test_feats = get_method(test, feature_index, True)

    return train_feats, test_feats


def scale_url_features(features: pd.DataFrame):
    '''
        Scale the url features

        Parameters:
            features: Dataframe of URL features

        Returns:
            features: Scaled URL features
    '''
    scaler = StandardScaler().fit(features)
    features = scaler.transform(features)

    return features


def scale_train_features(train: pd.DataFrame, test: pd.DataFrame):
    '''
        Scale the train and test features

        Parameters:
            train: Dataframe of training URLs
            test: Dataframe of testing URLs

        Returns:
            train_feats: Scaled training features
            test_feats: Scaled testing features
    '''
    scaler = StandardScaler().fit(train)
    train = scaler.transform(train)
    test = scaler.transform(test)

    return train, test
