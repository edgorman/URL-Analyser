import pandas as pd

from URLAnalyser.common.utils import bagOfWords


def get_host(urls, index):
    features = pd.DataFrame()

    if index == "0" or index == "1":
        bow_df = bagOfWords(urls['location'], []) # TODO use location vocab
        features = features.reset_index()
        features = pd.concat([features, bow_df], axis=1)
        features = features.drop(['index'], axis=1)
    
    if index == "0" or index == "2":
        features.insert(0, 'serverCount', urls['serverCount'], True)

    if index == "0" or index == "3":
        bow_df = bagOfWords(urls['registrar'], []) # TODO use lexical vocab
        features = features.reset_index()
        features = pd.concat([features, bow_df], axis=1)
        features = features.drop(['index'], axis=1)

    if index == "0" or index == "4":
        features.insert(0, 'creationMonth', urls['creationMonth'], True)
        features.insert(0, 'creationYear', urls['creationYear'], True)

    if index == "0" or index == "5":
        features.insert(0, 'updatedMonth', urls['updatedMonth'], True)
        features.insert(0, 'updatedYear', urls['updatedYear'], True)

    if index == "0" or index == "6":
        features.insert(0, 'expirationMonth', urls['expirationMonth'], True)
        features.insert(0, 'expirationYear', urls['expirationYear'], True)

    if index == "0" or index == "7":
        features.insert(0, 'latency', urls['latency'], True)

    return features
