from collections import defaultdict
import pandas as pd
from URLAnalyser.utils import bag_of_words


def get_host(urls, index, vocab=defaultdict()):
    features = pd.DataFrame()

    if index == '0' or index == '1':
        features = bag_of_words(features, urls['location'], vocab['location'])
    
    if index == '0' or index == '2':
        features.insert(0, 'server_count', urls['server_count'], True)

    if index == '0' or index == '3':
        features = bag_of_words(features, urls['registrar'], vocab['lexical'])

    if index == '0' or index == '4':
        features.insert(0, 'creation_month', urls['creation_date'].month, True)
        features.insert(0, 'creation_year', urls['creation_date'].year, True)

    if index == '0' or index == '5':
        features.insert(0, 'updated_month', urls['updated_date'].month, True)
        features.insert(0, 'updated_year', urls['updated_date'].year, True)

    if index == '0' or index == '6':
        features.insert(0, 'expiration_month', urls['expiration_date'].month, True)
        features.insert(0, 'expiration_year', urls['expiration_date'].year, True)

    if index == '0' or index == '7':
        features.insert(0, 'speed', urls['speed'], True)
        features.insert(0, 'latency', urls['latency'], True)

    return features
