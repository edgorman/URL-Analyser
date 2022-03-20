from collections import defaultdict
import pandas as pd
from URLAnalyser.utils import bag_of_words


def get_host(urls, index, vocab=defaultdict()):
    features = pd.DataFrame()

    if index == '0' or index == '1':
        features = bag_of_words(features, urls['location'], vocab['location'])
    
    if index == '0' or index == '2':
        features.insert(0, 'serverCount', urls['serverCount'], True)

    if index == '0' or index == '3':
        features = bag_of_words(features, urls['registrar'], vocab['lexical'])

    if index == '0' or index == '4':
        features.insert(0, 'creationMonth', urls['creationMonth'], True)
        features.insert(0, 'creationYear', urls['creationYear'], True)

    if index == '0' or index == '5':
        features.insert(0, 'updatedMonth', urls['updatedMonth'], True)
        features.insert(0, 'updatedYear', urls['updatedYear'], True)

    if index == '0' or index == '6':
        features.insert(0, 'expirationMonth', urls['expirationMonth'], True)
        features.insert(0, 'expirationYear', urls['expirationYear'], True)

    if index == '0' or index == '7':
        features.insert(0, 'latency', urls['latency'], True)

    return features
