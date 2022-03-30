import string
from collections import defaultdict
import numpy as np
import pandas as pd
from URLAnalyser.utils import bag_of_words


def label_count(url):
    return len(url.split('.'))

def average_label_length(url):
    return np.average(list(map(len, url.split('.'))))

def char_count(url, list):
    return sum(map(url.count, list))

def get_lexical(urls, index, vocab=defaultdict()):
    features = pd.DataFrame()

    if index == "0" or index == "1":
        features.insert(0, 'urlLength', urls['name'].apply(lambda x: len(x)), True)
    
    if index == '0' or index == '2':
        features.insert(0, 'labelCount', urls['name'].apply(lambda x: label_count(x)), True)

    if index == '0' or index == '3':
        features.insert(0, 'averageLabelLength', urls['name'].apply(lambda x: average_label_length(x)), True)

    if index == '0' or index == '4':
        features.insert(0, 'normalCharCount', urls['name'].apply(lambda x: char_count(x, string.ascii_letters)), True)

    if index == '0' or index == '5':
        features.insert(0, 'specialCharCount', urls['name'].apply(lambda x: char_count(x, './?=-_')), True)

    if index == '0' or index == '6':
        features.insert(0, 'numberCharCount', urls['name'].apply(lambda x: char_count(x, string.digits)), True)

    if index == '0' or index == '7':
        features = bag_of_words(features, urls['name'], vocab['lexical'])

    return features
