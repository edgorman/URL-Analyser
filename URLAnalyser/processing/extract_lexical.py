import string
import numpy as np
import pandas as pd

from URLAnalyser.common.utils import bagOfWords


def urlLength(url):
    return len(url)

def labelCount(url):
    return len(url.split('.'))

def averageLabelLength(url):
    return np.average(list(map(len, url.split('.'))))

def normalCharCount(url):
    return sum(map(url.count, string.ascii_letters))

def specialCharCount(url):
    return sum(map(url.count, ['.', '/', '?', '=', '-', '_']))

def numberCharCount(url):
    return sum(map(url.count, [d for d in string.digits]))

def get_lexical(urls, index):
    features = pd.DataFrame()

    if index == "0" or index == "1":
        features.insert(0, 'urlLength', urls['name'].apply(lambda x: urlLength(x)), True)
    
    if index == '0' or index == '2':
        features.insert(0, 'labelCount', urls['name'].apply(lambda x: labelCount(x)), True)

    if index == '0' or index == '3':
        features.insert(0, 'averageLabelLength', urls['name'].apply(lambda x: averageLabelLength(x)), True)

    if index == '0' or index == '4':
        features.insert(0, 'normalCharCount', urls['name'].apply(lambda x: normalCharCount(x)), True)

    if index == '0' or index == '5':
        features.insert(0, 'specialCharCount', urls['name'].apply(lambda x: specialCharCount(x)), True)

    if index == '0' or index == '6':
        features.insert(0, 'numberCharCount', urls['name'].apply(lambda x: numberCharCount(x)), True)

    if index == '0' or index == '7':
        features = bagOfWords(features, urls['name'], []) # TODO use lexical vocab

    return features
