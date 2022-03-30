import re
from collections import defaultdict
import pandas as pd
from URLAnalyser.utils import bag_of_words
from URLAnalyser.utils import safe_division


def average_word_length(words):
    return safe_division(sum(len(w) for w in words), len(words))

def average_js_length(words):
    js_words = re.findall(r'<script\b[^>]*>([\s\S]*?)</script>', ' '.join(words))
    if len(js_words) == 0:
        return 0
    else:
        return average_word_length(js_words[0].split(' '))

def get_content(urls, index, vocab=defaultdict()):
    features = pd.DataFrame()

    if index == "0" or index == "1":
        features.insert(0, 'isRedirect', urls['isRedirect'], True)

    if index == "0" or index == "2":
        features = bag_of_words(features, urls['type'], vocab['doctype'])

    if index == "0" or index == "3":
        features.insert(0, 'contentLength', urls['content'].apply(lambda x: len(x), True))

    if index == "0" or index == "4":
        features = bag_of_words(features, urls['content'], vocab['htmltag'])

    if index == "0" or index == "5":
        features.insert(0, 'averageWordLength', urls['content'].apply(lambda x: average_word_length(x.split()), True))

    if index == "0" or index == "6":
        features = bag_of_words(features, urls['content'], vocab['jstokens'])

    if index == "0" or index == "7":
        features.insert(0, 'averageJsLength', urls['content'].apply(lambda x: average_js_length(x), True))

    return features