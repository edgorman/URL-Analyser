import re
from collections import defaultdict
import pandas as pd
from URLAnalyser.utils import bag_of_words
from URLAnalyser.utils import safe_division


def averageWordLength(content):
    return safe_division(sum(len(w) for w in content.split()), len(content.split()))

def averageJsLength(content):
    js_words = re.findall(r'<script\b[^>]*>([\s\S]*?)</script>', str(content))
    return safe_division(sum(len(w) for w in js_words), len(js_words))

def get_content(urls, index, vocab=defaultdict()):
    features = pd.DataFrame()

    if index == "0" or index == "1":
        features.insert(0, 'isRedirect', urls['isRedirect'], True)

    if index == "0" or index == "2":
        features = bag_of_words(features, urls['type'], vocab['doctype'])

    if index == "0" or index == "3":
        features.insert(0, 'length', urls['length'], True)

    if index == "0" or index == "4":
        features = bag_of_words(features, urls['content'], vocab['htmltag'])

    if index == "0" or index == "5":
        features.insert(0, 'averageTagLength', urls['content'].apply(lambda x: averageWordLength(x), True))

    if index == "0" or index == "6":
        features = bag_of_words(features, urls['content'], vocab['jstokens'])

    if index == "0" or index == "7":
        features.insert(0, 'averageJsLength', urls['content'].apply(lambda x: averageJsLength(x), True))

    return features