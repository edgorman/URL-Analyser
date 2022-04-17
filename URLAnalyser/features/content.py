import re
import pandas as pd

from URLAnalyser.utils import bag_of_words
from URLAnalyser.utils import safe_division


def average_word_length(words: list) -> float:
    return safe_division(sum(len(w) for w in words), len(words))


def average_js_length(words: list) -> float:
    js_words = re.findall(r'<script\b[^>]*>([\s\S]*?)</script>', ' '.join(words))
    return 0 if len(js_words) == 0 else average_word_length(js_words[0].split(' '))


def get_content(urls: pd.DataFrame, index: int, use_cache: bool = True) -> pd.DataFrame:
    features = pd.DataFrame()

    if index == 0 or index == 1:
        features.insert(0, 'isRedirect', urls['isRedirect'], True)

    if index == 0 or index == 2:
        features = bag_of_words(features, urls['type'], 'doctype', use_cache)

    if index == 0 or index == 3:
        features.insert(0, 'contentLength', urls['content'].apply(lambda x: len(x), True))

    if index == 0 or index == 4:
        features = bag_of_words(features, urls['content'], 'htmltag', use_cache)

    if index == 0 or index == 5:
        features.insert(0, 'averageWordLength', urls['content'].apply(lambda x: average_word_length(x.split()), True))

    if index == 0 or index == 6:
        features = bag_of_words(features, urls['content'], 'jstokens', use_cache)

    if index == 0 or index == 7:
        features.insert(0, 'averageJsLength', urls['content'].apply(lambda x: average_js_length(x), True))

    return features
