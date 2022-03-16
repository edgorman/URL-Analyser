import pandas as pd

from URLAnalyser.common.utils import bagOfWords
from URLAnalyser.common.utils import safe_division


def averageHTMLTagLength(content):
    return safe_division(sum(len(w) for w in content.split()), len(content.split()))

def get_content(urls, index):
    features = pd.DataFrame()

    if index == "0" or index == "1":
        features.insert(0, 'redirect', urls['redirect'], True)

    if index == "0" or index == "2":
        features = bagOfWords(features, urls['type'], []) # TODO use doctype vocab

    if index == "0" or index == "3":
        features.insert(0, 'length', urls['length'], True)

    if index == "0" or index == "4":
        features = bagOfWords(features, urls['content'], []) # TODO use htmltag vocab

    if index == "0" or index == "5":
        urls['content'] = urls['content'].fillna(" ")
        features.insert(0, 'htmlavglength',
                        urls['content'].apply(lambda x: safe_division(sum(len(w) for w in x.split()), len(x.split())),
                                            True))
        features['htmlavglength'] = features['htmlavglength'].fillna(0)

    if index == "0" or index == "6":
        urls['content'] = urls['content'].fillna(" ")
        js_vec = generate_bow_vectorizer(JSFUNC_VOCAB)
        js_df = pd.DataFrame(js_vec.transform(urls['content']).todense(), columns=js_vec.get_feature_names())
        features = features.reset_index()
        features = pd.concat([features, js_df], axis=1)
        features = features.drop(['index'], axis=1)

    if index == "0" or index == "7":
        urls['content'] = urls['content'].fillna(" ")
        features.insert(0, 'jsavglength',
                        urls['content'].apply(lambda x: safe_division(
                            sum(len(w) for w in re.findall(r'<script\b[^>]*>([\s\S]*?)</script>', str(x))),
                            len(re.findall(r'<script\b[^>]*>([\s\S]*?)</script>', str(x)))), True))
        features['jsavglength'] = features['jsavglength'].fillna(0)

    return features