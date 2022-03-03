import pandas as pd

from urlanalyser.processing.extract_lexical import get_lexical
from urlanalyser.processing.extract_host import get_host
from urlanalyser.processing.extract_content import get_content


def get_urls():
    return pd.DataFrame()

def get_data(dataset_name, feature_index):
    url_dataframe = get_urls()

    if dataset_name == 'lexical':
        return get_lexical(url_dataframe, feature_index)
    if dataset_name == 'host':
        return get_host(url_dataframe, feature_index)
    if dataset_name == 'content':
        return get_content(url_dataframe, feature_index)
