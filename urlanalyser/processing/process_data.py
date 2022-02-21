"""
    process_data.py - Edward Gorman - eg6g17@soton.ac.uk
"""
import re
import math
import string
import argparse
import numpy as np
import talos as ta
import pandas as pd
from keras_preprocessing.sequence import pad_sequences
from keras_preprocessing.text import Tokenizer
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score, confusion_matrix
from sklearn.model_selection import train_test_split, RandomizedSearchCV
import matplotlib.pyplot as plt


# Get conditions of python call:
def get_conditions(args, params):
    # Parse arguments
    parser = argparse.ArgumentParser(prog="urlanalyser",
                                     description="Analyse a URL using a ML model.")
    parser.add_argument('-u', action='store', dest='url',
                        help='website to test')
    parser.add_argument('-m', default='rf', action='store', dest='model',
                        help='model to be run')
    parser.add_argument('-d', default='content', action='store', dest='data',
                        help='data to be used')
    parser.add_argument('-f', default='all', action='store', dest='feats',
                        help='features to try')
    parser.add_argument('-save', action='store_true', default=False, dest='save',
                        help='save model params (default: false)')
    parser.add_argument('-refine', action='store_true', default=False, dest='refine',
                        help='find optimal params (default: false)')
    parser.add_argument('-verbose', action='store_true', default=False, dest='verbose',
                        help='control info messages (default: false)')
    parser.add_argument('-version', action='version', version='%(prog)s dev')

    # Check program conditions
    args = parser.parse_args(args)

    # Check against program params
    if not check_conditions(params, args.url, args.model, args.data, args.feats):
        exit(-1)

    return args.url, args.model, args.data, args.feats, args.save, args.refine, args.verbose


# Return true if successful
def check_conditions(params, url, model, data, feat):
    # Check url exists
    if url is not None and valid_url(url) is None:
        print("Error: URL can not be accessed.")
        return False

    # Check model
    try:
        params[model]
    except KeyError:
        print("Error: Unknown model; received", model)
        return False

    # Check data
    try:
        params[model][data]
    except KeyError:
        print("Error: Unknown data; received", data)
        return False

    # Check feat_name
    try:
        params[model][data][feat]
    except KeyError:
        print("Error: Unknown feature; received", feat)
        return False

    return True


# Safe division
def safe_division(n, d):
    return n / d if n is not 0 and d is not 0 else 0


# Reads data from a given file
def load_data(file):
    df = pd.read_csv(file, sep="\t", low_memory=False)
    return df


# Cleans data
def clean_data(df):
    df = df.drop_duplicates()
    df = df.dropna()
    # df = df.sample(frac=1)
    return df


# Selects a random set of samples depending on limit
def normalise_data(df, sample_limit):
    df_mal = df[df['label'] == 1]
    df_ben = df[df['label'] == 0]

    if len(df_mal) > sample_limit:
        df_mal = resample(df_mal, replace=False, n_samples=math.floor(sample_limit / 2), random_state=123)

    if len(df_ben) > sample_limit:
        df_ben = resample(df_ben, replace=False, n_samples=math.floor(sample_limit / 2), random_state=123)

    return pd.concat([df_mal, df_ben])


# Generate training and testing data
def split_data(df):
    y = df['label']
    x = df.drop(['label'], axis=1)

    return train_test_split(x, y, test_size=0.2)


# Preprocess data using sklearn
def preprocess_data(df):
    return preprocessing.scale(df)


# Preprocess labels using sklearn
def preprocess_labels(labels):
    encoder = LabelEncoder()
    encoder.fit(labels)
    return encoder.transform(labels)


# Get top n most common values
def generate_topn_values(df, max_count, min_value=1):
    df = df[df >= min_value].index
    df = df.value_counts()[:max_count]
    return df


# Generate bow vocab from training data
def generate_bow_vectorizer(vocab):
    return CountVectorizer(vocabulary=vocab, decode_error='ignore')


# Generate tokenizer for training data
def generate_tokenizer(vocab, **params):
    tokenizer = Tokenizer(filters=params['filters'], char_level=params['char_level'], lower=True)
    tokenizer.word_index = vocab.copy()
    tokenizer.word_index[tokenizer.oov_token] = max(vocab.values()) + 1
    return tokenizer


# Extract features for single url
def extract_url_features(url, model_name, data_name, feat_name):
    url_df = pd.DataFrame(data=[url], columns=["url"])

    if model_name == 'cnn':
        url = pd.DataFrame({'url': [url]})
        return extract_cnn_features(url, feat_name)
    elif data_name == 'lexical':
        return extract_lexical_features(url_df, feat_name)
    elif data_name == 'host':
        url_df = get_urls(url_df, 1)
        url_df = url_df[['location', 'registrar', 'servercount', 'creation_m', 'creation_y', 'updated_m', 'updated_y',
                         'expiration_m', 'expiration_y', 'latency']]
        return extract_host_features(url_df, feat_name)
    elif data_name == 'content':
        url_df = get_urls(url_df, 1)
        url_df = url_df[['redirect', 'type', 'length', 'content']]
        return extract_content_features(url_df, feat_name)


# Extract neural network features
def extract_cnn_features(df, feat_name):
    features = []
    char_tk = generate_tokenizer(URLCHAR_VOCAB, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', char_level=True)
    word_tk = generate_tokenizer(URLWORD_VOCAB, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', char_level=False)
    labl_tk = generate_tokenizer(URLLABL_VOCAB, filters='!"#$%&()*+,./:;<=>?@[\\]^`{|}~\t\n', char_level=False)

    if feat_name == 'all' or feat_name == '1':
        sequences = char_tk.texts_to_sequences(df['url'])
        sequences = pad_sequences(sequences, padding='post', maxlen=200)
        features.append(np.array(sequences, dtype='float32'))

    if feat_name == 'all' or feat_name == '2':
        sequences = word_tk.texts_to_sequences(df['url'])
        sequences = pad_sequences(sequences, padding='post', maxlen=200)
        features.append(np.array(sequences, dtype='float32'))

    if feat_name == 'all' or feat_name == '3':
        sequences = labl_tk.texts_to_sequences(df['url'])
        sequences = pad_sequences(sequences, padding='post', maxlen=200)
        features.append(np.array(sequences, dtype='float32'))

    return features


# Extract lexical features
def extract_lexical_features(df, feat_name):
    features = pd.DataFrame()

    if feat_name == 'all' or feat_name == '1':
        features.insert(0, 'urllength', df['url'].apply(lambda x: len(x)), True)

    if feat_name == 'all' or feat_name == '2':
        features.insert(0, 'numberoflabels', df['url'].apply(lambda x: len(x.split('.'))), True)

    if feat_name == 'all' or feat_name == '3':
        features.insert(0, 'avglengthoflabels', df['url'].apply(lambda x: np.average(list(map(len, x.split('.'))))),
                        True)

    if feat_name == 'all' or feat_name == '4':
        features.insert(0, 'numberofnormchars', df['url'].apply(lambda x: sum(map(x.count, string.ascii_letters))),
                        True)

    if feat_name == 'all' or feat_name == '5':
        features.insert(0, 'numberofspecchars',
                        df['url'].apply(lambda x: sum(map(x.count, ['.', '/', '?', '=', '-', '_']))), True)

    if feat_name == 'all' or feat_name == '6':
        features.insert(0, 'numberofnumbchars', df['url'].apply(
            lambda x: sum(map(x.count, ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']))), True)

    if feat_name == 'all' or feat_name == '7':
        df['url'] = df['url'].fillna("")
        bow_vec = generate_bow_vectorizer(LEXICAL_VOCAB)
        bow_df = pd.DataFrame(bow_vec.transform(df['url']).todense(), columns=bow_vec.get_feature_names())
        features = features.reset_index()
        features = pd.concat([features, bow_df], axis=1)
        features = features.drop(['index'], axis=1)

    return features


# Extract host features
def extract_host_features(df, feat_name):
    features = pd.DataFrame()

    if feat_name == 'all' or feat_name == '1':
        df['location'] = df['location'].fillna("na")
        loc_vec = generate_bow_vectorizer(LOCATION_VOCAB)
        loc_df = pd.DataFrame(loc_vec.transform(df['location']).todense(), columns=loc_vec.get_feature_names())
        features = features.reset_index()
        features = pd.concat([features, loc_df], axis=1)
        features = features.drop(['index'], axis=1)

    if feat_name == 'all' or feat_name == '2':
        features['servercount'] = df['servercount']
        features['servercount'] = features['servercount'].fillna(-1)

    if feat_name == 'all' or feat_name == '3':
        df['registrar'] = df['registrar'].fillna("")
        reg_vec = generate_bow_vectorizer(LEXICAL_VOCAB)
        reg_df = pd.DataFrame(reg_vec.transform(df['registrar']).todense(), columns=reg_vec.get_feature_names())
        features = features.reset_index()
        features = pd.concat([features, reg_df], axis=1)
        features = features.drop(['index'], axis=1)

    if feat_name == 'all' or feat_name == '4':
        features['creation_m'] = df['creation_m']
        features['creation_y'] = df['creation_y']
        features['creation_m'] = features['creation_m'].fillna(-1)
        features['creation_y'] = features['creation_y'].fillna(-1)

    if feat_name == 'all' or feat_name == '5':
        features['updated_m'] = df['updated_m'].fillna(-1)
        features['updated_y'] = df['updated_y'].fillna(-1)
        features['updated_m'] = features['updated_m'].fillna(-1)
        features['updated_y'] = features['updated_y'].fillna(-1)

    if feat_name == 'all' or feat_name == '6':
        features['expiration_m'] = df['expiration_m'].fillna(-1)
        features['expiration_y'] = df['expiration_y'].fillna(-1)
        features['expiration_m'] = features['expiration_m'].fillna(-1)
        features['expiration_y'] = features['expiration_y'].fillna(-1)

    if feat_name == 'all' or feat_name == '7':
        features['latency'] = df['latency']
        features['latency'] = features['latency'].fillna(100000)

    return features


# Extract content features
def extract_content_features(df, feat_name):
    features = pd.DataFrame()

    if feat_name == 'all' or feat_name == '1':
        features.insert(0, 'redirect', df['redirect'].apply(lambda x: 1 if x else 0), True)

    if feat_name == 'all' or feat_name == '2':
        df['type'] = df['type'].fillna("na")
        doc_vec = generate_bow_vectorizer(DOCTYPE_VOCAB)
        doc_df = pd.DataFrame(doc_vec.transform(df['type']).todense(), columns=doc_vec.get_feature_names())
        features = features.reset_index()
        features = pd.concat([features, doc_df], axis=1)
        features = features.drop(['index'], axis=1)

    if feat_name == 'all' or feat_name == '3':
        features['length'] = df['length']
        features['length'] = features['length'].fillna(0)

    if feat_name == 'all' or feat_name == '4':
        df['content'] = df['content'].fillna(" ")
        html_vec = generate_bow_vectorizer(HTMLTAG_VOCAB)
        html_df = pd.DataFrame(html_vec.transform(df['content']).todense(), columns=html_vec.get_feature_names())
        features = features.reset_index()
        features = pd.concat([features, html_df], axis=1)
        features = features.drop(['index'], axis=1)

    if feat_name == 'all' or feat_name == '5':
        df['content'] = df['content'].fillna(" ")
        features.insert(0, 'htmlavglength',
                        df['content'].apply(lambda x: safe_division(sum(len(w) for w in x.split()), len(x.split())),
                                            True))
        features['htmlavglength'] = features['htmlavglength'].fillna(0)

    if feat_name == 'all' or feat_name == '6':
        df['content'] = df['content'].fillna(" ")
        js_vec = generate_bow_vectorizer(JSFUNC_VOCAB)
        js_df = pd.DataFrame(js_vec.transform(df['content']).todense(), columns=js_vec.get_feature_names())
        features = features.reset_index()
        features = pd.concat([features, js_df], axis=1)
        features = features.drop(['index'], axis=1)

    if feat_name == 'all' or feat_name == '7':
        df['content'] = df['content'].fillna(" ")
        features.insert(0, 'jsavglength',
                        df['content'].apply(lambda x: safe_division(
                            sum(len(w) for w in re.findall(r'<script\b[^>]*>([\s\S]*?)</script>', str(x))),
                            len(re.findall(r'<script\b[^>]*>([\s\S]*?)</script>', str(x)))), True))
        features['jsavglength'] = features['jsavglength'].fillna(0)

    return features


# Refine model
def refine_model(model, model_type, feat_name, x_train, y_train, x_test, y_test):
    test_params = {}

    if model_type == 'cnn':
        model_name = 'cnn-' + feat_name

        embedding_size = 0
        if feat_name is '1':
            embedding_size = len(URLCHAR_VOCAB)+2
        elif feat_name is '2':
            embedding_size = len(URLWORD_VOCAB)+2
        elif feat_name is '3':
            embedding_size = len(URLLABL_VOCAB) + 2

        test_params = {'first_activation': ['sigmoid', 'elu'],
                       'second_activation': ['sigmoid', 'elu'],
                       'dropout': [0.2, 0.5],
                       'first_neuron': [128, 256],
                       'second_neuron': [128, 256],
                       'embedding_size': [embedding_size]}

        scan = ta.Scan(x_train, y_train, x_val=x_test, y_val=y_test,
                       params=test_params,
                       model=refine_cnn,
                       experiment_name=model_name,
                       fraction_limit=0.1)

        results = scan.data[scan.data.recall_m == scan.data.recall_m.max()]
        best_params = {}
        for name, options in test_params.items():
            best_params[name] = results.iloc[0][name]
            print("\t", name, "=", options, " -> ", results.iloc[0][name], sep='')

        return build_cnn(feat_name, best_params)
    else:
        if model_type == 'svm':
            test_params = dict(C=[0.001, 0.01, 0.1, 1, 10],
                               kernel=['linear', 'poly', 'rbf'],
                               gamma=['scale', 'auto'],
                               degree=[1, 2, 3, 5, 10])
        elif model_type == 'rf':
            test_params = dict(n_estimators=[200, 600, 1000, 1400, 1800],
                               max_features=['sqrt', 'log2', None],
                               max_depth=[5, 10, 40, 70, 100, None],
                               min_samples_leaf=[1, 2, 4, 8],
                               min_samples_split=[2, 5, 10])
        elif model_type == 'pc':
            test_params = dict(penalty=['l2', 'l1', 'elasticnet', None],
                               alpha=[0.001, 0.001, 0.01, 0.1],
                               max_iter=[10, 100, 1000, 5000],
                               early_stopping=[False, True])

        classifier = RandomizedSearchCV(estimator=model, scoring=["f1", "recall"], param_distributions=test_params,
                                        cv=2, n_jobs=-1, verbose=3, refit="recall")
        classifier.fit(x_train, y_train)
        for name, options in test_params.items():
            print("\t", name, "=", options, " -> ", classifier.best_params_[name], sep='')

        return model.set_params(**classifier.best_params_)


# Train model
def train_model(model, feats, labels):
    model.fit(feats, labels)
    print("\tComplete")
    return model


# Test model
def test_model(model, model_name, feats, labels):
    preds = model.predict(feats)
    scores = preds

    if model_name == 'cnn':
        preds = (preds > 0.5).astype(np.int)

    acc = round(accuracy_score(labels, preds), 3) * 100
    pre = round(precision_score(labels, preds), 3)
    rec = round(recall_score(labels, preds), 3)
    f1s = round(f1_score(labels, preds, pos_label=1, average='binary'), 3)
    # print(confusion_matrix(labels, preds))

    return acc, pre, rec, f1s, preds, scores


# Plot model
def plot_graph(model, model_name, data_name, feat_name, x_train, y_train, x_test, y_test, y_preds, y_scores):
    # Scikit learn models
    # prefix = '../tmp/' + data_name + '-'
    #
    # Save file
    # d = plot_roc_curve(model, x_test, y_test)
    # with open(prefix + model_name + '.pkl', 'wb') as fid:
    #     pickle.dump(d.ax_, fid)
    # exit(1)
    #
    # # Load file
    # files = ['svm.pkl', 'rf.pkl', 'pc.pkl']
    # x_list, y_list, auc_list = [], [], []
    # for file in files:
    #     with open(prefix + file, 'rb') as fid:
    #         line = pickle.load(fid)
    #         data = line.figure.gca().get_lines()[0]
    #         legend = line.figure.gca().get_legend().get_texts()[0].get_text()
    #         auc = re.findall('\d*\.?\d+', legend)[0]
    #         x, y = data.get_data()
    #         x_list.append(x)
    #         y_list.append(y)
    #         auc_list.append(auc)
    #
    # plt.show()
    # plt.plot(x_list[0], y_list[0], label="SVM ( AUC="+str(auc_list[0])+" )")
    # plt.plot(x_list[1], y_list[1], label="RF ( AUC="+str(auc_list[1])+" )")
    # plt.plot(x_list[2], y_list[2], label="Pc ( AUC="+str(auc_list[2])+" )")
    # plt.plot([0, 1], [0, 1], color='black', linestyle='dashed')
    #
    # plt.title("ROC Curve - SL Algorithms on All Lexical Features")
    # plt.ylabel("True Positive Rate")
    # plt.xlabel("False Positive Rate")
    # plt.legend(loc="lower right")
    # plt.show()

    # Keras models
    prefix = '../tmp/' + model_name + '-'

    # Save file
    # fpr, tpr, thresholds = roc_curve(y_test, y_scores)
    # auc = roc_auc_score(y_test, y_preds)
    # df = pd.DataFrame(data={'fpr': fpr, 'tpr': tpr, 'auc': [auc] * len(fpr)})
    # df.to_csv(prefix + feat_name + '.csv', sep="\t")
    # exit(1)

    # Load file
    files = ['1.csv', '2.csv', '3.csv']
    x_list, y_list, auc_list = [], [], []
    for file in files:
        df = pd.read_csv(prefix + file, sep="\t")
        x_list.append(df['fpr'])
        y_list.append(df['tpr'])
        auc_list.append(df['auc'])

    plt.plot(x_list[0], y_list[0], label="Character ( AUC="+str(round(auc_list[0][0], 2))+" )")
    plt.plot(x_list[1], y_list[1], label="Word ( AUC="+str(round(auc_list[1][0], 2))+" )")
    plt.plot(x_list[2], y_list[2], label="Label ( AUC="+str(round(auc_list[2][0], 2))+" )")
    plt.plot([0, 1], [0, 1], color='black', linestyle='dashed')

    plt.title("ROC Curve - DL Algorithms on Lexical Input")
    plt.ylabel("True Positive Rate")
    plt.xlabel("False Positive Rate")
    plt.legend(loc="lower right")
    plt.show()


# Update model
def update_model(model, old_params):
    new_params = {}
    for param in old_params:
        new_params[param] = model.get_params()[param]

    return new_params
