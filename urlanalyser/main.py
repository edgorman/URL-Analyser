"""
    main.py - Edward Gorman - eg6g17@soton.ac.uk
"""
import os
import sys
import json
import joblib
import numpy as np
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron
from src import *


if __name__ == "__main__":

    # Open program parameters file
    with open(DATA_DIRECTORY + PARAMS_FILE) as json_file:
        prog_params = json.load(json_file)

    # Get run conditions from python arguments
    target_url, model_name, data_name, feat_name, save_flag, refine_flag, verbose_flag = \
        get_conditions(sys.argv[1:], prog_params)

    # If verbose off, disable print
    old_stdout = sys.stdout
    if not verbose_flag:
        sys.stdout = open(os.devnull, 'w')

    # Get model params and model file if exists
    model_config = None
    if model_name == 'cnn':
        model_config = prog_params[model_name][data_name][feat_name]["config"]
    model_params = prog_params[model_name][data_name][feat_name]["params"]
    model_filename = MODEL_DIRECTORY + model_name + "/" + data_name + "/" + model_name + "-" + data_name + "-" + feat_name
    model = load_model(model_name, model_filename, model_config)

    # Single URL prediction
    url_test = None
    if target_url is not None:
        # Generate features
        print("Extract features:")
        print("\t", prog_params[model_name][data_name][feat_name]["_name"], " from: ", target_url, sep='')
        url_test = extract_url_features(target_url, model_name, data_name, feat_name)

        # Preprocess features
        if model_name == 'svm' or model_name == 'rf' or model_name == 'pc':
            url_test = preprocess_data(url_test)

        if model is not None:
            # Predict
            result = model.predict(url_test)
            if model_name == 'cnn':
                result = (result > 0.5).astype(np.int)

            # Output
            sys.stdout = old_stdout
            if result[0] == 0:
                print("Benign")
            else:
                print("Malicious")
            exit(0)

    # Get data
    print("Loading", data_name, "data . . .")
    df = None
    if data_name == 'lexical':
        df = load_data(DATA_DIRECTORY + LEXICAL_FILE)
    elif data_name == 'host':
        df = load_data(DATA_DIRECTORY + HOST_FILE)
    elif data_name == 'content':
        df = load_data(DATA_DIRECTORY + CONTENT_FILE)

    # Set data_limit depending on model
    data_lim = 10000
    if model_name == 'cnn':
        data_lim = len(df)
        model_filename = model_filename + ".h5"
    else:
        model_filename = model_filename + ".pkl"

    # Clean data
    df = clean_data(df)
    df = normalise_data(df, data_lim)
    x_train, x_test, y_train, y_test = split_data(df)

    print("Extract features:")
    print("\t", prog_params[model_name][data_name][feat_name]["_name"], sep='')
    if model_name == 'cnn':
        x_test = extract_cnn_features(x_test, feat_name)
        x_train = extract_cnn_features(x_train, feat_name)
    elif data_name == 'lexical':
        x_test = extract_lexical_features(x_test, feat_name)
        x_train = extract_lexical_features(x_train, feat_name)
    elif data_name == 'host':
        x_test = extract_host_features(x_test, feat_name)
        x_train = extract_host_features(x_train, feat_name)
    elif data_name == 'content':
        x_test = extract_content_features(x_test, feat_name)
        x_train = extract_content_features(x_train, feat_name)

    # Preprocess features
    if model_name == 'svm':
        x_train = preprocess_data(x_train)
        x_test = preprocess_data(x_test)
    elif model_name == 'rf':
        x_train = preprocess_data(x_train)
        x_test = preprocess_data(x_test)
    elif model_name == 'pc':
        x_train = preprocess_data(x_train)
        x_test = preprocess_data(x_test)
    elif model_name == 'cnn':
        y_train = preprocess_labels(y_train)
        y_test = preprocess_labels(y_test)

    # Train model if required
    if model is None or refine_flag is True:
        # Set models
        if model_name == 'svm':
            model = svm.SVC(**model_params)
        elif model_name == 'rf':
            model = RandomForestClassifier(**model_params)
        elif model_name == 'pc':
            model = Perceptron(**model_params)
        elif model_name == 'cnn':
            model = build_cnn(feat_name, model_params)

        # Refine model
        if refine_flag is True:
            print("Refine model:")
            model = refine_model(model, model_name, feat_name, x_train, y_train, x_test, y_test)

        # Train model
        print("Training", model_name, "model:")
        model = train_model(model, x_train, y_train)

    # Single URL prediction
    if target_url is not None and model is not None:
        # Predict
        result = model.predict(url_test)
        if model_name == 'cnn':
            result = (result > 0.5).astype(np.int)

        # Output
        sys.stdout = old_stdout
        if result[0] == 0:
            print("Benign")
        else:
            print("Malicious")
        exit(0)

    print("Testing model:")
    acc, pre, rec, f1s, preds, scores = test_model(model, model_name, x_test, y_test)
    print("\tAccuracy:", acc)
    print("\tPrecision:", pre)
    print("\tRecall:", rec)
    print("\tF1 Score:", f1s)
    # plot_graph(model, model_name, data_name, feat_name, x_train, y_train, x_test, y_test, preds, scores)

    # Save model
    if save_flag:
        print("Saving model . . .")
        # Update performance metrics
        prog_params[model_name][data_name][feat_name]["accuracy"] = acc
        prog_params[model_name][data_name][feat_name]["precision"] = pre
        prog_params[model_name][data_name][feat_name]["recall"] = rec
        prog_params[model_name][data_name][feat_name]["f1score"] = f1s

        # Update model parameters
        if model_name == 'cnn':
            prog_params[model_name][data_name][feat_name]["params"] = {
                'first_activation': model.get_config()['layers'][15]['config']['activation'],
                'first_neuron': int(model.get_config()['layers'][15]['config']['units']),
                'dropout': model.get_config()['layers'][16]['config']['rate'],
                'second_activation': model.get_config()['layers'][17]['config']['activation'],
                'second_neuron': int(model.get_config()['layers'][17]['config']['units'])
            }
            prog_params[model_name][data_name][feat_name]["config"] = json.loads(model.to_json())
            model.save_weights(model_filename)
        else:
            prog_params[model_name][data_name][feat_name]["params"] = update_model(model, model_params.keys())
            joblib.dump(model, model_filename)

        # Save model parameters
        with open(DATA_DIRECTORY + PARAMS_FILE, 'w') as json_file:
            json.dump(prog_params, json_file)

        print("\tSaved", model_filename)
