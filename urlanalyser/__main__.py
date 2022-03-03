import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
import sys
import argparse

from urlanalyser import app
from urlanalyser.common.utils import load_json_as_dict
from urlanalyser.common.utils import is_valid_url
from urlanalyser.common.utils import is_valid_model
from urlanalyser.common.utils import is_model_stored

# App constants
DATA_DIRECTORY = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data")
MODELS_DIRECTORY = os.path.join(os.path.dirname(os.path.realpath(__file__)), "models")

if __name__ == '__main__':
    ''' 
        This script processes input from user and decides which top-level function to run in app.py.
    '''
    # Parse input arguments
    parser = argparse.ArgumentParser(prog="urlanalyser", description="Predict whether a URL is malicious using machine learning.")
    parser.add_argument('-u', action='store', dest='url', help="url to test")
    parser.add_argument('-m', action='store', dest='model', help="model to use", default='rf')
    parser.add_argument('-d', action='store', dest='data', help="data to use", default='content')
    parser.add_argument('-f', action='store', dest='feats', help="features to use", default='0')
    parser.add_argument('-train', action='store_true', dest='train', help="train this model's parameters (default: false)", default=False)
    parser.add_argument('-verbose', action='store_true', dest='verbose', help="show extra information while running (default: false)", default=False)
    parser.add_argument('-version', action='version', version='%(prog)s@dev')

    # Note: Help and Version command are handled by argparse
    args = parser.parse_args(sys.argv[1:])
    
    # Handle verbose-ness
    stdout = sys.stdout
    if not args.verbose:
        sys.stdout = open(os.devnull, 'w')
    
    # Load model and feature dicts
    try:
        models_dictionary = load_json_as_dict(os.path.join(DATA_DIRECTORY, "model-results.json"))
        features_dictionary = load_json_as_dict(os.path.join(DATA_DIRECTORY, "feature-sets.json"))
    except:
        print(f"Error: Could not load 'data/model-results.json' or 'data/feature-sets.json'.")
        exit(-1)

    # Validate chosen settings for model
    if is_valid_model(models_dictionary, args.model, args.data, args.feats):
        # Train model
        if args.train or not is_model_stored(args.model, args.data, args.feats):
            print(f"Info: Training '{args.model}' for data type '{args.data}' and feature index '{args.feats}'.")
            model = app.train_model(args.model, args.data, args.feats, models_dictionary)
        # Load model
        else:
            print(f"Info: Loading '{args.model}' for data type '{args.data}' and feature index '{args.feats}'.")
            model = app.load_model(args.model, args.data, args.feats)
        
        # Predict url
        if args.url is not None:
            # If url is valid
            if is_valid_url(args.url):
                print(f"Info: Predicting url '{args.url}'.")
                is_malicious = app.predict_url(args.url, model)

                sys.stdout = stdout
                result = "Malicious" if is_malicious else "Benign"
                print(f"The url '{args.url}' is predicted to be {result}")
            else:
                print(f"Error: Could not load url '{args.url}'.")
                exit(-1)
    else:
        print(f"Error: Could not load model '{args.model}' for data type '{args.data}' and feature index '{args.feats}'.")
        exit(-1)

    