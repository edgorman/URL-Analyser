import os
import sys
import argparse

from urlanalyser import app
from urlanalyser.processing import is_valid_url
from urlanalyser.processing import is_valid_model

if __name__ == '__main__':
    ''' 
        This script processes input from user and decides which top-level function to run in app.py.
    '''
    # Parse input arguments
    parser = argparse.ArgumentParser(prog="urlanalyser", description="Predict whether a URL is malicious using machine learning.")
    parser.add_argument('-u', action='store', dest='url', help="url to test")
    parser.add_argument('-m', action='store', dest='model', help="model to use", default='rf')
    parser.add_argument('-d', action='store', dest='data', help="data to use", default='content')
    parser.add_argument('-f', action='store', dest='feats', help="features to use", default='all')
    parser.add_argument('-save', action='store_true', dest='save', help="save this model's parameters (default: false)", default=False)
    parser.add_argument('-train', action='store_true', dest='train', help="train this model's parameters (default: false)", default=False)
    parser.add_argument('-verbose', action='store_true', dest='verbose', help="show extra information while running (default: false)", default=False)
    parser.add_argument('-version', action='version', version='%(prog)s@dev')

    # Note: Help and Version command are handled by argparse
    args = parser.parse_args(sys.argv[1:])
    if args.help or args.version:
        exit(0)
    
    # Handle verbose-ness
    stdout = sys.stdout
    if not args.verbose:
        sys.stdout = open(os.devnull, 'w')

    # Refine model
    if args.refine:
        if is_valid_model(args.model, args.data, args.feats):
            app.train_model(args.model, args.data, args.feats, args.save)

    # Predict url
    if args.url is not None:
        if is_valid_model(args.model, args.data, args.feats):
            if is_valid_url(args.url):
                sys.stdout = stdout
                is_malicious = app.predict_url(args.url, args.model, args.data, args.feats)
                print("Malicious") if is_malicious else print("Benign")
    