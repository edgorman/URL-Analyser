from URLAnalyser.utils import generate_model_filename, load_json_as_dict
from URLAnalyser.utils import is_url_valid
from URLAnalyser.utils import is_model_valid
from URLAnalyser.utils import is_model_stored
from URLAnalyser.log import Log
from URLAnalyser import app
import os
import sys
import colorama
import argparse
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


# App constants
DATA_DIRECTORY = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data")
MODELS_DIRECTORY = os.path.join(os.path.dirname(os.path.realpath(__file__)), "models")

if __name__ == '__main__':
    '''
        This script processes input from user and decides which top-level function to run in app.py.
    '''
    # Initialise coloured text
    colorama.init(convert=True)

    # Parse input arguments
    parser = argparse.ArgumentParser(
        prog="urlanalyser",
        description="Predict whether a URL is malicious using machine learning.")
    parser.add_argument('-u', action='store', dest='url', help="url to test")
    parser.add_argument(
        '-m',
        action='store',
        dest='model',
        help="model to use",
        default='rf')
    parser.add_argument(
        '-d',
        action='store',
        dest='data',
        help="data to use",
        default='content')
    parser.add_argument(
        '-ds',
        action='store',
        dest='sample',
        help="sample size of data",
        default='0.001')
    parser.add_argument(
        '-f',
        action='store',
        dest='feats',
        help="features to use",
        default='0')
    parser.add_argument(
        '-train',
        action='store_true',
        dest='train',
        help="train this model's parameters (default: false)",
        default=False)
    parser.add_argument(
        '-no-cache',
        action='store',
        dest='cache',
        help="use cached data",
        default=True
    )
    parser.add_argument(
        '-verbose',
        action='store_true',
        dest='verbose',
        help="show extra information while running (default: false)",
        default=False)
    parser.add_argument('-version', action='version', version='%(prog)s@dev')

    # Note: Help and Version command are handled by argparse
    args = parser.parse_args(sys.argv[1:])

    # Handle verboseness
    if args.verbose:
        Log.verboseness = 1

    # Load model and feature dicts
    try:
        models_dict = load_json_as_dict(os.path.join(DATA_DIRECTORY, "models", "results-dict.json"))
        feat_index_dict = load_json_as_dict(os.path.join(DATA_DIRECTORY, "features", "index-dict.json"))
        model_filename = generate_model_filename(args.model, args.data, args.feats)
        model_is_keras = models_dict[args.model]["isKeras"]
    except BaseException:
        Log.error("Could not load either 'results-dict.json' or 'index-dict.json' in 'data/models/'.")

    # Validate chosen settings for model
    if is_model_valid(models_dict, args.model, args.data, args.feats):
        # Train model
        if args.train or not is_model_stored(args.model, args.data, args.feats):
            Log.info(f"Generating features for data type '{args.data}' and feature index '{args.feats}'.")
            x_train, x_test, y_train, y_test = app.load_data(
                args.data, args.feats, float(args.sample), args.cache, model_is_keras)
            Log.info(f"Training '{args.model}' for data type '{args.data}' and feature index '{args.feats}'.")
            model = app.train_model(args.model, model_filename, x_train, y_train, models_dict)
        # Load model
        else:
            Log.info(f"Loading '{args.model}' for data type '{args.data}' and feature index '{args.feats}'.")
            model = app.load_model(model_filename, model_is_keras)

        # Predict url
        if args.url is not None:
            # If url is valid
            if is_url_valid(args.url):
                Log.info(f"Generating features for data type '{args.data}' and feature index '{args.feats}'.")
                features = app.load_url(args.data, args.feats, args.url)

                Log.info(f"Predicting url '{args.url}'.")
                result = "Benign"
                if app.test_url(model, model_is_keras, features):
                    result = "Malicious"

                Log.result(f"The url '{args.url}' is predicted to be {result}")
            else:
                Log.error(f"Could not load url '{args.url}'.")
        # Test model
        else:
            Log.info(f"Testing '{args.model}' for data type '{args.data}' and feature index '{args.feats}'.")
            model_results = app.test_model(model, model_is_keras, x_test, y_test)

            Log.info("Outputting reuslts to terminal.")
            Log.result(f"The scoring metrics for '{args.model}' are as follows:")
            for metric, value in model_results.items():
                Log.result(f"-> {metric} = {value}")
    else:
        Log.error(f"Could not load model '{args.model}' for data type '{args.data}' and feature index '{args.feats}'.")
