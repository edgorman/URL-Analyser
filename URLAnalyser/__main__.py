import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from URLAnalyser.log import Log  # noqa: E402
from URLAnalyser import app  # noqa: E402
import sys  # noqa: E402
import colorama  # noqa: E402
import argparse  # noqa: E402


if __name__ == '__main__':
    '''
        This script processes input from user and runs the main application.
    '''
    # Initialise coloured text
    colorama.init(convert=True)

    # Parse input arguments
    parser = argparse.ArgumentParser(prog="URLAnalyser", description="Predict whether a URL is malicious.")
    parser.add_argument('-u', action='store', dest='url', help="url to test")
    parser.add_argument('-m', action='store', dest='model', help="model to use", default='rf')
    parser.add_argument('-d', action='store', dest='data', help="data to use", default='content')
    parser.add_argument('-f', action='store', dest='feats', help="features to use", default='0')
    parser.add_argument('-s', action='store', dest='sample', help="sample size", default='0.001')
    parser.add_argument('-cache', action='store_false', dest='cache', help="turn caching off", default=True)
    parser.add_argument('-train', action='store_true', dest='train', help="train model anew", default=False)
    parser.add_argument('-verbose', action='store_true', dest='verbose', help="show extra output", default=False)
    parser.add_argument('-version', action='version', version='%(prog)s@2.0')

    # Note: Help and Version command are handled by argparse
    args = parser.parse_args(sys.argv[1:])

    # Handle verboseness
    if args.verbose:
        Log.verboseness = 1

    # Handle other types
    args.feats = int(args.feats)
    args.sample = float(args.sample)

    # Process arguments and run module
    app.main(args)
