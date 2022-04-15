import os
from URLAnalyser.log import Log
from URLAnalyser.utils import get_class
from URLAnalyser.utils import generate_model_filename, load_json_as_dict
from URLAnalyser.utils import url_is_valid
from URLAnalyser.utils import model_is_valid
from URLAnalyser.utils import model_is_stored
from URLAnalyser.data.data import load_url_data
from URLAnalyser.data.data import get_train_test_data
from URLAnalyser.models.keras import save_model as save_keras
from URLAnalyser.models.keras import load_model as load_keras
from URLAnalyser.models.sklearn import save_model as save_sklearn
from URLAnalyser.models.sklearn import load_model as load_sklearn
from URLAnalyser.models.training import tune_hyperparameters
from URLAnalyser.features.features import get_url_features, scale_features
from URLAnalyser.features.features import get_train_test_features
from URLAnalyser.models.testing import generate_predictions
from URLAnalyser.models.testing import calculate_metrics


# App constants
DATA_DIRECTORY = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data")
MODELS_DIRECTORY = os.path.join(os.path.dirname(os.path.realpath(__file__)), "models")


def load_data(dataset_name, feature_index, sample_size, use_cache, is_keras):
    '''
        Load the data with the given configuration

        Parameters:
            dataset_name: The dataset to use in training
            feature_index: The features to use in training
            sample_size: The size of the sample of data
            use_cache: Use cached data if available
            is_keras: Whether the model uses Keras

        Returns
            x_train: The features used in training
            x_test: The features used in testing
            y_train: The labels used in training
            y_test: The labels used in testing
    '''
    # Load URLs and split into train and test set
    url_data = load_url_data(dataset_name, sample_size, use_cache, is_keras)
    x_train, x_test, y_train, y_test = get_train_test_data(url_data)
    Log.success(f"Loaded url data for '{dataset_name}'.")

    # Extract the features
    x_train, x_test = get_train_test_features(x_train, x_test, dataset_name, feature_index)
    Log.success(f"Generated features for '{dataset_name}'.")

    # Normalise features
    x_train, x_test = scale_features(x_train, x_test)
    Log.success(f"Normalised features for '{dataset_name}'.")

    return x_train, x_test, y_train, y_test


def train_model(model_name, filename, x_train, y_train, models_dict):
    '''
        Train a model with the given configuration

        Parameters:
            model_name: The model to use in training
            filename: The file name of the model in storage
            x_train: The features used in training
            y_train: The labels used in training
            models_dict: Info on models

        Returns:
            model: The trained model
    '''
    # Instantiate object of model name
    class_object = get_class(models_dict[model_name]['class'])
    model = class_object()
    Log.success(f"Created model '{model_name}'.")

    # Tune hyperparameters
    model = tune_hyperparameters(model, models_dict[model_name]["hyperparameters"], x_train, y_train)
    Log.success(f"Tuned hyperparameters for '{model_name}'.")

    # Train the model
    model.fit(x_train, y_train)
    Log.success(f"Trained model '{model_name}'.")

    # Save the model
    save_method = save_keras if models_dict[model_name]["isKeras"] else save_sklearn
    save_method(model, filename)
    Log.success(f"Saved model as '{filename}' to default folder.")

    return model


def load_model(filename, is_keras):
    '''
        Load a model with the given configuration

        Parameters:
            filename: The file name of the model in storage

        Returns:
            model: The previously trained model
    '''
    try:
        load_method = load_keras if is_keras else load_sklearn
        model = load_method(filename)
        Log.success(f"Loaded model from '{filename}'.")
    except BaseException:
        Log.error(f"Could not load model from '{filename}'.")

    return model


def test_model(model, is_keras, x_test, y_test):
    '''
        Test a model in terms of f1-score and recall

        Parameters:
            model: The model to use in prediction
            is_keras: Whether the model uses Keras
            x_test: The labels used in training
            y_test: The labels used in testing

        Returns:
            None
    '''
    predictions = generate_predictions(model, is_keras, x_test)
    Log.success("Generated predictions from tests.")

    results = calculate_metrics(predictions, y_test)
    Log.success("Generated results for model.")

    Log.result("The scoring metrics for the model are as follows:")
    for metric, value in results.items():
        Log.result(f"-> {metric} = {value}")


def test_url(url_name, dataset_name, feature_index, model, is_keras, features):
    '''
        Predict a URL as malicious or benign using the given model

        Parameters:
            model: The model to use in prediction
            is_keras: Whether the model uses Keras
            features: The features of the URL

        Returns:
            bool: Whether the URL is malicious
    '''
    if not url_is_valid(url_name):
        Log.error(f"Could not load url '{url_name}'.")

    features = get_url_features(url_name, dataset_name, feature_index, url_name)
    result = generate_predictions(model, is_keras, features)[0]

    result = "Benign" if result else "Malicious"
    Log.result(f"The url '{url_name}' is predicted to be {result}")
    return generate_predictions(model, is_keras, features)[0]


def main(args):
    '''
        Process the arguments and determine the top-level functions to execute

        Parameters:
            args: The arguments input from the user

        Returns:
            None
    '''
    # Check the required resources are available
    try:
        models_dict = load_json_as_dict(os.path.join(DATA_DIRECTORY, "models", "results-dict.json"))
    except BaseException:
        Log.error("Could not load 'results-dict.json' in 'data/models/'.")

    # Check the arguments passed are valid
    if not model_is_valid(args.model, args.data, args.feats, models_dict):
        Log.error(f"Could not load model '{args.model}' for data type '{args.data}' and feature index '{args.feats}'.")

    # Generate model filename and whether it is a Keras model
    model_filename = generate_model_filename(args.model, args.data, args.feats)
    model_is_keras = models_dict[args.model]["isKeras"]

    # Load data and features
    if args.train or not model_is_stored(model_filename) or args.url is None:
        x_train, x_test, y_train, y_test = load_data(args.data, args.feats, args.sample, args.cache, model_is_keras)

    # Train or load the model from storage
    if args.train:
        model = train_model(args.model, model_filename, x_train, y_train, models_dict)
    else:
        model = load_model(model_filename, model_is_keras)

    # Predict url
    if args.url is not None:
        result = "Benign" if test_url(args.url, args.data, args.feats, model, model_is_keras) else "Malicious"
        Log.result(f"The url '{args.url}' is predicted to be {result}")
    # Else test model
    else:
        test_model(model, model_is_keras, x_test, y_test)
