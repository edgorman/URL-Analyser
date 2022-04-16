import os
import pandas as pd

from URLAnalyser.constants import MODEL_DATA_DIRECTORY
from URLAnalyser.log import Log
from URLAnalyser.utils import get_class
from URLAnalyser.utils import generate_config_filename
from URLAnalyser.utils import load_json_as_dict
from URLAnalyser.utils import url_is_valid
from URLAnalyser.utils import config_is_valid
from URLAnalyser.utils import model_is_stored
from URLAnalyser.data.data import get_url_data
from URLAnalyser.data.data import get_train_test_data
from URLAnalyser.features.features import get_url_features
from URLAnalyser.features.features import get_train_features
from URLAnalyser.features.features import scale_features
from URLAnalyser.models.keras import save_model as save_keras
from URLAnalyser.models.keras import load_model as load_keras
from URLAnalyser.models.sklearn import save_model as save_sklearn
from URLAnalyser.models.sklearn import load_model as load_sklearn
from URLAnalyser.models.training import tune_hyperparameters
from URLAnalyser.models.testing import generate_predictions
from URLAnalyser.models.testing import calculate_metrics


def load_data(dataset_name: str, feature_index: int, sample_size: float, use_cache: bool, model_info: dict) -> tuple:
    '''
        Load the data with the given configuration

        Parameters:
            dataset_name: Name of dataset to use in training
            feature_index: Index of features to use in training
            sample_size: Size of the sample of data
            use_cache: Use cached data if available
            model_info: Info on the chosen model

        Returns
            x_train: The features used in training
            x_test: The features used in testing
            y_train: The labels used in training
            y_test: The labels used in testing
    '''
    # Load URLs and split into train and test set
    Log.info(f"Loading url data for '{dataset_name}'.")
    url_data = get_train_features(dataset_name, sample_size, use_cache, model_info["isKeras"])
    x_train, x_test, y_train, y_test = get_train_test_data(url_data)
    Log.success(f"Loaded url data for '{dataset_name}'.")

    # Extract the features
    Log.info(f"Generating features for '{dataset_name}'.")
    x_train, x_test = get_train_features(x_train, x_test, dataset_name, feature_index)
    Log.success(f"Generated features for '{dataset_name}'.")

    # Normalise features
    Log.info(f"Normalising features for '{dataset_name}'.")
    x_train, x_test = scale_features(x_train, x_test)
    Log.success(f"Normalised features for '{dataset_name}'.")

    return x_train, x_test, y_train, y_test


def train_model(x_train: pd.Dataframe, y_train: pd.Series, filename: str, model_info: dict) -> object:
    '''
        Train a model with the given configuration

        Parameters:
            x_train: The features used in training
            y_train: The labels used in training
            filename: The file name of the model in storage
            model_info: Info on the chosen model

        Returns:
            model: The newly trained model
    '''
    # Instantiate object of model name
    Log.info(f"Creating model '{model_info['class']}'.")
    model = get_class(model_info['class'])()
    Log.success(f"Created model '{model_info['class']}'.")

    # Tune hyperparameters
    Log.info(f"Tuning hyperparameters for '{model_info['class']}'.")
    model = tune_hyperparameters(model, model_info["hyperparameters"], x_train, y_train)
    Log.success(f"Tuned hyperparameters for '{model_info['class']}'.")

    # Train the model
    Log.info(f"Training model '{model_info['class']}'.")
    model.fit(x_train, y_train)
    Log.success(f"Trained model '{model_info['class']}'.")

    # Save the model
    Log.info(f"Saving model as '{filename}' to default folder.")
    save_method = save_keras if model_info["isKeras"] else save_sklearn
    save_method(model, filename)
    Log.success(f"Saved model as '{filename}' to default folder.")

    return model


def load_model(filename: str, model_info: dict) -> object:
    '''
        Load a model with the given configuration

        Parameters:
            filename: The file name of the model in storage
            model_info: Info on the chosen model

        Returns:
            model: The previously trained model
    '''
    try:
        Log.info(f"Loading model from '{filename}'.")
        load_method = load_keras if model_info["isKeras"] else load_sklearn
        model = load_method(filename)
        Log.success(f"Loaded model from '{filename}'.")
    except BaseException:
        Log.error(f"Could not load model from '{filename}'.")

    return model


def test_url(url_name: str, dataset_name: str, feature_index: int, model: object, model_info: dict) -> None:
    '''
        Predict an individual URL as malicious or benign using the given model

        Parameters:
            model: The model to use in prediction
            is_keras: Whether the model uses Keras
            features: The features of the URL

        Returns:
            None
    '''
    if dataset_name != 'lexical' and not url_is_valid(url_name):
        Log.error(f"Could not load url '{url_name}'.")

    Log.info(f"Loading data for url '{url_name}'.")
    data = get_url_data(url_name, dataset_name, model_info["isKeras"])
    Log.success(f"Loaded data for url '{url_name}'.")

    Log.info(f"Loading features for url '{url_name}'.")
    features = get_url_features(data, dataset_name, feature_index)
    Log.success(f"Loaded features for url '{url_name}'.")

    Log.info(f"Generating prediction for url '{url_name}'.")
    result = generate_predictions(model, features, model_info["isKeras"])[0]
    Log.success(f"Generated prediction for url '{url_name}'.")

    result = "Benign" if result else "Malicious"
    Log.result(f"The url '{url_name}' is predicted to be {result}")


def test_model(x_test: pd.DataFrame, y_test: pd.Series, model: object, model_info: dict) -> None:
    '''
        Test a model in terms of f1-score and recall

        Parameters:
            x_test: The features used in testing
            y_test: The labels used in testing
            model: The model to use in prediction
            model_info: Info on the chosen model

        Returns:
            None
    '''
    Log.info("Generating predictions from tests.")
    predictions = generate_predictions(model, x_test, model_info["isKeras"])
    Log.success("Generated predictions from tests.")

    Log.info("Generating metrics for model.")
    results = calculate_metrics(predictions, y_test)
    Log.success("Generated metrics for model.")

    Log.result("The scoring metrics for the model are as follows:")
    for metric, value in results.items():
        Log.result(f"-> {metric} = {value}")


def main(args: dict) -> None:
    '''
        Process the arguments and determine the top-level functions to execute

        Parameters:
            args: Dict of arguments input from the user

        Returns:
            None
    '''
    # Check the required resources are available
    try:
        models_dict = load_json_as_dict(os.path.join(MODEL_DATA_DIRECTORY, "results-dict.json"))
    except BaseException:
        Log.error("Could not load 'results-dict.json' in 'data/models/'.")

    # Check the arguments passed are valid
    if not config_is_valid(args.model, args.data, args.feats):
        Log.error(f"Could not load model '{args.model}' for data type '{args.data}' and feature index '{args.feats}'.")

    # Generate model filename and whether it is a Keras model
    model_filename = generate_config_filename(args.model, args.data, args.feats)
    model_info = models_dict[args.model]

    # Load data and features
    if args.train or not model_is_stored(model_filename) or args.url is None:
        x_train, x_test, y_train, y_test = load_data(args.data, args.feats, args.sample, args.cache, model_info)

    # Train or load the model from storage
    if args.train or not model_is_stored(model_filename):
        model = train_model(x_train, y_train, model_filename, models_dict[args.model])
    else:
        model = load_model(model_filename, model_info)

    # Predict url
    if args.url is not None:
        test_url(args.url, args.data, args.feats, model, model_info)
    # Else test model
    else:
        test_model(x_test, y_test, model, model_info)
