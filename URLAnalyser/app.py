from URLAnalyser.log import Log
from URLAnalyser.utils import get_class
from URLAnalyser.data.data import load_url_data
from URLAnalyser.data.data import get_train_test_data
from URLAnalyser.models.keras import save_model as save_keras
from URLAnalyser.models.keras import load_model as load_keras
from URLAnalyser.models.sklearn import save_model as save_sklearn
from URLAnalyser.models.sklearn import load_model as load_sklearn
from URLAnalyser.models.training import tune_hyperparameters
from URLAnalyser.features.features import scale_features
from URLAnalyser.features.features import get_train_test_features
from URLAnalyser.models.testing import generate_predictions
from URLAnalyser.models.testing import calculate_metrics


def load_data(dataset_name, feature_index, sample_size, use_cache):
    '''
        Load the data with the given configuration

        Parameters:
            dataset_name: The dataset to use in training
            feature_index: The features to use in training
            sample_size: The size of the sample of data
            use_cache: Use cached data if available

        Returns
            x_train: The features used in training
            x_test: The features used in testing
            y_train: The labels used in training
            y_test: The labels used in testing
    '''
    # Load URLs and split into train and test set
    url_data = load_url_data(dataset_name, sample_size, use_cache)
    x_train, x_test, y_train, y_test = get_train_test_data(url_data)
    Log.success(f"Loaded url data for '{dataset_name}'.")

    # Extract the features
    x_train, x_test = get_train_test_features(
        x_train, x_test, dataset_name, feature_index)
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
    model = tune_hyperparameters(
        model,
        models_dict[model_name]["hyperparameters"],
        x_train,
        y_train)
    Log.success(f"Tuned hyperparameters for '{model_name}'.")

    # Train the model
    model.fit(x_train, y_train)
    Log.success(f"Trained model '{model_name}'.")

    # Save the model
    save_method = save_keras if models_dict[model_name]["isKeras"] else save_sklearn
    save_method(model, filename)
    Log.success(f"Saved model as '{filename}' to default folder.")

    return model


def load_model(filename, isKeras):
    '''
        Load a model with the given configuration

        Parameters:
            filename: The file name of the model in storage

        Returns:
            model: The previously trained model
    '''
    try:
        load_method = load_keras if isKeras else load_sklearn
        model = load_method(filename)
        Log.success(f"Loaded model from '{filename}'.")
    except BaseException:
        Log.error(f"Could not load model from '{filename}'.")

    return model


def test_model(model, isKeras, x_test, y_test):
    '''
        Test a model in terms of f1-score and recall

        Parameters:
            model: The model to use in prediction
            isKeras: Whether the model uses Keras
            x_test: The labels used in training
            y_test: The labels used in testing

        Returns:
            results: Results stored in a dict
    '''
    predictions = generate_predictions(model, isKeras, x_test)
    Log.success("Generated predictions from tests.")

    results = calculate_metrics(predictions, y_test)
    Log.success("Generated results for model.")

    return results


def predict_url(model, isKeras, url):
    '''
        Predict a URL as malicious or benign using the given model

        Parameters:
            model: The model to use in prediction
            isKeras: Whether the model uses Keras
            url: The URL name to test

        Returns:
            bool: Whether the URL is malicious
    '''
    return False
