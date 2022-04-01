from URLAnalyser.log import Log
from URLAnalyser.utils import get_class
from URLAnalyser.utils import get_urls
from URLAnalyser.utils import split_urls
from URLAnalyser.utils import save_sklearn_model
from URLAnalyser.utils import load_sklearn_model
from URLAnalyser.models.training import tune_hyperparameters
from URLAnalyser.features.features import get_train_test_features


'''
    TODO:
    * Add static logger class to handle verboseness and types of messages
    * Normalise features before sending to training
    * Handle saving and loading of keras models automatically
    * Add testing to new functions
    * Make sure function comments are correct
    * Implement host and content data requests from old code
    * Maybe add a flag for how much data to use (default 5%)
'''


def load_data(dataset_name, feature_index):
    '''
        Load the data with the given configuration

        Parameters:
            dataset_name: The dataset to use in training
            feature_index: The features to use in training
        
        Returns
            x_train: The features used in training
            x_test: The features used in testing
            y_train: The labels used in training
            y_test: The labels used in testing
    '''
    # Load URLs and split into train and test set
    x_train, x_test, y_train, y_test = split_urls(get_urls(0.05))
    Log.success(f"Loaded urls from 'data/urls'.")

    # Extract the features
    x_train, x_test = get_train_test_features(dataset_name, x_train, x_test, feature_index)
    Log.success(f"Generated features for '{dataset_name}'.")

    # Normalise features
    # TODO
    # Log.info(f"Normalised features for '{dataset_name}'.")

    return x_train, x_test, y_train, y_test


def train_model(model_name, filename, x_train, y_train, model_results_dict):
    '''
        Train a model with the given configuration

        Parameters:
            model_name: The model to use in training
            filename: The file name of the model in storage
            x_train: The features used in training
            y_train: The labels used in training
            model_results_dict: Info on models
        
        Returns:
            model: The trained model
    '''
    # Instantiate object of model name
    class_object = get_class(model_results_dict[model_name]['class'])
    model = class_object()
    Log.success(f"Created model '{model_name}'.")

    # Tune hyperparameters
    model = tune_hyperparameters(model, model_results_dict[model_name]["hyperparameters"], x_train, y_train)
    Log.success(f"Tuned hyperparameters for '{model_name}'.")

    # Train the model
    model.fit(x_train, y_train)
    Log.success(f"Trained model '{model_name}'.")

    # Save the model
    save_sklearn_model(model, filename)
    Log.success(f"Saved model as '{filename}' to default folder.")

    return model

def load_model(filename):
    '''
        Load a model with the given configuration
    
        Parameters:
            filename: The file name of the model in storage
        
        Returns:
            model: The previously trained model
    '''
    try:
        model = load_sklearn_model(filename)
        Log.success(f"Loaded model from '{filename}'.")
    except:
        Log.error(f"Could not load model from '{filename}'.")
    
    return model

def test_model(model, x_test, y_test):
    '''
        Test a model in terms of f1-score and recall

        Parameters:
            model: The model to use in prediction
            x_test: The labels used in training
            y_test: The labels used in testing
        
        Returns:
            results: Results stored in a dict
    '''
    Log.success(f"Generated results for model.")
    return {
        "f1-score": 0,
        "recall": 0
    }

def predict_url(url, model):
    '''
        Predict a URL as malicious or benign using the given model

        Parameters:
            url: The URL name to test
            model: The model to use in prediction
        
        Returns:
            bool: Whether the URL is malicious 
    '''
    return False
