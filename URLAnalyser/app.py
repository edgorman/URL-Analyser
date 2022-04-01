from URLAnalyser.utils import get_class
from URLAnalyser.utils import get_urls
from URLAnalyser.utils import split_urls
from URLAnalyser.utils import save_model
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
    print(f"\tInfo: Successfully loaded urls")

    # Extract the features
    x_train, x_test = get_train_test_features(dataset_name, x_train, x_test, feature_index)
    print(f"\tInfo: Successfully generated features")

    # Normalise features
    # TODO
    print(f"\tInfo: Successfully normalised features")

    return x_train, x_test, y_train, y_test


def train_model(model_name, filename, x_train, y_train, models_dictionary):
    '''
        Train a model with the given configuration

        Parameters:
            model_name: The model to use in training
            filename: The file name of the model in storage
            x_train: The features used in training
            y_train: The labels used in training
            models_dictionary: Info on models
        
        Returns:
            model: The trained model
    '''
    # Instantiate object of model name
    class_object = get_class(models_dictionary[model_name]['class'])
    model = class_object()
    print(f"\tInfo: Successfully created model {model_name}")

    # Tune hyperparameters
    model = tune_hyperparameters(model, models_dictionary[model_name]["hyperparameters"], x_train, y_train)
    print(f"\tInfo: Successfully tuned hyperparameters for {model_name}")

    # Train the model
    model.fit(x_train, y_train)
    print(f"\tInfo: Successfully trained model {model_name}")

    # Save the model
    save_model(model, filename)
    print(f"\tInfo: Successfully saved model to default folder")

    return model

def load_model(model_name, dataset_name, feature_index):
    '''
        Load a model with the given configuration
    
        Parameters:
            model_name: The model to load from storage
            dataset_name: The dataset to load from storage
            feature_index: The features to load from storage
        
        Returns:
            model: The previously trained model
    '''
    return None

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
