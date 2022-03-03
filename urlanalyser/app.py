from urlanalyser.common.utils import get_class
from urlanalyser.processing.load_data import get_data

def train_model(model_name, dataset_name, feature_index):
    '''
        Train a model with the given configuration

        Parameters:
            model_name: The model to use in training
            dataset_name: The dataset to use in training
            feature_index: The features to use in training
        
        Returns:
            model: The newly trained model
    '''
    # Extract module and path names from class name
    # e.g. sklearn.svm.SVC
    class_object  = get_class(model_name)

    # Initialise the model (default hyperparameters)
    model = class_object()

    # Initialise the data
    data = get_data(dataset_name, feature_index)

    # Extract the features

    # Split into training and testing

    # Tune hyperparameters

    # Train the model

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
