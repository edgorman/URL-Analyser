from URLAnalyser.utils import get_class
from URLAnalyser.features.lexical import get_lexical
from URLAnalyser.features.host import get_host
from URLAnalyser.features.content import get_content


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
    class_object = get_class(model_name)

    # Initialise the model (default hyperparameters)
    model = class_object()

    # Generate list of URLs
    url_list = []

    # Extract the features
    if dataset_name == 'lexical': data = get_lexical(url_list, feature_index)
    if dataset_name == 'host': data = get_host(url_list, feature_index)
    if dataset_name == 'content': data = get_content(url_list, feature_index)

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
