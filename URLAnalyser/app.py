from URLAnalyser.utils import get_class
from URLAnalyser.utils import get_urls
from URLAnalyser.utils import split_urls
from URLAnalyser.utils import get_train_method
from URLAnalyser.models.training import tune_hyperparameters
from URLAnalyser.features.features import get_train_test_features


def train_model(model_name, dataset_name, feature_index, models_dictionary):
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
    model = class_object()
    print(f"Info: Successfully loaded model {model_name}")

    # Load URLs and split into train and test set
    x_train, x_test, y_train, y_test = split_urls(get_urls())
    print(f"Info: Successfully loaded URLs")

    # Extract the features
    x_train, x_test = get_train_test_features(dataset_name, x_train, x_test, feature_index)
    print(f"Info: Successfully generated features")

    # Tune hyperparameters
    model = tune_hyperparameters(model, models_dictionary[model_name]["hyperparamters"], x_train, y_train)
    print(f"Info: Successfully tuned hyperparameters for {model_name}")
    print(model.best_params_)

    # Train the model
    train_method = get_train_method(model_name)
    model = train_method(model, x_train, y_train)
    print(f"Info: Successfully trained model {model_name}")

    # Save the model

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
