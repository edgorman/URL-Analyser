import os
import json

def load_json_as_dict(filename):
    with open(filename, 'r') as f:
        return json.load(f)

def is_valid_url(url):
    return True

def is_valid_model(models_dictionary, model_name, dataset_name, feature_index):
    if model_name in models_dictionary:
        if dataset_name in models_dictionary[model_name]:
            if feature_index in models_dictionary[model_name][dataset_name]['indexes']:
                return True
    return False

def generate_model_filename(model_name, dataset_name, feature_index):
    return model_name + "-" + dataset_name + "-" + feature_index

def is_model_stored(model_name, dataset_name, feature_index):
    filename = generate_model_filename(model_name, dataset_name, feature_index)
    model_path = os.path.join(os.path.realpath(__file__), "..", "data", "models")
    saved_models = [n.split(".")[0] for n in os.listdir(model_path)]

    return filename in saved_models
