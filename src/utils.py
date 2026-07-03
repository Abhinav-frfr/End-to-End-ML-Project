import os 
import sys
import numpy as np
import pandas as pd
import dill
from sklearn.metrics import r2_score

def save_object(file_path: str, obj: object) -> None:
    """
    Save an object to a file using pickle.

    Args:
        file_path (str): The path to the file where the object will be saved.
        obj (object): The object to be saved.

    Raises:
        Exception: If there is an error while saving the object.
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise Exception(f"Error occurred while saving object: {e}")
    
def evaluate_models(X, y, X_test, y_test, models):
    """
    Evaluate multiple machine learning models and return their performance scores.

    Args:
        X (array-like): Training features.
        y (array-like): Training labels.
        X_test (array-like): Testing features.
        y_test (array-like): Testing labels.
        models (dict): A dictionary of model names and their corresponding model instances.

    Returns:
        dict: A dictionary containing model names as keys and their performance scores as values.
    """
    try:
        model_report = {}
        for model_name, model in models.items():
            model.fit(X, y)
            y_test_pred = model.predict(X_test)
            test_model_score = r2_score(y_test, y_test_pred)
            model_report[model_name] = test_model_score

        return model_report

    except Exception as e:
        raise Exception(f"Error occurred while evaluating models: {e}")