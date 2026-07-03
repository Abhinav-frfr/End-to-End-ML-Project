import os 
import sys
import numpy as np
import pandas as pd
import dill

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