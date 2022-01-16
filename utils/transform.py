import pickle
import numpy as np
import pandas as pd
import os

def lbs_to_kg(lb: float) -> float:
    """
    Converts weight in lbs to kg
    """

    return lb/2.205

def ft_to_cm(ft: int, inch: int) -> float:
    """
    Converts height in feet and inches to cm
    """

    ft_float = (ft) + (inch/12)
    return ft_float * 30.48

def inputs_to_array(input_data: dict) -> np.array:
    """
    Takes a dictionary of climber input values and returns the formatted numpy array

    The input format should be:
    input_data = {
        "age" : int,
        "feet" : int,
        "inches" : int,
        "age" : int,
        "weight" : float,
        "years_climbing" : int
    }
    """

    # Convert height to  
    height = ft_to_cm(input_data["feet"], input_data["inches"])
    
    raw_weight = input_data["weight"]["value"]

    if input_data["weight"]["kg"]:
        weight = raw_weight    
    else:
        weight = lbs_to_kg(raw_weight)
    
    input_array = np.array([[height, weight, input_data["years_climbing"], input_data["age"]]])

    return input_array
