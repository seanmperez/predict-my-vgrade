import pickle
import numpy as np
import pandas as pd

def load_model(path: str = "../climber_modeling/models/xgboost.pkl"):
    """
    Loads a the climbing model
    """

    return pickle.load(open(path), 'rb')

def load_scaler(path: str = "../climber_modeling/models/std_scaler"):
    """
    Loads the standard scaler.
    """

    return pickle.load(open(path), 'rb')

def standardize_input(scaler, input_array: np.array) -> np.array:
    """
    Takes a numpy array of  inputs and transforms standardizes them
    """

    return scaler.transform(input_array)

def translate_grade(grade_id: float) -> int:
    """
    Translates a grade ID to a V-grade.
    """

    rounded_grade_id = round(grade_id)

    df = pd.read_csv("../climber_modeling/data/processed/grade_conversion.csv")

    vgrade = df.loc[(df["grade_id"] == rounded_grade_id),["usa_boulders"]].values

    return vgrade.item()

def predict(input_array: np.array) -> str:
    """
    Uses a numpy input array of climber metrics and returns their predicticted their maximum v-grade.
    """
    model = load_model()
    scaler = load_scaler()
    std_input = standardize_input(scaler, input_array)
    raw_prediction = model.predict(std_input)

    return translate_grade(raw_prediction)