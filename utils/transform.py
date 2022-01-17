from numpy import array

def lbs_to_kg(lb: float) -> float:
    """
    Converts weight in lbs to kg
    """

    return lb/2.205

def ft_to_cm(ft: int, inch: int) -> float:
    """
    Converts height in feet and inches to cm
    """

    ft_float = ft + inch/12
    return ft_float * 30.48

def inputs_to_array(input_data: dict) -> array:
    """
    Takes a dictionary of climber input values and returns the formatted numpy array

    The input format should be:
    input_data = {
        "age" : str, -> float
        "feet" : str, -> float
        "inches" : str, -> float
        "age" : str, -> float
        "weight" : str, -> float
        "years_climbing" : str, -> float
    }
    """
    # Convert dictionary values to floats
    input_numerical = {key : round(float(val), 1) for key, val in input_data.items()}

    height = ft_to_cm(input_numerical["feet"], input_numerical["inches"])
    
    weight = lbs_to_kg(input_numerical["weight"])
    
    model_inputs = array([[height, weight, input_numerical["years_climbing"], input_numerical["age"]]])

    return model_inputs
