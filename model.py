import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

def predict_single_sample(sample):
    # Create a DataFrame from the sample data
    sample_df = pd.DataFrame([sample])

    # Load the saved model from the pickle file
    with open('random_forest_model.pkl', 'rb') as model_file:
        loaded_model = pickle.load(model_file)

    # Make predictions on the single sample
    prediction = loaded_model.predict(sample_df)

    return prediction

# Example usage
sample_data = {
    "Bp": 70,
    "Sg": 1.020,
    "Al": 1,
    "Su": 0,
    "Rbc": 1,
    "Bu": 94.0,
    "Sc": 7.3,
    "Sod": 137.00,
    "Pot": 4.30,
    "Hemo": 7.9,
    "Wbcc": 8406,
    "Rbcc": 4.71
}

result = predict_single_sample(sample_data)


