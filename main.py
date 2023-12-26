import pickle
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

class SampleData(BaseModel):
    Bp: str
    Sg: str
    Al: str
    Su: str
    Rbc: str
    Bu: str
    Sc: str
    Sod: str
    Pot: str
    Hemo: str
    Wbcc: str
    Rbcc: str

app = FastAPI()

def predict_single_sample(sample):
    # Create a DataFrame from the sample data
    sample_df = pd.DataFrame([sample])

    # Load the saved model from the pickle file
    with open('random_forest_model.pkl', 'rb') as model_file:
        loaded_model = pickle.load(model_file)

    # Make predictions on the single sample
    prediction = loaded_model.predict(sample_df)

    # Convert the prediction to a dictionary
    result_dict = {"prediction": int(prediction[0])}

    return result_dict

@app.post('/answers')
def predict(sample: SampleData):
    try:
        result = predict_single_sample(sample.dict())
        print("Result:", result)  # Add this line to print the result for debugging

        if not result:
            raise HTTPException(status_code=400, detail="Prediction failed")

        return JSONResponse(content=result)
    except Exception as e:
        error_message = f"An error occurred: {str(e)}"
        print(error_message)  # Log the error for debugging
        raise HTTPException(status_code=500, detail=error_message)
