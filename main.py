from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from model import predict_single_sample

app = FastAPI()

origins = ['*']

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

class SampleData(BaseModel):
    Bp: int
    Sg: float
    Al: int
    Su: int
    Rbc: int
    Bu: float
    Sc: float
    Sod: float
    Pot: float
    Hemo: float
    Wbcc: int
    Rbcc: float

@app.post('/answers')
def predict(sample: SampleData):
    try:
        result = predict_single_sample(sample.dict())
        print("Result:", result)  # Add this line to print the result for debugging

        if not result:
            raise HTTPException(status_code=400, detail="Prediction failed")

        return {"result": result}
    except Exception as e:
        error_message = f"An error occurred: {str(e)}"
        print(error_message)  # Log the error for debugging
        raise HTTPException(status_code=500, detail=error_message)
