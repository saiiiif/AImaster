from fastapi import FastAPI
from joblib import load
from pydantic import BaseModel
import numpy as np

app = FastAPI()
model = load('random_forest_model.joblib')

class InputData(BaseModel):
    feature: list


@app.post("/predict/")
def predict(data: InputData):
    input_data = np.array(data.feature).reshape(1,-1)
    prediction = model.predict(input_data)
    return {"prediction": prediction.tolist()}


@app.get("/hallo")
async def root():
    return {"hello":"world"}
