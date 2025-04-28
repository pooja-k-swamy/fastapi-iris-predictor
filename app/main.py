import os
import uvicorn
from pydantic import BaseModel
from fastapi import FastAPI
from sklearn import datasets
import pickle
'''
Leverage the FAST API framework to build the API
1. Define model file name 
2. To build a prediction service, we need the trained pickle file to be loaded
3. Build an app using the FAST API module
4. Build a class that inherits from pydantic basemodel(building automatic validation into the API)
5. Define an endpoint
'''
# define model name to be loaded
MODEL_FILENAME = os.path.join("model", "classifier.pkl")

# generate FastAPI app and load model to be used in /predict endpoint

app = FastAPI()
model = pickle.load(open(MODEL_FILENAME, "rb"))
labels = datasets.load_iris().target_names

# define class that represents flower measurements
class IrisSpecies(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# define predict endpoint
@app.post('/predict')
def predict_species(iris: IrisSpecies):
    data_processed = [[iris.sepal_length, iris.sepal_width, iris.petal_length, iris.petal_width]]
    prediction = model.predict(data_processed)[0]
    probability = model.predict_proba(data_processed).max()
    return {
        "top_prediction": labels[prediction],
        "confidence_score": probability
    }
