# Import Union since our Item object will have tags that can be strings or a list.
from typing import Union 
from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder

# BaseModel from Pydantic is used to define data objects.
from pydantic import BaseModel, Field
from sqlalchemy import alias

# Declare the data object with its components and their type.
class Subject(BaseModel):
    age: int
    workclass: str
    education: str
    marital_status: str = Field(alias='marital-status')
    occupation: str
    relationship: str
    race: str
    sex: str
    hours_per_week: int = Field(alias='hours-per-week')
    native_country: str = Field(alias='native-country')
    

app = FastAPI()

# This allows sending of data (our TaggedItem) via POST to the API.
@app.post("/salary/")
async def predict_salary(data: Subject):
    return jsonable_encoder(data)

@app.get("/")
async def greetings():
    return {"Welcome to the Salary Predictor"}