#fastapi[standard]
#scikit-learn
#numpy
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi import FastAPI,Form
from fastapi.responses import HTMLResponse
import numpy as n
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # this is already comes in matplotlib library
from sklearn.linear_model import LinearRegression


app = FastAPI()

class User_Input(BaseModel):
    experience : int
    age        : int
    skillscores: int

# X REPRESENTS FEAUTURES
X = n.array( [[1,18,100],
              [2,20,200],
              [3,21,300],
              [4,23,400],
              [5,24,500],
              [6,25,600]])# X REPRESENTS EXPERIENCE
Y = n.array([100,300,400,500,1000,2000]) # Y REPRESENTS SALARY 

model = LinearRegression() # CREATING OBJECT

# TRAIN MODEL
model.fit(X,Y)

@app.get("/",response_class=HTMLResponse)
def home():
    with open("Index.html", "r") as f:
        return f.read()

@app.post("/predict")
def predict(experience: int = Form(...),age : int = Form(...) ,skillscores : int = Form(...)):
    exp_num = int(experience)
    age_num = int(age)
    skill_num = int(skillscores)
    value = model.predict([[   exp_num,age_num,skill_num]])
    return {"Message" : "Prediction Successful" , "Predicted_Salary": round(float(value[0]))}