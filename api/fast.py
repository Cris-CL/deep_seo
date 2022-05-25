from http.client import LENGTH_REQUIRED
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from joblib import load
import pandas as pd
from sklearn.linear_model import LinearRegression
from datetime import datetime
import pytz


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.get("/")
def index():
    return {"greeting": "Hello world"}

@app.get("/evaluation")
def predict(title,description,feature,category,brand,image):

    key = '3'
    len_title = len(title)
    len_description = len(description)
    # len_feature = len(feature)
    # category = ?
    brand_cat = 0 if brand == "" else 1
    # image = ?

    data = {'title':[len_title],
        'description':[len_description],
        'brand':[brand_cat],
        'image':[1]}


    X_pred = pd.DataFrame(data)

    print(X_pred)

    model = load('model.joblib')

    prediction = model.predict(X_pred)
    prediction=int(prediction[0])

    print(prediction)
    return {'ranking':prediction}
    # return prediction[0]

@app.get("/test")
def test():
    return predict('','','','','',0)

print(predict('','','','','',0))

# title,description,feature,category,brand,image
# /evaluation?key=3&title=Socks&description=Pink socks stripes&feature=Long pink sock&category=clothing&brand=uniqlo&image=2
