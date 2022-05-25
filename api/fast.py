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
q
@app.get("/")
def index():
    return {"greeting": "Hello world"}


@app.get("/evaluation")
def predict(title,
            description,
            feature,
            category,
            brand,
            image):

    # key = '3'

    len_title = len(title)
    len_description = len(description)
    len_feature = len(feature)
    category = ?
    brand_cat = 0 if brand == "" else 1
    image = ?


data = {'key':key,
        'title':len_title,
        'description':len_description
        'feature':len_feature,
        'category':?,
        'brand':brand_cat,
        'image':?
        }


#     X_pred = pd.DataFrame(data)

#     # X_pred = X_pred.astype({'pickup_longitude':'float64',
#     #                'pickup_latitude':'float64',
#     #                'dropoff_longitude':'float64',
#     #                'dropoff_latitude':'float64',
#     #                'passenger_count':'int64'
#     #                })

#     print(X_pred)

#     model = load('model.joblib')

#     prediction = model.predict(X_pred)

#     print(prediction)

#     return {'ranking':prediction[0]}


# # {'pickup_datetime':pickup_datetime,
# #             'pickup_longitude':pickup_longitude,
# #             'pickup_latitude':pickup_latitude,
# #             'dropoff_longitude':dropoff_longitude,
# #             'dropoff_latitude':dropoff_latitude,
# #             'passenger_count':passenger_count}


# # predict('2013-07-06 17:18:00','-73.950655','40.783282','-73.984365','40.769802','1')
