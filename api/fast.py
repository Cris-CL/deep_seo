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

@app.get("/predict")
def predict(pickup_datetime,
            pickup_longitude,
            pickup_latitude,
            dropoff_longitude,
            dropoff_latitude,
            passenger_count):


    key = '2013-07-06 17:18:00.000000119'

    # create a datetime object from the user provided datetime
    pickup_datetime = datetime.strptime(pickup_datetime, "%Y-%m-%d %H:%M:%S")

    # localize the user datetime with NYC timezone

    eastern = pytz.timezone("US/Eastern")
    localized_pickup_datetime = eastern.localize(pickup_datetime, is_dst=None)
    utc_pickup_datetime = localized_pickup_datetime.astimezone(pytz.utc)
    formatted_pickup_datetime = utc_pickup_datetime.strftime("%Y-%m-%d %H:%M:%S UTC")

    data = {'key':key,
            'pickup_datetime':formatted_pickup_datetime,
            'pickup_longitude':[pd.to_numeric(pickup_longitude,downcast='float')],
            'pickup_latitude':[pd.to_numeric(pickup_latitude,downcast='float')],
            'dropoff_longitude':[pd.to_numeric(dropoff_longitude,downcast='float')],
            'dropoff_latitude':[pd.to_numeric(dropoff_latitude,downcast='float')],
            'passenger_count':[pd.to_numeric(passenger_count,downcast='integer')]
            }



    X_pred = pd.DataFrame(data)

    # X_pred = X_pred.astype({'pickup_longitude':'float64',
    #                'pickup_latitude':'float64',
    #                'dropoff_longitude':'float64',
    #                'dropoff_latitude':'float64',
    #                'passenger_count':'int64'
    #                })

    print(X_pred)

    model = load('model.joblib')

    prediction = model.predict(X_pred)

    print(prediction)

    return {'fare':prediction[0]}


# {'pickup_datetime':pickup_datetime,
#             'pickup_longitude':pickup_longitude,
#             'pickup_latitude':pickup_latitude,
#             'dropoff_longitude':dropoff_longitude,
#             'dropoff_latitude':dropoff_latitude,
#             'passenger_count':passenger_count}


# predict('2013-07-06 17:18:00','-73.950655','40.783282','-73.984365','40.769802','1')
