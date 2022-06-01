from http.client import LENGTH_REQUIRED
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from joblib import load
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from datetime import datetime
import pytz
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import pad_sequences
from deep_seo.utils import url_predict_rank
import os

app = FastAPI()

# ## DL model initializer
# train_seo = trainer()
# model_ini = train_seo.init_nlp_model()
# model = train_seo.model
# tokenizer_seo = train_seo.tokenizer

## Model Loading at the beginning of the api execution

current_dir =os.path.dirname(__file__)
model_path = os.path.join(current_dir,'..', 'model')

model = load_model(os.path.join(model_path,'model_nlp.h5'))
tokenizer_seo = load(os.path.join(model_path,'token.joblib'))

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

@app.get("/seo_eval")
def predict_seo(title,description,feature,imageone, imagetwo,imagethree):

    key = '3'
    len_title = len(title)
    len_description = len(description)


    data = pd.DataFrame({'title':[title+' '],
                             'description':[description+' '],
                             'feature':[feature+' '],})

    X_pred = np.array(data['title'] + data['description'] + data['feature'])

    X_pred_token = tokenizer_seo.texts_to_sequences(X_pred)
    X_pred_pad = pad_sequences(X_pred_token, dtype='float32', padding='post', maxlen=3278)

    prediction = model.predict(X_pred_pad)

    result = np.argmax(prediction)

    class_names = ['rankeight', 'rankfive', 'rankfour', 'ranknine', 'rankone', 'rankseven', 'ranksix', 'rankten', 'rankthree', 'ranktwo']
    current_dir =os.path.dirname(__file__)
    model_path = os.path.join(current_dir,'..', 'imagemodel')
    model_img = load_model(os.path.join(model_path,'imagemodelfinal.h5'))

    image_arr_one=url_predict_rank(imageone)
    predictions = model_img.predict(image_arr_one)
    predicted_class_one = class_names[np.argmax(predictions[0])]
    print(predicted_class_one)

    image_arr_two=url_predict_rank(imagetwo)
    predictions = model_img.predict(image_arr_two)
    predicted_class_two = class_names[np.argmax(predictions[0])]
    print(predicted_class_two)

    image_arr_three=url_predict_rank(imagethree)
    predictions = model_img.predict(image_arr_three)
    predicted_class_three = class_names[np.argmax(predictions[0])]
    print(predicted_class_three)

    return {'Classification': str(result), 'Rank Image 1': predicted_class_one, 'Rank Image 2': predicted_class_two, 'Rank Image 3': predicted_class_three}


@app.get("/test")
def test():
    return predict('','','','','',0)


# title,description,feature,category,brand,image
# /evaluation?key=3&title=Socks&description=Pink socks stripes&feature=Long pink sock&category=clothing&brand=uniqlo&image=2
# /seo_eval?title=fake title for a fake product &description=this description is awesome&feature=I have no features&imageone=https://cdn-image02.casetify.com/usr/4787/34787/~v3/22690451x2_iphone13_16003249.png.1000x1000-w.m80.jpg&imagetwo=https://m.media-amazon.com/images/I/81MLO3k15iL._AC_SL1500_.jpg&imagethree=https://m.media-amazon.com/images/I/71HiwDwAcoL._AC_SL1500_.jpg
