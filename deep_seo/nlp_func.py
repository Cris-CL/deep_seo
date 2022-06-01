import pandas as pd
import numpy as np
import os
from joblib import load
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import pad_sequences


class nlp_amazon():
    def __init__(self):
        current_dir =os.path.dirname(__file__)
        model_path = os.path.join(current_dir,'..', 'model')

        self.model = load_model(os.path.join(model_path,'model_nlp.h5'))

        self.tokenizer = load(os.path.join(model_path,'token.joblib'))


    def predict_category(self,data):

        X_pred = np.array(data['title'] + data['description'] + data['feature'])
        X_pred_token = self.tokenizer.texts_to_sequences(X_pred)
        X_pred_pad = pad_sequences(X_pred_token, dtype='float32', padding='post', maxlen=3278)

        prediction = self.model.predict(X_pred_pad)
        result = np.argmax(prediction)

        return result
