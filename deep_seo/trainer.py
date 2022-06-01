from sklearn.svm import SVC
from termcolor import colored
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import make_union
from sklearn.compose import make_column_transformer
from sklearn.model_selection import train_test_split
from deep_seo.data import get_computer_data, get_telephone_data
from deep_seo.utils import prepare_nlp
from tensorflow.keras.models import load_model
import os


class deep_seo_trainer(object):

    def __init__(self):

        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.data = get_computer_data()
        self.X = self.data[["brand_cat","title_count","desc_count","img_count"]]
        self.y = self.data["rank_cat"]
        self.model = None
        self.tel_data = get_telephone_data()
        self.X_tel = self.tel_data[['title','clean_description','clean_feature','brand_cat']]
        self.y_tel =self.tel_data['rank_binss']


    def run_model(self):

        X = self.X
        y= self.y


        ## Spliting the set on training and test

        X_train,X_test,y_train,y_test= train_test_split(X,y, test_size=0.3)

        sc = StandardScaler()

        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)


        svc=SVC(kernel="linear",C=10,gamma='auto')

        model_svc =svc.fit(X_train,y_train)
        model_svc.score(X_test,y_test)

        self.model = model_svc

    def init_nlp_model(self):

        ## Load deep learning model
        current_dir =os.path.dirname(__file__)
        model_path = os.path.join(current_dir,'..', 'model')
        self.model = load_model(os.path.join(model_path,'cris_model_nlp.h5'))

        tokenizer, X_2_pad, y_2_cat, brands_and_cat, vocab_size = prepare_nlp(self.tel_data)
        self.tokenizer = tokenizer
        self.brand_cat = brands_and_cat
        self.vocab_s = vocab_size
        self.X_2 = X_2_pad
        self.y_cat = y_2_cat

        return self


    def save_model_locally(self):
        """Save the model into a .joblib format"""
        joblib.dump(self.model, 'model.joblib')
        print(colored("model.joblib saved locally", "green"))



if __name__ == "__main__":
    print('hello deep seo is working!!!!')


    trainer = deep_seo_trainer()

    X_train, X_test, y_train, y_test = train_test_split(trainer.X, trainer.y, test_size=0.3)
    # Train and save model, locally and
    trainer.run_model()
    accuracy = trainer.model.score(X_test, y_test)
    print(f"accuracy: {accuracy}")
    trainer.save_model_locally()
