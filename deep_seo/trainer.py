from sklearn.svm import SVC
from termcolor import colored
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import make_union
from sklearn.compose import make_column_transformer
from sklearn.model_selection import train_test_split
from deep_seo.data import get_computer_data


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

    def save_model_locally(self):
        """Save the model into a .joblib format"""
        joblib.dump(self.model, 'model.joblib')
        print(colored("model.joblib saved locally", "green"))



if __name__ == "__main__":
    print('hello deep seo is working!!!!')
    # Get and clean data
    # df = get_data_from_gcp(nrows=N)
    # df = clean_data(df)
    # y = df["fare_amount"]
    # X = df.drop("fare_amount", axis=1)

    trainer = deep_seo_trainer()

    X_train, X_test, y_train, y_test = train_test_split(trainer.X, trainer.y, test_size=0.3)
    # Train and save model, locally and
    trainer.run_model()
    accuracy = trainer.model.score(X_test, y_test)
    print(f"accuracy: {accuracy}")
    trainer.save_model_locally()
