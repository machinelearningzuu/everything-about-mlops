import joblib, os
import numpy as np
from sklearn.svm import SVC
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

class CommonModel(object):
    def __init__(self, model_name, dataset_name, params):
        self.params = params
        self.model_name = model_name
        self.dataset_name = dataset_name

        self.data = None
        self.model = None
        self.weight_path = None

        # Create model
        if self.model_name == 'SVM':
            C = self.params["C"]
            self.model = SVC(C=C)
            self.weight_path = 'weights/svm.pkl'

        elif self.model_name == 'KNN':
            K = self.params["K"]
            self.model = KNeighborsClassifier(n_neighbors=K)
            self.weight_path = 'weights/knn.pkl'

        elif self.model_name == 'RF':
            n_estimators = self.params["n_estimators"]
            max_depth = self.params["max_depth"]
            self.model = RandomForestClassifier(
                                                n_estimators=n_estimators, 
                                                max_depth=max_depth
                                                )
            self.weight_path = 'weights/rf.pkl'

        else:
            raise Exception('Invalid model name : {}'.format(self.model_name))

        # Load dataset
        if self.dataset_name == "Iris":
            self.data = datasets.load_iris()

        elif self.dataset_name == "BreastCancer":
            self.data = datasets.load_breast_cancer()
            
        elif self.dataset_name == "WineQuality":
            self.data = datasets.load_wine()

        else:
            ValueError("Invalid dataset name : {}".format(self.dataset_name))

    def get_dataset(self):
        X = self.data.data
        y = self.data.target
        
        self.X = X
        self.y = y

    def train(self):
        self.model.fit(self.X, self.y)
        joblib.dump(self.model, self.weight_path)

    def predict(self):
        return self.model.predict(self.X)

    def score(self):
        return self.model.score(self.X, self.y)

    def process(self):
        self.get_dataset()
        self.train() 

        # if not os.path.exists(self.weight_path):
        #     self.train() 
        # else:
        #     self.model = joblib.load(self.weight_path)