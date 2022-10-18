import string
from tokenize import String
from ec import EvaluationCriteria
from typing import Iterable
import numpy as np

class ClassifierFusion:
    """
    Classifier Fusion
    """
    def __init__(self, classifiers, evaluation_criterias: Iterable[EvaluationCriteria], ec_weights: Iterable[np.float32]):
        # sanity checking for passed params
        for c in classifiers:
            if not (hasattr(c, 'fit')) and callable(getattr(c, 'fit')):
                raise TypeError("Classifier expected to have fit method")
            if not (hasattr(c, 'predict') and callable(getattr(c, 'predict'))):
                raise TypeError("Classifier expected to have predict method")
            if not (hasattr(c, 'predict_proba') and callable(getattr(c, 'predict_proba'))):
                raise TypeError("Classifier expected to have predict_proba method")


        self.classifiers = classifiers
        self.evaluation_criterias = evaluation_criterias
        self.ec_weights = ec_weights

    def fit (self, X, y):
        """Train classifiers on the data"""
        for classifier in self.classifiers:
            classifier.fit(X, y)

    def predict_proba (self, X_test, y_test):
        """Return the prediction for"""
        ec_matrix = []          # n_classifier * n_ec
        probaPerClassifier = [] # n_sample * n_class * n_classifier

        for classifier in self.classifiers:
            y_pred = classifier.predict(X_test)         # n_sample
            proba = classifier.predict_proba(X_test)    # n_sample * n_class
            probaPerClassifier.append(np.array(proba))
            
            # calculate ec values
            ec_vector = []          # n_ec
            for ec in self.evaluation_criterias:
                ec_value = ec.getValue(y_true= y_test, y_pred= y_pred, proba= proba)
                ec_vector.append(ec_value)
            # append a row
            ec_matrix.append(ec_vector)
        
        assert np.shape(ec_matrix) == (len(self.classifiers), len(self.evaluation_criterias))

        ec_matrix = np.array(ec_matrix)
        ec_matrix = self.normalize_and_scale(ec_matrix)
        ec_min = np.min(ec_matrix, axis= 0)     # column wise min
        ec_max = np.max(ec_matrix, axis= 0)     # column wise max

        weightPerClassifier = []    # n_classifier
        for i in range(len(self.classifiers)):
            dis_from_max = np.linalg.norm(ec_matrix[i] - ec_max)
            dis_from_min = np.linalg.norm(ec_matrix[i] - ec_min)
            wt = dis_from_min/(dis_from_min + dis_from_max)
            weightPerClassifier.append(wt)

        # normalize weights
        weightPerClassifier = np.array(weightPerClassifier)
        weightPerClassifier = weightPerClassifier/np.sum(weightPerClassifier)
        
        # weight the class probabilities for each classifier
        for i in range(len(self.classifiers)):
            probaPerClassifier[i] = probaPerClassifier[i] * weightPerClassifier[i]
        
        # sum of all weighted proba
        fusionProba = np.zeros(np.shape(probaPerClassifier[0])) # n_sample * n_class
        for proba in probaPerClassifier:
            fusionProba = fusionProba + proba
        
        assert len(fusionProba) == len(X_test)
        for row in fusionProba:
            assert np.sum(row) <= 1

        
        self.weightPerClassifier = weightPerClassifier
        
        return fusionProba


    def normalize_and_scale (self, ec_matrix: np.ndarray):
        """normalize and scale by weights"""
        
            

        