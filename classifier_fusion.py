from ec import EvaluationCriteria
from typing import Iterable
import numpy as np

class ClassifierFusion:
    """
    Classifier Fusion
    """
    def __init__(self, classifiers, evaluation_criterias: Iterable[EvaluationCriteria], ec_weights: np.ndarray):
        # sanity checking for passed params
        for c in classifiers:
            if not (hasattr(c, 'fit')) and callable(getattr(c, 'fit')):
                raise TypeError(f"Classifier {c} expected to have fit method")
            if not (hasattr(c, 'predict') and callable(getattr(c, 'predict'))):
                raise TypeError(f"Classifier {c} expected to have predict method")
            if not (hasattr(c, 'predict_proba') and callable(getattr(c, 'predict_proba'))):
                raise TypeError(f"Classifier {c} expected to have predict_proba method")


        self.classifiers = classifiers
        self.evaluation_criterias = evaluation_criterias
        self.ec_weights = ec_weights

        self.ec_matrix = None
        self.weightPerClassfier = None

    def fit (self, X, y):
        """Train classifiers on the data"""
        for classifier in self.classifiers:
            classifier.fit(X, y)

    def getFusionWeights (self, X_val, y_val):
        """
        Using the validation X and y values calculate the weights of classifier and
        fusion probabilities for each class for each sample in X_val
        """
        ec_matrix = []          # n_classifier * n_ec
        probaPerClassifier = [] # n_classifier * n_sample * n_class

        for classifier in self.classifiers:
            y_pred = classifier.predict(X_val)         # n_sample
            proba = classifier.predict_proba(X_val)    # n_sample * n_class
            probaPerClassifier.append(np.array(proba))
            
            # calculate ec values
            ec_vector = []          # n_ec
            for ec in self.evaluation_criterias:
                ec_value = ec.getValue(y_true= y_val, y_pred= y_pred, proba= proba)
                ec_vector.append(ec_value)
            # append a row
            ec_matrix.append(ec_vector)
        
        assert np.shape(ec_matrix) == (len(self.classifiers), len(self.evaluation_criterias))

        ec_matrix = np.array(ec_matrix)
        ec_matrix = self.normalize_and_scale(ec_matrix)
        self.ec_matrix = ec_matrix

        ec_min = np.min(ec_matrix, axis= 0)     # column wise min
        ec_max = np.max(ec_matrix, axis= 0)     # column wise max


        weightPerClassifier = []    # n_classifier
        for i in range(len(self.classifiers)):
            dis_from_max = np.linalg.norm(ec_matrix[i] - ec_max)
            dis_from_min = np.linalg.norm(ec_matrix[i] - ec_min)
            # avoid division by zero
            if dis_from_min + dis_from_max != 0:
                wt = dis_from_min/(dis_from_min + dis_from_max)
            else:
                wt = 1/len(self.classifiers)
            weightPerClassifier.append(wt)

        # normalize weights
        weightPerClassifier = np.array(weightPerClassifier)
        weightPerClassifier = weightPerClassifier/np.sum(weightPerClassifier)
        self.weightPerClassfier = weightPerClassifier
        
        # weight the class probabilities for each classifier
        for i in range(len(self.classifiers)):
            probaPerClassifier[i] = probaPerClassifier[i] * weightPerClassifier[i]
        
        # sum of all weighted proba
        fusionProba = np.zeros(np.shape(probaPerClassifier[0])) # n_sample * n_class
        for proba in probaPerClassifier:
            fusionProba = fusionProba + proba
        
        assert len(fusionProba) == len(X_val)
        # TODO
        # for row in fusionProba:
        #     assert np.sum(row) <= 1

        
        return (weightPerClassifier, fusionProba)
    

    def predict_proba (self, X_test):
        """Return class probabilities"""
        probaPerClassifier = [] # n_classifier * n_sample * n_class
        for classifier in self.classifiers:
            proba = classifier.predict_proba(X_test)
            probaPerClassifier.append(proba)
        
        # weight the proba with classifier weights
        if self.weightPerClassfier is None:
            raise SyntaxError("Unknown weights, please call fit first")
        
        weightedProba = probaPerClassifier * self.weightPerClassfier.reshape(-1, 1, 1)
        fusedProba = np.sum(weightedProba, axis= 0)

        return fusedProba   # n_sample * n_class


    def getFusedEvaluationCriteriaValues (self):
        """Return the weighted value of evalutation criteria values according to model weights"""
        if self.ec_matrix is None:
            return None
        
        fusionECVals = np.sum(          # columnwise sum
                                self.ec_matrix * self.weightPerClassfier.reshape(-1, 1),     # multiply each row by classifier wt
                                axis= 0
                             )
        return fusionECVals


    def normalize_and_scale (self, ec_matrix: np.ndarray):
        """normalize and scale by weights"""
        norm_vector = np.linalg.norm(ec_matrix, axis= 0)
        # avoid divison by zero
        for idx in range(len(norm_vector)):
            if norm_vector[idx] == 0:
                norm_vector[idx] += 0.000001
        
        normalized_ec_matrix = ec_matrix/norm_vector
        scaled_ec_matrix = normalized_ec_matrix * self.ec_weights   # multiply each column by ec wt
        return scaled_ec_matrix


    def setFusionWeights (self, weightsPerClassifier: np.ndarray):
        self.weightPerClassfier = weightsPerClassifier
            

        