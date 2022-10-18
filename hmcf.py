import numpy as np
import pandas as pd
from typing import Iterable
from sklearn.model_selection import KFold

from ec import EvaluationCriteria
from classifier_fusion import ClassifierFusion


class HMCF:
    """HMCF model implementation"""
    def __init__(self, classifiers, evaluation_criterias: Iterable[EvaluationCriteria], ec_weights: Iterable[np.float32],
                modality_cross_validation:int = 10):
        self.classifiers = classifiers
        self.evaluation_criterias = evaluation_criterias
        self.ec_weights = ec_weights

        self.mcfs = []  # Classifier Fusion objects
        self.modality_cross_validation = modality_cross_validation  # number of folds for cross validation for a single modality
        self.modality_ec_matrix = []    # ec_vals for each modality
        self.weightPerModality = None

    def fit (self, XList: Iterable[pd.DataFrame], yList: Iterable[pd.Series], modalityNames):
        probaPerModality = []
        for modality, X, y in zip(modalityNames, XList, yList):
            print(f"Fitting on modality {modality}")
            modalityProba = self.fitModality(X, y)
            probaPerModality.append(modalityProba)

        
        assert isinstance(self.modality_ec_matrix, list) and np.shape(self.modality_ec_matrix) == (len(modalityNames), len(self.evaluation_criterias))
        self.modality_ec_matrix = np.array(self.modality_ec_matrix)

        self.modality_ec_matrix = self.normalize_and_scale(self.modality_ec_matrix)

        ec_min = np.min(self.modality_ec_matrix, axis= 0)     # column wise min
        ec_max = np.max(self.modality_ec_matrix, axis= 0)     # column wise max

        weightPerModality = []    # n_modality
        for i in range(len(modalityNames)):
            dis_from_max = np.linalg.norm(self.modality_ec_matrix[i] - ec_max)
            dis_from_min = np.linalg.norm(self.modality_ec_matrix[i] - ec_min)
            wt = dis_from_min/(dis_from_min + dis_from_max)
            weightPerModality.append(wt)
        
        # normalize weights
        weightPerModality = np.array(weightPerModality)
        weightPerModality = weightPerModality/np.sum(weightPerModality)
        self.weightPerModality = weightPerModality

    
    def fitModality (self, X, y):
        """Apply fit on a single modality using classifier fusion"""
        mcf = ClassifierFusion(self.classifiers, self.evaluation_criterias, self.ec_weights)
        kf = KFold(n_splits= self.modality_cross_validation)

        kfoldWeights = []       # k * n_classifiers
        kfoldECValues = []      # k * n_EC 
        kfoldProba = []         # k * n_sample * n_class

        for train_index, val_index in kf.split(X, y):
            X_train, X_val = X[train_index], X[val_index]
            y_train, y_val = y[train_index], y[val_index]
            mcf.fit(X_train, y_train)

            weights, proba = mcf.getFusionWeights(X_val, y_val)
            kfoldWeights.append(weights)
            kfoldProba.append(proba)

            ecVals = mcf.getFusedEvaluationCriteriaValues()
            kfoldECValues.append(ecVals)
        
        kfoldWeights = np.array(kfoldWeights)
        kfoldECValues = np.array(kfoldWeights)

        # get EC values for this modality
        avgECVals = np.sum(kfoldECValues, axis= 0)/self.modality_cross_validation
        self.modality_ec_matrix.append(avgECVals)
        # update the modality specific MCF weights from KFolds
        avgWeights = np.sum(kfoldWeights, axis= 0)/self.modality_cross_validation
        mcf.setFusionWeights(avgWeights)

        # add to the global list
        self.mcfs.append(mcf)

        kfoldProba = np.swapaxes(kfoldProba, 0, 1)  # n_sample * k * n_class
        modalityProba = np.sum(kfoldProba, axis= 1) # sum across k folds
        return modalityProba



    ## TODO
    def normalize_and_scale (self, ec_matrix: np.ndarray):
        """normalize and scale by weights"""
        return ec_matrix