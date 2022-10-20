import numpy as np
import pandas as pd
from typing import Iterable
from sklearn.model_selection import KFold

from ec import EvaluationCriteria
from classifier_fusion import ClassifierFusion


class HMCF:
    """
    HMCF model implementation
    
    Parameters
    ------------------------------
    classifiers: An iterable of classifiers from scikit-learn library
                constraints are as desrcribed in ClassifierFusion
    evaluation_criterias: An iterable of EvaluationCriteria objects
                        that implements various Evaluation crietrias
    ec_weights: weights of respective evaluation criterias

    """
    def __init__(self, classifiers, evaluation_criterias: Iterable[EvaluationCriteria], ec_weights: Iterable[np.float32],
                modality_cross_validation:int = 10):
        self.classifiers = classifiers
        self.evaluation_criterias = evaluation_criterias
        self.ec_weights = ec_weights
        self.modality_cross_validation = modality_cross_validation  # number of folds for cross validation for a single modality

        # intialize
        self.mcfs = []  # Classifier Fusion objects
        self.modality_ec_matrix = []    # ec_vals for each modality
        self.weightPerModality = None   # weight of each modality

    def fit (self, XList: Iterable[pd.DataFrame], yList: Iterable[pd.Series], modalityNames):
        # sanity check on passed data
        assert len(XList) > 1, "Single modality is not expected"
        n_sample = len(XList[0])
        for X, y in zip(XList, yList):
            assert len(X) == n_sample and len(y) == n_sample, "Mismatching sample size in different modalities"
        

        probaPerModality = []   # n_modality * n_sample * n_class
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
            # fit the mcf
            mcf.fit(X_train, y_train)
            # use val set to get weights
            weights, proba = mcf.getFusionWeights(X_val, y_val)
            kfoldWeights.append(weights)
            kfoldProba.append(proba)
            # get the fused ec_values of all classfiers from the mcf 
            ecVals = mcf.getFusedEvaluationCriteriaValues()
            kfoldECValues.append(ecVals)
        
        kfoldWeights = np.array(kfoldWeights)
        kfoldECValues = np.array(kfoldWeights)

        # get EC values for this modality
        # average over k folds
        avgECVals = np.average(kfoldECValues, axis= 0)
        self.modality_ec_matrix.append(avgECVals)
        # update the modality specific MCF weights from KFolds
        # average over k folds
        avgWeights = np.average(kfoldWeights, axis= 0)
        mcf.setFusionWeights(avgWeights)

        # add to the global list
        self.mcfs.append(mcf)
        
        # get class probabilty values by averaging over k folds
        modalityProba = np.average(kfoldProba, axis= 0)
        return modalityProba


    def predict_proba (self, X_testList: Iterable[pd.DataFrame], modalityNames):
        """Return class probabilies"""
        # sanity check on passed data
        assert len(X_testList) > 1, "Single modality is not expected"
        n_sample = len(X_testList[0])
        for X in X_testList:
            assert len(X) == n_sample , "Mismatching sample size in different modalities"
        

        probaPerModality = [] # n_modality * n_sample * n_class
        for idx, modality, X_test in enumerate(zip(modalityNames, X_testList)):
            print(f"Predicting with modality {modality}")
            modalityProba = self.mcfs[idx].predict_proba(X_test)
            probaPerModality.append(modalityProba)

        
        # weight with modality weights
        if self.weightPerModality is None:
            raise SyntaxError("Unknown weights, please call fit first")
        
        weightedProba = probaPerModality * self.weightPerModality.reshape(-1, 1, 1)
        fusedProba = np.sum(weightedProba, axis= 0)     # n_samples * n_class
        return fusedProba




    ## TODO
    def normalize_and_scale (self, ec_matrix: np.ndarray):
        """normalize and scale by weights"""
        norm_vector = np.linalg.norm(ec_matrix, axis= 0)
        normalized_ec_matrix = ec_matrix/norm_vector
        scaled_ec_matrix = normalized_ec_matrix * self.ec_weights.reshape(-1, 1)
        return scaled_ec_matrix