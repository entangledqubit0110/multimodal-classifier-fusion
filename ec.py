from abc import ABC, abstractmethod
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix

class EvaluationCriteria (ABC):
    """An abstract class under which all evaluation cirterias are defined"""

    @abstractmethod
    def getValue (self, y_true, y_pred, proba):
        """
        Return the value of the evaluation criteria
        y_true : true labels for samples
        y_pred : predicted labels for samples
        proba : class probabilities for every sample
        """

class Accuracy (EvaluationCriteria):

    def getValue(self, y_true, y_pred, proba):
        return accuracy_score(y_true= y_true, y_pred= y_pred)

class ROC_AUC (EvaluationCriteria):

    def getValue(self, y_true, y_pred, proba):
        return roc_auc_score(y_true= y_true, y_score= proba)
        

class Specificity (EvaluationCriteria):

    def getValue(self, y_true, y_pred, proba):
        tn, fp, fn, tp = confusion_matrix(y_true= y_true, y_pred= y_pred).ravel()
        return tn/(tn + fp)

class Sensitivity (EvaluationCriteria):

    def getValue(self, y_true, y_pred, proba):
        tn, fp, fn, tp = confusion_matrix(y_true= y_true, y_pred= y_pred).ravel()
        return tp/(tp + fn)
