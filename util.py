import numpy as np
import pandas as pd

POOR = 0
FAIR = 1
GOOD = 2

class Categorizer:
    """Make a numerical continuous feature into 3 categories"""
    def __init__(self, d1, d2, dir):
        assert d1 < d2
        self.d1 = d1
        self.d2 = d2
        self.dir= dir

    def categoryDown (self, val):
        if val < self.d1:
            return GOOD
        elif val >= self.d1 and val < self.d2:
            return FAIR
        else:
            return POOR

    def categoryUp (self, val):
        if val < self.d1:
            return POOR
        elif val >= self.d1 and val < self.d2:
            return FAIR
        else:
            return GOOD
    
    def category (self, val):
        if self.dir == 'UP':
            return self.categoryUp(val)
        elif self.dir == 'DOWN':
            return self.categoryDown(val)


data_categorizers = {}
features = {"PCI": [1350, 5000, "UP"], 
            "POV": [20, 30, "DOWN"], 
            "DIV": [20, 70, "UP"], 
            "MIG": [10, 30, "UP"], 
            "PHC": [26, 60, "UP"],
            "MGA": [20, 50, "UP"],
            "SHG": [15, 40, "UP"], 
            "PEN": [21, 50, "UP"], 
            "ILT": [11, 30, "DOWN"], 
            "LIT": [21, 80, "UP"], 
            "SKL": [11, 20, "UP"], 
            "FOR": [11, 41, "UP"], 
            "WAT": [20, 41, "UP"], 
            "GRZ": [20, 41, "UP"]}

for f in features.keys():
    data_categorizers[f] = Categorizer(d1= features[f][0], d2= features[f][1], dir= features[f][2])
    

def categorize (df: pd.DataFrame):
    """Make the features categorical from numerical"""
    for f in features.keys():
        df[f] = df[f].map(data_categorizers[f].category)
    return df

    

def label (df: pd.DataFrame, weights):
    """Make labels following the weights"""
    if isinstance(weights, str) and weights == "uniform":
        weights = np.ones(len(features))/len(features)
    elif isinstance(weights, str) and weights == "random":
        weights = np.random.rand(len(features))
        weights = weights/np.sum(weights)

    df['weighted_sum_label'] = df.dot(weights)
    df["weighted_sum_label"] = df["weighted_sum_label"].map(lambda x : int(x < 1))

    return weights, df
    