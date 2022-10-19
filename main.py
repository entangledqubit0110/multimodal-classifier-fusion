import numpy as np
import pandas as pd

from hmcf import HMCF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier


modalityNames = ["ECO", "SOC", "HMC", "NAP"]
modalityCols =  {
                    "ECO": ["PCI", "POV", "DIV", "MIG"],
                    "SOC": ["PHC", "MGA", "SHG", "PEN"],
                    "HMC": ["ILT", "LIT", "SKL"],
                    "NAP": ["FOR", "WAT", "GRZ"]
                }

dataPath = "./fusion_primary_data.csv"

dfTotal = pd.read_csv(dataPath, index_col="Village")

dfList = []
for modality in modalityNames:
    df = dfTotal[modalityCols[modality]]
    dfList.append(df)

sample_size = len(dfTotal)
print(f"Number of samples: {sample_size}")



# create classifiers
dtree = DecisionTreeClassifier(random_state= 193)
svm = SVC(kernel= "linear", random_state= 27)
rfc = RandomForestClassifier(n_estimators=50, random_state= 36)
knn = KNeighborsClassifier(n_neighbors= 3)

classifiers = [dtree, svm, rfc, knn]