import numpy as np
import pandas as pd
from ec import ROC_AUC, Accuracy, Sensitivity, Specificity

from util import categorize, label

from hmcf import HMCF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import LinearRegression


modalityNames = ["ECO", "SOC", "HMC", "NAC"]
modalityCols =  {
                    "ECO": ["PCI", "POV", "DIV", "MIG"],
                    "SOC": ["PHC", "MGA", "SHG", "PEN"],
                    "HMC": ["ILT", "LIT", "SKL"],
                    "NAC": ["FOR", "WAT", "GRZ"]
                }

dataPath = "./fusion_primary_data.csv"

dfTotal = pd.read_csv(dataPath, index_col="Village")
# print(dfTotal.head())
dfProcessed = categorize(dfTotal)
# print(dfProcessed.head())


sample_size = len(dfTotal)
print(f"Number of samples: {sample_size}")

# label 
heuristic_weights = [20, 20, 20, 20, 
                    10, 10, 10, 10,
                    15, 15, 15,
                    5, 5, 5]
heuristic_weights = np.array(heuristic_weights)/np.sum(heuristic_weights)


# wts, dfLabeled = label(dfProcessed, weights="random")
# wts, dfLabeled = label(dfProcessed, weights="uniform")
wts, dfLabeled = label(dfProcessed, weights=heuristic_weights)
# print(wts)
print(dfLabeled)

y = dfLabeled['weighted_sum_label']
X = dfLabeled.drop(['weighted_sum_label'], axis= 1)
dfList = []
for modality in modalityNames:
    df = X[modalityCols[modality]]
    dfList.append(df)
# for df in dfList:
#     print(df.head())
# print(y)


# create classifiers
dtree = DecisionTreeClassifier(random_state= 193)
svm = SVC(kernel= "linear", probability= True, random_state= 27)
rfc = RandomForestClassifier(n_estimators=50, random_state= 36)
knn = KNeighborsClassifier(n_neighbors= 3)

classifiers = [dtree, svm, rfc, knn]
ecs = [Accuracy(), Specificity(), Sensitivity()]
ec_wts = np.ones(len(ecs))/len(ecs)
hmcf = HMCF(classifiers= classifiers, evaluation_criterias= ecs, 
            ec_weights= ec_wts, modality_cross_validation= 2)

yList = [y]*len(dfList)
modalityNames = ["ECO", "SOC", "HMC", "NAC"]
hmcf.fit(XList= dfList, yList=yList, modalityNames= modalityNames)

print(f"weights of modalities: {hmcf.weightPerModality}")

for name in modalityNames:
    print(f"weights in {name}: {hmcf.mcfs[name].weightPerClassfier}")


test = X.iloc[:2, :]
print(test)
testList = []
for modality in modalityNames:
    dfTest = test[modalityCols[modality]]
    testList.append(dfTest)

# for df in testList:
#     print(df.head())
# print(y)

print("----Test----")
print(hmcf.predict_proba(testList, modalityNames))

lc = LinearRegression()
lc.fit(X, y)
print(lc.coef_)
