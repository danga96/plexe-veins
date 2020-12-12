import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.metrics import classification_report
from heatmap import heatmap, corrplot
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

csv_path = "../TrendStepWindow/TrendDB/DB.csv"
attacks = pd.read_csv(csv_path)

print(attacks.info())

#dfObj.drop([dfObj.columns[1] , dfObj.columns[2]] , axis='columns', inplace=True)
cols = [1,2,4,5,12]
print("X",attacks.values[2])
#X = attacks.drop(attacks.columns[[3,4,5,6,-1]], axis=1).values
X = attacks.drop(['Detection'], axis=1).values
Y = attacks["Detection"].values
print("X",X[2])
indices = np.arange(len(Y))
X_train, X_test, Y_train, Y_test, idx1, idx2 = train_test_split(X, Y, indices, test_size = 0.3, random_state = 0, shuffle=True)

lr = LogisticRegression()
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
print(X_train)
print("MEAN: ",scaler.mean_)
print("STD: ",scaler.var_**0.5)
X_test = scaler.transform(X_test)

lr.fit(X_train, Y_train)

Y_pred = lr.predict(X_test)
Y_pred_proba = lr.predict_proba(X_test)


print("ACCURACY: ", str(metrics.accuracy_score(Y_test,Y_pred)))
print("LOG LOSS: ", str(metrics.log_loss(Y_test, Y_pred_proba)))

print(classification_report(Y_test, Y_pred))

from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(Y_test, Y_pred)


print(confusion_matrix)
"""
print("Y_pred: \n",Y_pred)
print("Y_pred_proba: \n",Y_pred_proba)
print("X_test: \n",X_test)
print("idx1: \n",idx1, " len:",len(idx1))
print("idx1: \n",idx2, " len:",len(idx2))
print("TEST: ", X[idx2[0]])
"""

sns.heatmap(attacks.corr())
plt.show()

print(lr.coef_[0])
importance = lr.coef_[0]
clrs = ['blue' if (x < 0) else 'red' for x in importance ]
plt.bar([x for x in range(len(importance))],  importance,  color=clrs)
plt.show()

"""
model = lr
plt.bar( range(len(model.feature_importances_)), model.feature_importances_)
plt.xticks(range(len(model.feature_importances_)), X_train.columns)
plt.show()
"""
