import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import classification_report

csv_path = "/home/tesi/src/plexe-veins/examples/injectionDetection/analysis/Other/DB_norm.csv"
attacks = pd.read_csv(csv_path)

print(attacks.info())

X = attacks.drop(["Detection"], axis=1).values
Y = attacks["Detection"].values

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state = 0)

lr = LogisticRegression()

lr.fit(X_train, Y_train)

Y_pred = lr.predict(X_test)
Y_pred_proba = lr.predict_proba(X_test)

print("ACCURACY: ", str(metrics.accuracy_score(Y_test,Y_pred)))
print("LOG LOSS: ", str(metrics.log_loss(Y_test, Y_pred_proba)))

print(classification_report(Y_test, Y_pred))

from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(Y_test, Y_pred)
print(confusion_matrix)