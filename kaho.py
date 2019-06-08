import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC

raw_iris = datasets.load_iris()

df_X =pd.DataFrame(raw_iris.data)
df_Y =pd.DataFrame(raw_iris.target)

X_train, X_test, Y_train, Y_test = train_test_split(df_X,df_Y, test_size=0.3)
print("The Dataset of Y:",len(df_Y))
print("The Dataset of X:",len(df_X))
print("The Dataset of testing X:",len(X_test))
print("The Dataset of training X:",len(X_train))
print("The Dataset of testing Y:",len(Y_test))
print("The Dataset of training Y:",len(Y_train))
lin_clf = LinearSVC
lin_clf.fit(X_train, Y_train)
LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True,
          intercept_scaling=1, loss="squared_hinge",max_iter=1000,
          multi_class='ovr',penalty='12',random_state=None, tol=0.0001,
          verbose=0)
knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
                     metric_params=None, n_jobs=1, n_neighbors= 5, p=2,
                     weights='uniform')