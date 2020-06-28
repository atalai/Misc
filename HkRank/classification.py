#   @@@@@@@@@@@@@@@@@@@@@@@@
#   **Code by Aron Talai****
#   @@@@@@@@@@@@@@@@@@@@@@@@

# Calculate missing grade for the Indian university enterance exam
# data can be found here : https://www.hackerrank.com/challenges/predict-missing-grade/problem

# def libs
import os
import json
import pandas as pd
import numpy as np
import warnings
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import BernoulliNB
warnings.filterwarnings('ignore')




# split for training and testing
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.1, shuffle = True)

# train different inherently multi-class classifeirs
nb_clf = BernoulliNB(fit_prior=False)
nb_model = nb_clf.fit(X_train, y_train)

dt_clf = DecisionTreeClassifier(max_depth=7,random_state=1993)
dt_model = dt_clf.fit(X_train, y_train)

rf_clf = RandomForestClassifier(bootstrap=True, n_estimators=100, max_depth=5, random_state=1993)
rf_model = rf_clf.fit(X_train, y_train)  

ld_clf = LinearDiscriminantAnalysis(solver='eigen')
ld_model = ld_clf.fit(X_train, y_train)  

svm_clf = LinearSVC(random_state=1993, tol=1e-4, multi_class  = 'crammer_singer', max_iter=1000)
svm_model = svm_clf.fit(X_train, y_train)

