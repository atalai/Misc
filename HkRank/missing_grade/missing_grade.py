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

# def funcs
def dataframe_subsection (dataframe, subject_1, subject_2, subject_3, subject_4, subject_5):
	'''return a sub section of the input dataframe based on the topic restrictions'''
	new_dataframe = dataframe[(dataframe.topic_1 == subject_1) & (dataframe.topic_2 == subject_2)  & (dataframe.topic_3 == subject_3) 
					& (dataframe.topic_4 == subject_4) & (dataframe.topic_5 == subject_5)] 
	return new_dataframe

def evaluate_model(model, X_test, y_test):
	'''calcualtes correct vs wrongly predicted scores following a with a +/- 1 margin'''
	correct_prediction = 0
	wrong_prediction = 0 

	for i in range(0,X_test.shape[0]):
		predicted = model.predict([X_test[i]])

		if abs(int(predicted) - int(y_test[i])) <= 1:
			correct_prediction += 1

		else:
			wrong_prediction += 1

	print ((correct_prediction/int(X_test.shape[0])*100))

# global init 
list_of_files = []

# read training file
selected_json_path = 'C:/Users/aront/Downloads/trainingAndTest/training-and-test/training.json'
with open(selected_json_path, 'r') as f:
    lines = [x.strip() for x in f.readlines()]

# generate a pandas dataframe of records instead of jsons for better data handling
for i in range(0,len(lines)):

	topic_1 = lines[i].split(',')[0].replace('{','').replace('"','').split(':')[0]
	topic_1_score = lines[i].split(',')[0].replace('{','').replace('"','').split(':')[1]

	topic_2 = lines[i].split(',')[1].replace('"','').split(':')[0]
	topic_2_score = lines[i].split(',')[1].replace('"','').split(':')[1]

	topic_3 = lines[i].split(',')[2].replace('"','').split(':')[0]
	topic_3_score = lines[i].split(',')[2].replace('"','').split(':')[1]

	topic_4 = lines[i].split(',')[3].replace('"','').split(':')[0]
	topic_4_score = lines[i].split(',')[3].replace('"','').split(':')[1]

	topic_5 = lines[i].split(',')[4].replace('"','').split(':')[0]
	topic_5_score = lines[i].split(',')[4].replace('"','').split(':')[1]

	list_of_files.append([topic_1, topic_2, topic_3, topic_4, topic_5, topic_1_score, topic_2_score,topic_3_score, topic_4_score, topic_5_score])


# save and read later for faster data loading
list_of_files = pd.DataFrame(list_of_files, columns = ['topic_1','topic_2','topic_3','topic_4','topic_5','topic_1_score','topic_2_score','topic_3_score','topic_4_score','topic_5_score'])
list_of_files = list_of_files.to_csv('C:/Users/aront/Downloads/trainingAndTest/training-and-test/data.csv', index = False)


# read stored master csv file 
data = pd.read_csv('C:/Users/aront/Downloads/trainingAndTest/training-and-test/data.csv')

data_a = dataframe_subsection(data, 'Physics','Chemistry','Computer Science','English','Mathematics')
data_b = dataframe_subsection(data, 'Physics','Chemistry','Physical Education','English','Mathematics')
data_c = dataframe_subsection(data, 'Physics','Chemistry','Economics','English','Mathematics')
data_d = dataframe_subsection(data, 'Physics','Chemistry','Biology','English','Mathematics')
data_e = dataframe_subsection(data, 'Physics','Accountancy','Business Studies','English','Mathematics')


## drop nominal column values and convert the rest into numpy
## split numpy array into math vs all the other topics
data_d = data_d.drop(columns = ['topic_1','topic_2','topic_3','topic_4','topic_5'])
data = data_d.values
features = data[:,0:4]
labels = data[:,-1]

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

evaluate_model(nb_model,X_test,y_test)
evaluate_model(dt_model,X_test,y_test)
evaluate_model(rf_model,X_test,y_test)
evaluate_model(ld_model,X_test,y_test)
evaluate_model(svm_model,X_test,y_test)