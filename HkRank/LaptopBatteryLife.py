#   @@@@@@@@@@@@@@@@@@@@@@@@
#   **Code by Aron Talai****
#   @@@@@@@@@@@@@@@@@@@@@@@@

# solution to https://www.hackerrank.com/challenges/battery/problem
# def libs
import pandas as pd
import numpy as np
import urllib.request
import requests
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

#read and save data from host
response = requests.get('https://s3.amazonaws.com/hr-testcases/399/assets/trainingdata.txt')

with open('C:/Users/aront/Desktop/HkRank/data/data.txt', 'wb') as f:  
    f.write(response.content)

# open and feed as lines to data frame
file = open('C:/Users/aront/Desktop/HkRank/data/data.txt','r') 
contents = file.readlines()

# convert text files ionto pandas dataframe
data_base = [[float(line.split(',')[0]), float(line.split(',')[1].split('\n')[0])] for line in contents]
data_base = pd.DataFrame(data_base, columns = ['charged','lasted'])

# based on what was seen in the plot
altered_data_base = data_base[data_base['charged'] < 4]
data_base = altered_data_base

X = data_base.charged.tolist()
X = np.array(X).reshape(-1,1)
y = data_base.lasted.tolist()

model = LinearRegression().fit(X, y)

y_pred = [0.09]
y_pred = np.array(y_pred).reshape(-1,1)
prediction = model.predict(y_pred)

#print (data_base.describe())
print (prediction)

# plot to see relationships
plt.plot(X,y,'o')
plt.show()