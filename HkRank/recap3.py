#   @@@@@@@@@@@@@@@@@@@@@@@@
#   **Code by Aron Talai****
#   @@@@@@@@@@@@@@@@@@@@@@@@

# Hackertank solution to https://www.hackerrank.com/challenges/correlation-and-regression-lines-7/problem

# def libs
import math 
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# sklearn solution 
phys = np.array(phys).reshape(-1,1) # always reshape the independent variable in a sklearn regression
hist = np.array(hist)
model = LinearRegression().fit(phys,hist)

prediction_list = [10]
prediction_set = np.array(prediction_list).reshape(-1,1)
print (model.predict(prediction_set))