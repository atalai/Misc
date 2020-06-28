#   @@@@@@@@@@@@@@@@@@@@@@@@
#   **Code by Aron Talai****
#   @@@@@@@@@@@@@@@@@@@@@@@@

# Hackertank solution to https://www.hackerrank.com/challenges/correlation-and-regression-lines-7/problem

# def libs
import math 
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# OG solution
phys = [15,12,8,8,7,7,7,6,5,3]
hist = [10,25,17,11,13,17,20,13,9,15]
a = sum(phys)/len(phys)
b = sum(hist)/len(hist)
l1 = [i-a for i in phys]
l2 = [i-b for i in hist]
l3 = [i*j for i,j in zip(l1,l2)]
l4 = [i*i for i in l1]
print("%0.3f"%(sum(l3)/sum(l4)))

# sklearn solution 
phys = np.array(phys).reshape(-1,1) # always reshape the independent variable in a sklearn regression
hist = np.array(hist)
model = LinearRegression().fit(phys,hist)
print (model.coef_, model.intercept_)

