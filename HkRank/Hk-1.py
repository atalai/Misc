#   @@@@@@@@@@@@@@@@@@@@@@@@
#   **Code by Aron Talai****
#   @@@@@@@@@@@@@@@@@@@@@@@@

# Hackertank solution to https://www.hackerrank.com/challenges/predicting-office-space-price/problem
# Enter your code here. Read input from STDIN. Print output to STDOUT

# def libs
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import statsmodels.api as sm
import statsmodels.formula.api as smf

# init
input_features = []
input_prices = []
predictions = []


header_info = input()
num_of_features = int(header_info.split(' ')[0])
num_of_houses = int(header_info.split(' ')[1])

for i in range(0, num_of_houses):

    current_line = input()
    input_features.append([float(current_line.split(' ')[0]),float(current_line.split(' ')[1])])
    input_prices.append(float(current_line.split(' ')[2]))

number_of_predictions = int(input())

for i in range(0,number_of_predictions):
    current_predictions = input()
    predictions.append([float(current_predictions.split(' ')[0]),float(current_predictions.split(' ')[1])])

input_features = np.asarray(input_features)
input_prices = np.asarray(input_prices)
predictions = np.asarray(predictions)
predictions = sm.add_constant(predictions)

#input_features = pd.DataFrame(input_features)
X_constant = sm.add_constant(input_features)

#print (X_constant)
model = sm.OLS(input_prices, X_constant)
lin_reg = model.fit()

# regr = RandomForestRegressor(max_depth=50, random_state=1993, n_estimators=25)
# model = regr.fit(input_features, input_prices)
# model = LinearRegression().fit(input_features, input_prices)

for i in range(0,number_of_predictions):

    print (lin_reg.predict(predictions)[i])