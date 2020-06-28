#   @@@@@@@@@@@@@@@@@@@@@@@@
#   **Code by Aron Talai****
#   @@@@@@@@@@@@@@@@@@@@@@@@

# Hackertank solution to https://www.hackerrank.com/challenges/correlation-and-regression-lines-6/problem

# def libs
import math 
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

Physics = [15,  12,  8,   8,   7,   7,   7,   6,   5,   3]
History = [10,  25,  17, 11,  13,  17,  20,  13,  9,   15]

def pearson_coefficient(A,B):

	N = len(A)
	sum_of_products = sum([B[i]*A[i] for i in range(0,len(B))]) 
	sum_of_physics = sum(A)
	sum_of_history = sum(B)
	sum_of_sq_physics = sum([A[i]*A[i] for i in range(0,len(A))]) 
	sum_of_sq_history = sum([B[i]*B[i] for i in range(0,len(B))]) 

	numerator = (N*sum_of_products) - (sum_of_physics*sum_of_history)
	denominator_1 = (N*sum_of_sq_physics - (sum_of_physics*sum_of_physics))
	denominator_2 = (N*sum_of_sq_history - (sum_of_sq_history*sum_of_sq_history))
	denominator = denominator_1*denominator_2

	result = numerator/(math.sqrt(abs(denominator))) 
	print (result)


def karl_pearson_coefficient(A,B):

	x_mean = sum(A)/len(A)
	y_mean = sum(B)/len(B)

	numerator_sum = [(A[i] - x_mean)*(B[i] - y_mean) for i in range(0,len(A))]
	numerator_sum = sum(numerator_sum)
	denom_1_sum = [(A[i] - x_mean)*(A[i] - x_mean) for i in range(0,len(A))]
	denom_1_sum = sum(denom_1_sum)
	denom_2_sum = [(B[i] - y_mean)*(B[i] - y_mean) for i in range(0,len(B))]
	denom_2_sum = sum(denom_2_sum)
	result = numerator_sum/((math.sqrt(denom_1_sum))*(math.sqrt(denom_2_sum)))
	print (result)


