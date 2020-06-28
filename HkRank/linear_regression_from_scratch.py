#   @@@@@@@@@@@@@@@@@@@@@@@@
#   **Code by Aron Talai****
#   @@@@@@@@@@@@@@@@@@@@@@@@

# Hackertank solution to https://www.hackerrank.com/challenges/correlation-and-regression-lines-7/problem
# good link: https://stats.stackexchange.com/questions/22718/what-is-the-difference-between-linear-regression-on-y-with-x-and-x-with-y
# def libs
import math 
import statistics as st

# def funcs
def pearson_coefficient(A,B):
	''' calculate pearson_coefficient using th eog method '''
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
	return result

def std(number_list):
	'''calculate the standard deviation of a list using the og method'''
	mean_A = sum(number_list)/len(number_list)
	term = (1/len(number_list)) * (sum((number_list[i] - mean_A)**2 for i in range(0,len(number_list))))
	result = math.sqrt(term)
	return result


Physics = [15,  12,  8,   8,   7,   7,   7,   6,   5,   3]
History = [10,  25,  17, 11,  13,  17,  20,  13,  9,   15]

tt = std(History)/std(Physics)

slope = pearson_coefficient(Physics,History) * (std(History)/std(Physics))
print (slope)