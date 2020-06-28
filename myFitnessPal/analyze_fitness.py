#   @@@@@@@@@@@@@@@@@@@@@@@@
#   **Code by Aron Talai****
#   @@@@@@@@@@@@@@@@@@@@@@@@


# Analyze myFitnessPal Data



# def libs
import os
import numpy as np
import pandas as pd
import time
from datetime import date, timedelta

# def func
def delete_todays_entry(data_frame):
	indexNames = data_frame[data_frame['Date'] == date.today()].index
	data_frame = data_frame.drop(indexNames)
	return data_frame

def convert_column_to_datetime (data_frame):
	data_frame['Date']= pd.to_datetime(data_frame['Date'])
	return data_frame



# global init 
categorical_columns = ['Date', 'Meal','Time']
container = []
# read path to data 
filenames = os.listdir('.')
nutrition = list(filter(lambda x: 'Nutrition' in x, filenames))
excercise = list(filter(lambda x: 'Exercise' in x, filenames))

# read data
nutrition_data = pd.read_csv(os.path.join(os.getcwd(), nutrition[0]))
excercise_data = pd.read_csv(os.path.join(os.getcwd(), excercise[0]))

# convert date column to datetime for better handling
nutrition_data = convert_column_to_datetime(nutrition_data)
excercise_data = convert_column_to_datetime(excercise_data)


# delete todays entry
nutrition_data = delete_todays_entry(nutrition_data)

list_of_categories = list(filter(lambda x: x not in categorical_columns , nutrition_data.columns.values))

for i in range(1,9):

	analysis_date = date.today() - timedelta(days=i)
	filtered_data = nutrition_data[nutrition_data['Date'] == analysis_date]
	filtered_data = filtered_data.sum(axis = 0, skipna = True)
	filtered_data = filtered_data.T
	print (filtered_data)