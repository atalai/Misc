#   @@@@@@@@@@@@@@@@@@@@@@@@
#   **Code by Aron Talai****
#   @@@@@@@@@@@@@@@@@@@@@@@@

# More details on this great and informative repo
# https://github.com/reachsumit/digital-data-scientist-hiring-test-powered-by-hackerrank

###############################################################################################################################################################################
# dummy inputs
scores = [1,3,5,6,8]
scores = [4,8,7]

lowerLimits = [2,4]
upperLimits = [8,4]


def jobOffers(score_array,lower_limit, upper_limit):

    result = []
    ith_query_params = [[lowerLimits[i], upperLimits[i]] for i in range(0,len(lowerLimits))]

    for i in range(0,len(ith_query_params)):

        #make a new list that falls in the range(lower,upper) for the ith limit combination
        target_scores = [score_array[idx] for idx in range(0,len(score_array)) if ith_query_params[i][0] <= score_array[idx] 
                        and score_array[idx] <= ith_query_params[i][1]]


        result.append(len(target_scores))
    
    return result


#print(jobOffers(scores,lowerLimits,upperLimits))
###############################################################################################################################################################################

#import libraries
#import datetime
from datetime import datetime, timedelta
from dateutil.parser import parse
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
from statsmodels.tsa.arima_model import ARIMA


# dummy inputs
startDate = '2013-01-01'
endDate = '2013-01-02'

knownTimestamps = ['2013-01-01 07:00', '2013-01-01 08:00', '2013-01-01 09:00', '2013-01-01 10:00',
                      '2013-01-01 11:00', '2013-01-01 12:00']
humidity = [10.0, 11.1, 13.2, 14.8, 15.6, 16.7]
timestamps= ['2013-01-01 13:00', '2013-01-01 14:00']


def messy_solution_predictMissingHumidity(startDate, endDate, knownTimestamps, humidity, timestamps):

    '''Convoluted index based solution... Not recommended'''

    # implement simple linear regression model
    # build datetime variables

    # convert date strings to datetime objects
    # startDate = datetime.strptime(startDate, '%Y-%m-%d')
    # endDate = datetime.strptime(endDate, '%Y-%m-%d')
    startDate = parse(startDate)
    endDate = parse(endDate)
    knownTimestamps_humidity = [[parse(str(knownTimestamps[i])),float(humidity[i])] for i in range(0,len(knownTimestamps))] 
    knownTimestamps = [parse(str(knownTimestamps[i])) for i in range(0,len(knownTimestamps))] 


    # # generate list of hourly dates between start time and end time
    list_of_times = []
    idx = 0

    while startDate < endDate:
        
        startDate = startDate + timedelta(hours=1)

        if startDate in knownTimestamps:

            list_of_times.append([idx, startDate, 1])

        if startDate not in knownTimestamps:

            list_of_times.append([idx, startDate, 0])

        idx += 1

    nominal_date_with_humidity = []
    for i in range(0,len(list_of_times)):

      if list_of_times[i][2] == 1:

          for j in range(0,len(knownTimestamps_humidity)):

              if  knownTimestamps_humidity[j][0] == list_of_times[i][1]:
                  nominal_date_with_humidity.append([list_of_times[i][0],knownTimestamps_humidity[j][1]])


    x_axis = [nominal_date_with_humidity[f][0] for f in range(0,len(nominal_date_with_humidity))]
    x_axis = np.array(x_axis).reshape((-1, 1))

    y_axis = [nominal_date_with_humidity[f][1] for f in range(0,len(nominal_date_with_humidity))]
    y_axis = np.array(y_axis)

    # linear regression fit on known data
    model = LinearRegression().fit(x_axis, y_axis)
    # metrics
    #print (model.coef_, model.intercept_, model.score(x_axis, y_axis))


    # find corresponding idx from start and end dates
    timestamps_datetime = [parse(timestamps[i]) for i in range(0,len(timestamps))]
    prediction_timestamps = []
    for i in range(0,len(timestamps_datetime)):

        for j in range(0,len(list_of_times)):

            if timestamps_datetime[i] == list_of_times[j][1]:

                prediction_timestamps.append(list_of_times[j][0])

    y_pred = np.array(prediction_timestamps).reshape((-1, 1))
    y_pred = model.predict(y_pred)

    print (y_pred)


# #predictMissingHumidity(startDate, endDate, knownTimestamps, humidity, timestamps)
# ##############################################################################################################################################################################

# def neat_solution_predictMissingHumidity(startDate, endDate, knownTimestamps, humidity, timestamps):
#     ''' Very nice answer taken from OP, link above'''
#     x = [int(abs((datetime.datetime.utcfromtimestamp(0) - datetime.datetime.strptime(item,"%Y-%m-%d %H:%M")).total_seconds())) for item in knownTimestamps]
#     y = humidity

#     lm = LinearRegression()
#     lm.fit(np.array(x).reshape(-1,1),y)

#     z = [int(abs((datetime.datetime.utcfromtimestamp(0) - datetime.datetime.strptime(item,"%Y-%m-%d %H:%M")).total_seconds())) for item in timestamps]
#     return lm.predict(np.array(z).reshape(-1,1))

# answer = predictMissingHumidity(startDate, endDate, knownTimestamps, humidity, timestamps)
# print(answer)


#def temp_predict(startDate, endDate, temp_array, next_days)


# # dummy inputs (one day)
# data=[10.0,11.1,12.3,13.2,14.8,15.6,16.7,17.5,18.9,19.7,20.7,21.1,22.6,23.5,24.9,25.1,26.3,27.8,28.8,29.6,30.2,31.6,32.1,33.7]
# startDate = '2013-01-01'
# endDate = '2013-01-02'

# #### start
# data = np.array(data)
# startDate = parse(startDate)
# endDate = parse(endDate)

# date_column = []
# while startDate < endDate:
#     startDate = startDate + timedelta(hours=1)
#     date_column.append(startDate)

# date_column = np.array(date_column)

# # convert and merge both arrays into pandas dataframe 
# data_base = np.stack((date_column, data)).T
# data_base = pd.DataFrame(data_base, columns = ['date','temp'])

# #X = data_base.temp
# X = np.array(data_base)
# #ignore seasonal componenet as we are operating in hourly format

# model = ARIMA(X, order=(1,0,1))
# model_fit = model.fit(disp=0)
# # print summary of fit model
# print(model_fit.summary())