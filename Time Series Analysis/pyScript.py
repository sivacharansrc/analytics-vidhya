# https://courses.analyticsvidhya.com/courses/take/creating-time-series-forecast-using-python
# Practise Problem: https://datahack.analyticsvidhya.com/contest/practice-problem-time-series-2/

# Time Series Analysis

# A collection of data points collected in time order is called time series. However, not all data collected with respect to time represents a time series.
# Some most common time series data:
#       Demand of Products
#       Website Traffic
#       Number of Calls received
#       Sales Number for next year
#       Stock Market Analysis
#       Census Analysis
#       Budgetary Analysis

### COMPONENTS OF A TIME SERIES ###
# Trend: Trend is a general direction in which something is developing or changing
# Seasonality: Any predictible change or pattern in time series that recurs or repeats over a specific time period

# Main differences between a regression series and a time series: In a regression, the observations are independent whereas in a timeseries, the
# observations are dependent. (i.e. when a Seasonality occurs, the observations are actually dependent from the previous observations or time period)

# Time Series Forecasting is a technique of predicting future values using historical observations

##### PRACTISE PROBLEM - TIME SERIES FORECASTING FOR UNICORN INVESTORS ########
# Problem Statement:
# Unicorn Investors wants to make an investment in a new form of transportation - JetRail. JetRail uses Jet propulsion technology to run rails and move people at a high speed!
# The investment would only make sense, if they can get more than 1 Million monthly users with in next 18 months. In order to help Unicorn Ventures in their decision, you need to
# forecast the traffic on JetRail for the next 7 months. You are provided with traffic data of JetRail since inception in the test file

# HYPOTHESIS GENERATION
# These are the general assumptions which support the goal of the problem statement. In this case, assumptions to be true in order to achieve 1 million visitors in 18 months
# Generally, hypothesis should be generated even before we look at the data so that we avoid any bias. The hypothesis should be generated as soon as we have the problem statement

# Some of the general assumptions we can come up are:
#       - There will be an increase in the traffic as years pass by
#       - Traffic will be high from May through October (general tourism will be at peak during these months)
#       - Traffic on weekdays will be more when compared to that of the weekend (commuting traffic)
#       - Traffic during peak hours will be high

# Let us now analyze the data set and see if we can validate our hypothesis

### Importing necessary libraries ###

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from pandas import Series

### Reading the input and test data ###

# os.chdir("C:\\Users\\ssoma\\OneDrive - Monsanto\\Migrated from My PC\\Documents\\Analytics\\Time Series Analysis")
os.chdir("C:\\Users\\sivac\\Documents\\Analytics\\analytics-vidhya\\Time Series Analysis")
cwd = os.getcwd()

input = pd.read_csv(cwd+"\\input\\train.csv")
test = pd.read_csv(cwd+"\\input\\test.csv")

### Exlporing the dataset ###
input.columns
input.dtypes

input.shape
test.shape

#### FEATURE EXTRACTION #####

# Let us first change the variable format for the datetime object, and extract all possible information from the date and time

input.Datetime.head()

input['Datetime'] = pd.to_datetime(input.Datetime, format='%d-%m-%Y %H:%M')
test['Datetime'] = pd.to_datetime(test.Datetime, format='%d-%m-%Y %H:%M')

# Let us extract all information from the datetime object

for i in (input, test):
    i['Year'] = i.Datetime.dt.year
    i['Month'] = i.Datetime.dt.month
    i['Day'] = i.Datetime.dt.day
    i['Hour'] = i.Datetime.dt.hour
    i['Weekend'] = np.where((i.Datetime.dt.dayofweek == 5) | (i.Datetime.dt.dayofweek == 6), 1, 0)
    i['Day of the week'] = i.Datetime.dt.dayofweek

input.head(10)

# Visualizing the time SERIES

input.index = input['Datetime']
ts = input['Count']

plt.figure(figsize=(16, 8))
plt.plot(ts, label="Count of Passengers")
plt.title("Passenger Count Overtime")
plt.xlabel("Time(Year - Month)")
plt.ylabel("Passenger Count")
plt.legend(loc="best")
plt.show()

### EXPLORATORY ANALYSIS ####
# Let us try to validate all our hypothesis
# Hypothesis 1: Passenger Count increases over years

input.groupby(input.Year).Count.sum().plot.bar()

# From the picture above, the hypothesis is true (i.e. passenter count grows over period)

# Hypothesis 2: Traffic will be high from May through October (general tourism will be at peak during these months)

input.groupby(input.Month).Count.sum().plot.bar()

plt.figure(figsize=(16, 8))
input.groupby(['Year', 'Month']).Count.sum().plot.bar()

# It is important to note that not much can be inferred from the months because, in the year 2012 only 5 month data is available,
# where as for the year 2014 only 9 month data is available. However, there is a very steady growing trend in the passenger Count

# Hypothesis 3: Traffic on weekdays will be more when compared to that of the weekend (commuting traffic)

input.groupby('Day').Count.sum().plot.bar()

# Not much can be inferred from the above plot, however let us see the mean passenger count on weekdays vs weekends

input.groupby('Weekend').Count.mean().plot.bar()

input.groupby('Day of the week').Count.mean().plot.bar()

# The above two plot shows that indeed, more passengers travel on weekdays when compared with weekends

# Hypothesis 4: Traffic during peak hours will be high

input.groupby('Hour').Count.mean().plot.bar()

# From the above plot, the peak traffic is at 7 PM, and then the passenger count starts to decline until 5 AM after which
# there is again a rise in the passenger count until 12 Noon

temp = input.drop('ID',1)
temp['Timestamp'] = pd.to_datetime(temp.Datetime, format='%d-%m-%Y %H:%M')
temp.index = temp.Timestamp



# Let us create different set of timeseries:
# Hourly time series
hourly = temp.resample('H').mean()
hourly.head()

# Daily time series
daily = temp.resample('D').mean()
daily.head()

# Weekly time series
weekly = temp.resample('W').mean()
weekly.head()

# Montly time series
monthly = temp.resample('M').mean()
monthly.head()


plt.figure(figsize=(16, 8))
plt.subplot(4, 1, 1)
hourly.Count.plot(title='Hourly')
plt.subplot(4, 1, 2)
daily.Count.plot(title='Daily')
plt.subplot(4, 1, 3)
weekly.Count.plot(title='Weekly')
plt.subplot(4, 1, 4)
monthly.Count.plot(title='Monthly')

# As we aggregate more and more data, the series becomes more and more stable. For our analysis, we will use the daily timeseries

test['Timestamp'] = pd.to_datetime(test['Datetime'], format='%d-%m-%Y %H:%M')
test.head()
test.index = test.Timestamp
test.head()
test_daily = test.resample('D').mean()

# test.drop('ID', 1).reset_index().head()
test_daily.head()

# Let us use the daily data for our forecasting purpose

# Let us split the input dataset in to input and validation

train = daily.loc['2012-08-25':'2014-06-24']
validation = daily.loc['2014-06-25':'2014-09-25']

train.head()

train.Count.plot(figsize=(16, 8), title="Daily Ridership", label="input")
validation.Count.plot(figsize=(16, 8), title="Daily Ridership", label='validation')
plt.xlabel("Datetime")
plt.ylabel("Count of Passengers")
plt.legend(loc="best")
plt.show()

## Let us discuss various approaches for forecasting time series

# 1. MOVING AVERAGE:
#       Here the predictions are made by taking the average of the last few points. Let us calculate the moving avearge by taking the
#   last 60 observations (basically 2 months of data)

validation_copy = validation.copy()
validation_copy['moving_average_60'] = train.Count.rolling(60).mean().iloc[-1] # Calculating the rolling average using last 60 observations

plt.figure(figsize=(16,8))
plt.plot(train.Count, label = "train")
plt.plot(validation.Count, label = "validation")
plt.plot(validation_copy.moving_average_60, label="moving average 60")
plt.legend(loc="best")
plt.show()

# Calculating Error Metrics

from sklearn.metrics import mean_squared_error
rms = np.sqrt(mean_squared_error(validation.Count, validation_copy.moving_average_60))
print(rms)

# Let us try to take average of different set of observations and try to observe the Error
validation_copy['moving_average_50'] = train.Count.rolling(50).mean().iloc[-1]
validation_copy['moving_average_40'] = train.Count.rolling(40).mean().iloc[-1]
validation_copy['moving_average_30'] = train.Count.rolling(30).mean().iloc[-1]
validation_copy['moving_average_20'] = train.Count.rolling(20).mean().iloc[-1]

rms = np.sqrt(mean_squared_error(validation.Count, validation_copy.moving_average_50))
print(rms)
rms = np.sqrt(mean_squared_error(validation.Count, validation_copy.moving_average_40))
print(rms)
rms = np.sqrt(mean_squared_error(validation.Count, validation_copy.moving_average_30))
print(rms)
rms = np.sqrt(mean_squared_error(validation.Count, validation_copy.moving_average_20))
print(rms)

# There seems to be an increase in error as the number of observation increases after 30 data points

# 2. SIMPLE EXPONENTIAL SMOOTHING
#       In this technique, we assign larger weights to more recent observationn that to observations from the distant past.
#   The weights decrease exponentially as observations come from further in the past, the smallest weights are associated with the oldest
#   observations. Note: If we give entire weight to the last observation, it becomes a naive forecast


from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt

fit2 = SimpleExpSmoothing(np.asarray(train.Count)).fit(smoothing_level=0.6, optimized=False)
validation_copy['SES'] = fit2.forecast(validation.shape[0])

# Plotting Chart to visualize the forecasts
plt.figure(figsize=(16,8))
plt.plot(train.Count, label = "train")
plt.plot(validation.Count, label = "validation")
plt.plot(validation_copy.moving_average_30, label="moving average 30")
plt.plot(validation_copy.SES, label="simple exponential smoothing")
plt.legend(loc="best")
plt.show()

# Computing the error using RMSE
rms = np.sqrt(mean_squared_error(validation.Count, validation_copy.SES))
print(rms)

# All of the previous models forecasted adding some components to average, and hence the forecasts were more or less a straight line
# We can see other class of models that can forecast the trend, and also capture seasonality etc.

# HOLT's LINEAR TREND MODEL
#       This is an extension of SES to allow forecasting of data with a trend. This method takes in to account the trend of the dataset.
#   The forecast function in this method is a function of level and trend
#   First, let us decompose the series in to four parts:
#       Observed - The original time series
#       Trend - Increasing or the decreasing behavior in the time series
#       Seasonality - The cyclic recurrence of patterns in the time series
#       Residual - Obtained by removing trend, and seasonality from the time series. These are basically the irregular patterns that
#                   do not follow a specific pattern

import statsmodels.api as sm
sm.tsa.seasonal_decompose(train.Count).plot()
result = sm.tsa.stattools.adfuller(train.Count) # Not sure what this is doing
plt.show()

# From the above image, an increasing trend can be observed. We can not configure the model accordingly:
fit1 = Holt(np.asarray(train.Count)).fit(smoothing_level=0.3, smoothing_slope=0.1)
validation_copy['holt_linear'] = fit1.forecast(validation.shape[0])

# Lets visualize the forecast:
plt.figure(figsize=(16,8))
plt.plot(train.Count, label = "train")
plt.plot(validation.Count, label = "validation")
plt.plot(validation_copy.moving_average_30, label="moving average 30")
plt.plot(validation_copy.SES, label="simple exponential smoothing")
plt.plot(validation_copy.holt_linear, label="holt linear model")
plt.legend(loc="best")
plt.show()

# We can see the forecasts are moving up as the model is considering the trend of the dataset
# Let us compute the error:

rms = np.sqrt(mean_squared_error(validation_copy.Count, validation_copy.holt_linear))
rms

# We can see that the error rate has further decreased than the exponential smoothing method
# Let us now use this model to predict values for the test data set

predictions = fit1.forecast(test.shape[0])
test_daily['predictions'] = predictions

# Converting daily data to hourly data

input['ratio'] = input.Count / input.Count.sum()
grouped_ratio = input.groupby(['Hour']).ratio.sum()

grouped_ratio = pd.DataFrame(grouped_ratio, columns=['ratio'])
grouped_ratio.to_csv('grouped_ratio.csv')

test.columns
test.shape


test_daily.columns
test_daily.shape

test_final = pd.merge(test, test_daily, on=('Day', 'Month', 'Year'), how='left')
test_final.head()

test_final = test_final[['ID_x', 'Datetime', 'Hour_x', 'predictions']]
test_final.columns
test_final = pd.merge(test_final, grouped_ratio, how='left', left_on='Hour_x', right_on = 'Hour')
test_final.columns

test_final['Count'] = test_final['predictions'] * test_final['ratio']*24 #Multiplying by 24 since we found the mean for daily, and then converting back to count
test_final['ID'] = test_final['ID_x']
submission_file = test_final[['ID', 'Count']].copy()
submission_file.to_csv('first_draft_submission.csv')

# The submitted model gives an rmse of 290, and the first rank is 132








# Pushing the updates to git
! git add .
! git commit -am "uploaded the first draft submission"
! git push