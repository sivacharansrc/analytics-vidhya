# https://courses.analyticsvidhya.com/courses/take/creating-time-series-forecast-using-python
# Practise Problem: https://datahack.analyticsvidhya.com/contest/practice-problem-time-series-2/

### Time Series Analysis

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
## Trend: Trend is a general direction in which something is developing or changing
## Seasonality: Any predictible change or pattern in time series that recurs or repeats over a specific time period

# Main differences between a regression series and a time series: In a regression, the observations are independent whereas in a timeseries, the
# observations are dependent. (i.e. when a Seasonality occurs, the observations are actually dependent from the previous observations or time period)

# Time Series Forecasting is a technique of predicting future values using historical observations

##### PRACTISE PROBLEM - TIME SERIES FORECASTING FOR UNICORN INVESTORS ########
## Problem Statement:
# Unicorn Investors wants to make an investment in a new form of transportation - JetRail. JetRail uses Jet propulsion technology to run rails and move people at a high speed!
# The investment would only make sense, if they can get more than 1 Million monthly users with in next 18 months. In order to help Unicorn Ventures in their decision, you need to
# forecast the traffic on JetRail for the next 7 months. You are provided with traffic data of JetRail since inception in the test file

## HYPOTHESIS GENERATION
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

### Reading the train and test data ###
os.chdir("C:\\Users\\ssoma\\OneDrive - Monsanto\\Migrated from My PC\\Documents\\Analytics\\Time Series Analysis")
cwd = os.getcwd()

train = pd.read_csv(cwd+"\\input\\train.csv")
test = pd.read_csv(cwd+"\\input\\test.csv")

### Exlporing the dataset ###
train.columns
train.dtypes

train.shape
test.shape

#### FEATURE EXTRACTION #####

# Let us first change the variable format for the datetime object, and extract all possible information from the date and time

train.Datetime.head()

train['Datetime'] = pd.to_datetime(train.Datetime, format = '%d-%m-%Y %H:%M')
test['Datetime'] = pd.to_datetime(test.Datetime, format = '%d-%m-%Y %H:%M')

# Let us extract all information from the datetime object

for i in (train, test):
    i['Year'] = i.Datetime.dt.year
    i['Month'] = i.Datetime.dt.month
    i['Day'] = i.Datetime.dt.day
    i['Hour'] = i.Datetime.dt.hour
    i['Weekend'] = np.where((i.Datetime.dt.dayofweek == 5) | (i.Datetime.dt.dayofweek == 6),1,0)
    i['Day of the week'] = i.Datetime.dt.dayofweek

train.head(100)

### Visualizing the time SERIES

train.index = train['Datetime']
ts = train['Count']
plt.figure(figsize=(16,8))
plt.plot(ts, label="Count of Passengers")
plt.title("Passenger Count Overtime")
plt.xlabel("Time(Year - Month)")
plt.ylabel("Passenger Count")
plt.legend(loc="best")

### EXPLORATORY ANALYSIS ####
# Let us try to validate all our hypothesis
# Hypothesis 1: Passenger Count increases over years

train.groupby(train.Year).Count.sum().plot.bar()

# From the picture above, the hypothesis is true (i.e. passenter count grows over period)

# Hypothesis 2: Traffic will be high from May through October (general tourism will be at peak during these months)

train.groupby(train.Month).Count.sum().plot.bar()

plt.figure(figsize = (16,8))
train.groupby(['Year', 'Month']).Count.sum().plot.bar()

# It is important to note that not much can be inferred from the months because, in the year 2012 only 5 month data is available,
# where as for the year 2014 only 9 month data is available. However, there is a very steady growing trend in the passenger Count

# Hypothesis 3: Traffic on weekdays will be more when compared to that of the weekend (commuting traffic)

train.groupby('Day').Count.sum().plot.bar()

# Not much can be inferred from the above plot, however let us see the mean passenger count on weekdays vs weekends

train.groupby('Weekend').Count.mean().plot.bar()

train.groupby('Day of the week').Count.mean().plot.bar()

# The above two plot shows that indeed, more passengers travel on weekdays when compared with weekends

# Hypothesis 4: Traffic during peak hours will be high

train.groupby('Hour').Count.mean().plot.bar()

# From the above plot, the peak traffic is at 7 PM, and then the passenger count starts to decline until 5 AM after which
# there is again a rise in the passenger count until 12 Noon

temp = train.iloc[:,1:3]

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


plt.figure(figsize=(16,8))
plt.subplot(4,1,1)
hourly.Count.plot(title='Hourly')
plt.subplot(4,1,2)
daily.Count.plot(title='Daily')
plt.subplot(4,1,3)
weekly.Count.plot(title='Weekly')
plt.subplot(4,1,4)
monthly.Count.plot(title='Monthly')

# As we aggregate more and more data, the series becomes more and more stable. For our analysis, we will use the daily timeseries
train = daily
test.index=test.Datetime
test = test.resample('D').mean()

test.head()
