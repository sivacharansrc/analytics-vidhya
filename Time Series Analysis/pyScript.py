
# Importing libraries
import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
pd.set_option('display.max_rows', 10)
pd.set_option('display.max_columns', 10)
pd.set_option('expand_frame_repr', False)

# %matplotlib inline

sns.set_palette("cubehelix")  # Other palettes: "Set2", "husl", "cubehelix", "hls"

# Reading the input file
os.chdir("C:\\Users\\sivac\\Documents\\Analytics\\analytics-vidhya\\")
cwd = os.getcwd()

train = cwd + "\\Analytics Vidhya\\Time Series Analysis\\input\\train.csv"
train = pd.read_csv(train)

test = cwd + "\\Analytics Vidhya\\Time Series Analysis\\input\\test.csv"
test = pd.read_csv(test)

# Visualizing the first few components of train dataset
train.head(24)

train.dtypes

# Converting the date time object

train.Datetime = pd.to_datetime(train['Datetime'], format="%d-%m-%Y %H:%M")
# train.index = train.Datetime

# Plotting the data

plt.figure(figsize=(20, 8))
plt.plot(train['Datetime'], train['Count'])
plt.show()

# ### From the above image, it is evident that the sales has definitely increased over the period, and there is a gradual trend from 2012 through 2014
# Let us now see the behavior for the first 5 days

five_days = train.iloc[0:120, ]

plt.figure(figsize=(20, 8))
plt.plot(five_days['Datetime'], five_days['Count'], marker="o")


# Reference link
# https://www.analyticsvidhya.com/blog/2018/10/predicting-stock-price-machine-learningnd-deep-learning-techniques-python/
# https://www.analyticsvidhya.com/blog/2018/09/non-stationary-time-series-python/

# Pushing the changes
! cd C: \\Users\\sivac\\Documents\\Analytics\\Analytics Vidhya
! git add .
! git commit - m "AV-TSA-Adding Exploratory Analysis"
! git push
