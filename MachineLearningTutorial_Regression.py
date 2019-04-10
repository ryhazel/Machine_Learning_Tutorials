import pandas as pd
import quandl
import math, datetime
import numpy as np
from sklearn import preprocessing, svm, model_selection
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
import pickle

style.use('ggplot')

df = quandl.get('WIKI/GOOGL')  # getting data set from quandl on stocks
df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]  # indexing just the desired columns
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100.0  # adding a new column to df
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0  # adding new column to df

df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']] # indexing the new columns that we want

forecast_col = 'Adj. Close'
df.fillna(-99999, inplace=True)  # need to replace NaN data with a real number -- causes algorithm to treat as outlier

# predict out 10 percent of the data frame with 0.1*len(df) -->currently 35 days
forecast_out = int(math.ceil(0.01*len(df)))  # math.ceil rounds everything up to the nearest whole
print(forecast_out)

# add a label column and shifting column negatively (up the spreadsheet) to predict
df['label'] = df[forecast_col].shift(-forecast_out)

# print(df.head())  # prints first 5 rows of the data frame

X = np.array(df.drop(['label'], 1))  # drops label from df so we are left with just the features
X = preprocessing.scale(X)  # keep in mind this scales all the values -- might add computing time
X_lately = X[-forecast_out:]  # we don't have a Y value for these Xs, the last few days
X = X[:-forecast_out]
df.dropna(inplace=True)  # remove empty data
Y = np.array(df['label'])  # just taking the label

# mix data up while keeping labels with features and create train and test sets
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=0.2)


clf = LinearRegression()  # define a classifier, switching the algorithm is as easy as replacing "LinearRegression"
clf.fit(X_train, Y_train)  # this is the training where we fit features and labels
with open('linearregression.pickle', 'wb') as f:
    pickle.dump(clf, f)

pickle_in = open('linearregression.pickle', 'rb')
clf = pickle.load(pickle_in)

accuracy = clf.score(X_test, Y_test)  # see how accurate our training is against the test data

# print(accuracy)

# this is how you do a prediction
forecast_set = clf.predict(X_lately)

print(forecast_set, accuracy, forecast_out)

df['Forecast'] = np.nan  # specifies that that column is full of non-numbered data

# we need to get the date value for the x axis since our features are not the dates
last_date = df.iloc[-1].name  # iloc gets rows or columns at particular positions in the index
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

# populate the df with the new dates and the forecast values. so we can add dates to the graph
for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [i]

df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('price')
plt.show()

