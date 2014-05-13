# -*- coding: utf-8 -*-

#Require the necessary modules
import time
import csv as csv
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.externals import joblib
from sklearn import cross_validation
from sklearn import datasets
from sklearn.metrics import mean_squared_error
from math import sqrt

###############################################################
#Reads features from training data

features_raw = pd.read_csv("/Users/jeanettepettibone/AllFeatures.csv", delimiter=',')
features = np.array(features_raw)
print features.shape

###############################################################

###############################################################
#Reads results from training data

data_raw = pd.read_csv("/Users/jeanettepettibone/training_solutions_rev1.csv", delimiter=',',index_col=0)
data = np.array(data_raw)
print data.shape

###############################################################

###############################################################
#Reads features from test data

test_raw = pd.read_csv("/Users/jeanettepettibone/TestAllFeatures.csv", delimiter=',')
test = np.array(test_raw)
print test.shape

###############################################################

###############################################################
#Generates random forest

print 'fitting ...'
rfr = RandomForestRegressor(n_estimators=100, max_features=50)
featureFitting = features[0:61578, 1:406]
fitData = data[0:61578]
rfr.fit(featureFitting, fitData)

###############################################################

###############################################################
#Predicts results for test data

print 'predicting ...'
predictData = test[0:79975, 1:406]
predict = rfr.predict(predictData)
joblib.dump(predict, 'Solutions/output-predict-50MaxFeat')

###############################################################

###############################################################
# Write predictions to csv

print 'writing prediction ...'
galIDs = test[:,0].astype(int)
headerRow = 'GalaxyID,Class1.1,Class1.2,Class1.3,Class2.1,Class2.2,Class3.1,Class3.2,Class4.1,Class4.2,Class5.1,Class5.2,Class5.3,Class5.4,Class6.1,Class6.2,Class7.1,Class7.2,Class7.3,Class8.1,Class8.2,Class8.3,Class8.4,Class8.5,Class8.6,Class8.7,Class9.1,Class9.2,Class9.3,Class10.1,Class10.2,Class10.3,Class11.1,Class11.2,Class11.3,Class11.4,Class11.5,Class11.6'
formatting = '%d,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f'
prediction = np.column_stack((galIDs, predict))

np.savetxt('Solutions/predtest-100-50.csv', prediction, fmt=formatting, delimiter=',', newline='\n', header=headerRow, footer='', comments='')

print "done!"

###############################################################
