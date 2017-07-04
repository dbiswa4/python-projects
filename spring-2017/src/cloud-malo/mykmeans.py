from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext
from pyspark.mllib.clustering import KMeans
from pyspark.mllib.feature import Normalizer
from numpy import array
from math import sqrt
import sys
import datetime
import time


#Create Spark Context
sparkConf = SparkConf().setAppName("Cloud Computing - KMeans")
sc = SparkContext(conf=sparkConf)
sqlContext = SQLContext(sc)

'''
Dataset:
https://archive.ics.uci.edu/ml/machine-learning-databases/00292/
'''

#Data Load
start_time = time.time()
filename = sys.argv[1]
data = sc.textFile(filename)
data_no_hdr = data.filter(lambda x: "Channel" not in x)

print 'data with header : \n', data.take(5)
print 'Data w/o header  : \n', data_no_hdr.take(5)


#Data Transformation and Normalization
parsedData = data_no_hdr.map(lambda line: array([float(x) for x in line.split(',')]))

print 'Fields converted to float: \n', parsedData.take(5)

# Build the model (cluster the data)
clusters = KMeans.train(parsedData, 3, maxIterations=50, initializationMode="random")

print 'Centroids:'
for center in clusters.centers:
  print center

centroid_calc_time = time.time()

print("Time to calculate centroids : --- %s seconds ---" % (time.time() - start_time))

def error(point):
    center = clusters.centers[clusters.predict(point)]
    return sqrt(sum([x**2 for x in (point - center)]))

WSSSE = parsedData.map(lambda point: error(point)).reduce(lambda x, y: x + y)
print("Within Set Sum of Squared Error = " + str(WSSSE))

print("Time to calculate WSSSE : --- %s seconds ---" % (time.time() - centroid_calc_time))
