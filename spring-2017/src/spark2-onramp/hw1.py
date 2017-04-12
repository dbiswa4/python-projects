from pyspark.sql import SparkSession
from pyspark.sql import Row
#mllib library
from pyspark.mllib.regression import LabeledPoint
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import MinMaxScaler
from pyspark.ml.linalg import Vectors
from pyspark.mllib.linalg import SparseVector
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import sys
import math

spark = SparkSession.builder.config("spark.sql.warehouse.dir", "spark-workspace").appName("DS OnRamp Spark 2 - hw1").getOrCreate()
sc = spark.sparkContext

###Part 1 : Store raw data as RDD with each element of RDD representing an instance with comma delimited strings.
filename = sys.argv[1]
filename = '/FileStore/tables/u7ejaa2u1490473808422/MSD.txt'
msd = sc.textFile(filename)
msd.take(40)



###Part 2 & 3:
# Choose two features and generate a heat map for each feature on grey scale and shows variation of each feature across 40 sample instances.
# Normalize features between 0 and 1 with 1 representing darkest shade in heat map
# https://spark.apache.org/docs/latest/ml-features.html#minmaxscaler

#Converting data to LabeledPoint
def transform_to_labeled_point(line):
  values = [float(x) for x in line.split(',')]
  #return values
  return LabeledPoint(values[0], values[1:])

labeled_data = msd.map(transform_to_labeled_point)

#However, I used ml libraries for scaler. Hence, I converted to DataFrame with label and features.
lines = msd.map( lambda line : line.split(','))
msd_transformed = lines.map( lambda line: (line[0], Vectors.dense(line[1:])) )
print 'msd_transformed type : ', type(msd_transformed)
#Conver it to DataFrame
msd_labeled_df = spark.createDataFrame(msd_transformed,["label", "features"])
msd_labeled_df.show(2)

#Experiment with minmaxscaler : https://spark.apache.org/docs/latest/ml-features.html#minmaxscaler

scaler = MinMaxScaler(inputCol="features", outputCol="scaled_features")

# Compute summary statistics and generate MinMaxScalerModel for 40 samples
scaler_model = scaler.fit(msd_labeled_df.limit(40))
#scaler_model = scaler.fit(msd_labeled_df)

# rescale each feature to range [min, max].
scaled_data = scaler_model.transform(msd_labeled_df)
print("Features scaled to range: [%f, %f]" % (scaler.getMin(), scaler.getMax()))
scaled_data.select("features", "scaled_features").show(2)
scaled_data.show(1, False)

#ToDo : Select any two features and plot heat map

###Part 4
#In learning problem, its natural to shift labels if its not starting from zero. Find out the range of prediction year and shift labels if necessary so that lowest one starts from zero.
#Hint: If year ranges from 1900-2000 then you can also represent year between 0-100 as well by subtracting minimum year.

labels = scaled_data.rdd.map(lambda x : x.label)
minLabel = labels.min()
scaled_data.count()
print 'minLabel : ', minLabel
shifted_scaled_data = scaled_data.withColumn("shifted_label", scaled_data.label - minLabel)
shifted_scaled_data.show(20)


###Part 5
#Split dataset into training, validation and test set.
#Create a baseline model where we always provide the same prediction irrespective of our input. (Use training data)
#Implement a function to give Root mean square error given a RDD.
#Measure our performance of base model using it. (Use test data)
#Hint 1: Intro to train, validation, test set, https://en.wikipedia.org/wiki/Test_set
#Hint 2: Root mean squared error - https://en.wikipedia.org/wiki/Root-mean-square_deviation

shifted_scaled_data_cols = shifted_scaled_data.select("shifted_label","scaled_features").toDF("label","features")
(training_data, test_data) = shifted_scaled_data_cols.randomSplit([0.7, 0.3])
print 'Training Data Count : ', training_data.count()
print 'Test Data Count     : ', test_data.count()
training_data.show(2)

#Build the model on training data
#https://spark.apache.org/docs/latest/ml-classification-regression.html
lr = LinearRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8)
lr_model = lr.fit(training_data)

#Print the metrics
print 'Coefficients : ' + str(lr_model.coefficients)
print 'Intercept    : ' + str(lr_model.intercept)

#Predict on the training data
predictions = lr_model.transform(training_data)
predictions.select("prediction","label","features").show()

#Predict on the test data
predictions = lr_model.transform(test_data)
predictions.select("prediction","label","features").show(10)

# Summarize the model over the training set and print out some metrics
trainingSummary = lr_model.summary
print("numIterations: %d" % trainingSummary.totalIterations)
print("objectiveHistory: %s" % str(trainingSummary.objectiveHistory))
trainingSummary.residuals.show()
print("RMSE: %f" % trainingSummary.rootMeanSquaredError)
print("r2: %f" % trainingSummary.r2)

###Part5 Contd...
#Implement a function to give Root mean square error given a RDD. Measure our performance of base model using it. (Use test data)
#Evaluate the model on training data

predictions_actuals = predictions.select("label", "prediction")
MSE = predictions_actuals.rdd.map(lambda (v, p): (v - p)**2).reduce(lambda x, y: x + y) / predictions_actuals.count()
print 'Mean Squared Error = ' + str(MSE)
RMSE = math.sqrt(MSE)
print 'Root mean square error = ' + str(RMSE)

###Part 6
#Visualize predicted vs actual using a scatter plot.
actual = np.asarray(predictions_actuals.select('label').collect())
predicted = np.asarray(predictions_actuals.select('prediction').collect())
print actual
print predicted

plt.scatter(predicted, actual, alpha=0.5)
plt.title('Year : Predicted vs Actual')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

