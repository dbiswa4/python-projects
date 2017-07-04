from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext
from pyspark.mllib.linalg import Vectors
from pyspark.ml.feature import MinMaxScaler
from pyspark.mllib.regression import LabeledPoint
import matplotlib.pyplot as plt
import sys
import numpy as np

#Create spark context
sparkConf = SparkConf().setAppName("Spark2 Practice Assignment 1")
sc = SparkContext(conf=sparkConf)
sqlContext = SQLContext(sc)

'''
Part 1

The data is stored in text file as comma separated values.
- Store raw data as RDD with each element of RDD representing an instance with comma delimited strings.
- Count the number of data points we have. Print the list of first 40 instances.
'''
input_file = sys.argv[1]
input = sc.textFile(input_file)
print 'Count of records in file : ', input.count()
print '********Sample 40 records*********** \n', input.take(40)


'''
Part 2
We need to store data as LabeledPoint in order to train our model. Write a function which takes an input string, parses it and return a LabeledPoint.
Hint: https://spark.apache.org/docs/latest/mllib-data-types.html#labeled-point
'''
#LabeledPoint transformation
def convert_to_labeled_point(line):
    values = [float(x) for x in line.split(',')]
    return LabeledPoint(values[0], values[1:])

input_labelled_points = input.map(convert_to_labeled_point)
print '**************Converted to labeled point************* \n', input_labelled_points.take(5)


'''
Part 3
- Choose two features and generate a heat map for each feature on grey scale and shows variation of each feature across 40 sample instances.
- Normalize features between 0 and 1 with 1 representing darkest shade in heat map.
Hint: https://spark.apache.org/docs/latest/ml-features.html#minmaxscaler
'''

lines = input.map(lambda line : line.split(','))
transformed = lines.map(lambda line: (line[0], Vectors.dense(line[1:])))
labelled_dataframe = sqlContext.createDataFrame(transformed, ["label", "features"])
scalar = MinMaxScaler(inputCol="features", outputCol="features_scaled")
scalar_mod = scalar.fit(labelled_dataframe.limit(40))
scaled_data = scalar_mod.transform(labelled_dataframe)
print '******Scaled Features******* : \n', scaled_data.show(5, False)


heat1 = np.asarray(labelled_dataframe.rdd.map(lambda f: (float(f.features[1]), float(f.features[2]))).take(40))
plt.imshow(heat1, cmap='gray')
plt.show()

heat2 = np.asarray(scaled_data.rdd.map(lambda f: (float(f.features[1]), float(f.features[2]))).take(40))
plt.imshow(heat2, cmap='gray')
plt.show()


'''
Part 4
In learning problem, its natural to shift labels if its not starting from zero. Find out the range of prediction year and shift labels if necessary so that lowest one starts from zero.
Hint: If year ranges from 1900-2000 then you can also represent year between 0-100 as well by subtracting minimum year.
'''
labels = scaled_data.rdd.map(lambda x : x.label)
minimum_label = labels.min()
scaled_data.count()
print 'minLabel : ', minimum_label
shifted_scaled_data = scaled_data.withColumn("label_shifted", scaled_data.label - minimum_label)
shifted_scaled_data.show(20)

"""
Part 5
Split dataset into training, validation and test set.
Create a baseline model where we always provide the same prediction irrespective of our input. (Use training data)
Implement a function to give Root mean square error given a RDD.
Measure our performance of base model using it. (Use test data)
Hint 1: Intro to train, validation, test set, https://en.wikipedia.org/wiki/Test_set
Hint 2: Root mean squared error - https://en.wikipedia.org/wiki/Root-mean-square_deviation
"""

def get_rmse(actual_prediction):
    return np.sqrt(actual_prediction.map(lambda t: (t[0] - t[1]) ** 2).mean())

shifted_scaled_data_cols = shifted_scaled_data.select("label_shifted","features_scaled").toDF("label","features")
(training_data, validation_data, test_data) = shifted_scaled_data_cols.randomSplit([0.7, 0.2, 0.1])
training_data_count = training_data.count()
print 'Training Data Count : ', training_data_count

validation_data_count = validation_data.count()
print 'Validation Data Count      : ', validation_data_count

test_data_count = test_data.count()
print 'Test Data Count     : ', test_data_count

print 'Training data       : \n', training_data.show(2)

base_model = (training_data.map(lambda p: p.label).mean())
train_actual_pred = training_data.rdd.map(lambda points: (points.label, base_model))
print 'train_actual_prediction : ', train_actual_pred.take(20)

val_actual_pred = validation_data.rdd.map(lambda points: (points.label, base_model))
print '******val actual prediction******: ', val_actual_pred.take(20)

test_actual_pred = test_data.rdd.map(lambda points: (points.label, base_model))
print 'test_actual_prediction  : ', test_actual_pred.take(20)

test_rmse = get_rmse(test_actual_pred)
print '*******Test RMSE ******         : ', test_rmse

"""
Part 6
Visualize predicted vs actual using a scatter plot.
"""

actual_value = np.asarray(test_actual_pred.map(lambda x : x[0]).collect())
predicted = np.asarray(test_actual_pred.map(lambda x : x[1]).collect())
plt.scatter(actual_value, predicted, alpha=0.5)
plt.title('Predicted vs Actual Year')
plt.xlabel('predicted values')
plt.ylabel('actual values')
plt.show()
