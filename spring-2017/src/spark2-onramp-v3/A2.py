import sys
import numpy as np
from pyspark import SparkContext, SparkConf
from pyspark.mllib.linalg import Vectors
import matplotlib.pyplot as plt
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.feature import Normalizer

spark_conf = SparkConf().setAppName("Spark 2 practice - Assignment 2")
sp_context = SparkContext(conf=spark_conf)
file_name = sys.argv[1]
data = sp_context.textFile(file_name)

"""
Part 1

Write a function that computes the value (wTx - y) x and test this function on two examples.

Args:
weights (DenseVector): An array of model weights (betas).
lp (LabeledPoint): The `LabeledPoint` for a single observation.
"""


def loss_function(weights, lp):
    wTx = weights.dot(lp.features)
    return np.dot((wTx - lp.label), lp.features)

#Test 1
weights = Vectors.dense([1.0, 2.0, 3.0, 4.0])
lp = LabeledPoint(10.0, [2.0, 1.0, 4.0, 5.0])
print 'Test 1 :\n Weights:\n',weights,'Labeled Point:\n',lp,'Loss: ',loss_function(weights, lp)

#Test 2
weights = Vectors.dense([2.0, 1.0, 1.0, 6.7])
lp = LabeledPoint(6.0, [3.0, 2.0,1.0, 2.5])
print 'Test 2 :\n Weights:\n',weights,'Labeled Point:\n',lp,'Loss: ',loss_function(weights, lp)


"""
Part 2

Implement a function that takes in weight and LabeledPoint instance and returns a <label, prediction tuple>

Args:
weights (np.ndarray): An array with one weight for each features in `trainData`.
observation (LabeledPoint): A `LabeledPoint` that contain the correct label and the
features for the data point.
"""


def prediction(weights, lp):
    return lp.label, weights.dot(lp.features)

#Test 1
weights = Vectors.dense([1.0, 2.0, 3.0, 4.0])
lp = LabeledPoint(10.0, [2.0, 1.0, 4.0, 5.0])
print 'Test 1 :\n Weights:\n',weights,'Labeled Point:\n',lp,'Prediction: ',prediction(weights, lp)

#Test 2
weights = Vectors.dense([2.0, 1.0, 1.0, 6.7])
lp = LabeledPoint(5.0, [3.0, 2.0,1.0, 2.5])
print 'Test 2 :\n Weights:\n',weights,'Labeled Point:\n',lp,'Prediction: ',prediction(weights, lp)


"""
Part 3

Implement a gradient descent function for linear regression. Test this function on an example.

Args:
trainData (RDD of LabeledPoint): The labeled data for use in training the model.
numIters (int): The number of iterations of gradient descent to perform.
"""

def calculate_rmse(actual_prediction):
    return np.sqrt(actual_prediction.map(lambda x: (x[0] - x[1])**2).mean())

def gradientDescent(trainData, numIters):
    m = trainData.count()
    print 'Total training data: ', m
    n = len(trainData.take(1)[0].features)
    print 'Total features: ', n
    weights = np.zeros(n)
    learn_rate = 1.0
    error_vector = [0] * numIters
    for i in range(numIters):
        print 'Iteration # ', i + 1, '    Error:', error_vector[i]
        predicted_values = trainData.map(lambda r: prediction(weights, r))
        error_vector[i] = calculate_rmse(predicted_values)
        grad = trainData.map(lambda r: loss_function(weights, r)).sum()
        temp_learn_rate = learn_rate / (m * np.sqrt(i + 1))
        weights -= temp_learn_rate * grad
    return weights, error_vector


test_rdd = sp_context.parallelize([LabeledPoint(0.9, np.array([0.2, 0.1, 0.4, 0.7])),
                                   LabeledPoint(0.5, np.array([0.3, 0.2, 0.1,0.5])),
                                   LabeledPoint(0.7, np.array([0.3, 0.2, 0.6,0.9]))])

#Testing
test_weights, test_error = gradientDescent(test_rdd, 30)
print 'Test Weights : ', test_weights
print 'Error in Final Iteration: ', test_error[-1]


"""
Part 4

Train our model on training data and evaluate the model based on validation set.
"""

#Converting data to LabeledPoint
def getDataAndLabel(line):
  values = [float(x) for x in line.split(',')]
  return LabeledPoint(values[0], values[1:])

labelled_data = data.map(getDataAndLabel)
labels = labelled_data.map(lambda x : x.label)
features = labelled_data.map(lambda x: x.features)
minYear = labels.min()
scaledLabeleData = labelled_data.map(lambda m: LabeledPoint(m.label - minYear, m.features))
scaledLabels = scaledLabeleData.map(lambda x: x.label)
normalizer = Normalizer()
normalized_data = scaledLabels.zip(normalizer.transform(features))
normalized_lp_data = normalized_data.map(lambda x: LabeledPoint(x[0], x[1]))

#Splitting the dataset into training, validation, test sets
trainSplit, valSplit, testSplit = normalized_lp_data.randomSplit([.6, .2, .2], 50)
iteration = 50
trainWeights, trainError = gradientDescent(trainSplit, iteration)
trainingResult = trainSplit.map(lambda x: prediction(trainWeights, x))
trainRMSE = calculate_rmse(trainingResult)
valResult = valSplit.map(lambda y: prediction(trainWeights, y))
valRMSE = calculate_rmse(valResult)

print 'Error in Training data: ', trainRMSE
print 'Error in Validation data: ', valRMSE


"""
Part 5

Visualize the log of the training error as a function of iteration. The scatter plot visualizes the logarithm of the training error for all 50 iterations
"""

plt.scatter(range(0, 50), np.log(trainError), alpha=0.5)
plt.title('Iteration Vs. RMSE')
plt.xlabel('Iteration')
plt.ylabel('RMSE')
plt.show()


"""
Part 6

Use this model for prediction on test data. Calculate Root Mean Square Error of our model.
"""

test_result = testSplit.map(lambda t: prediction(trainWeights, t))
test_rmse = calculate_rmse(test_result)
print 'Error in Test data prediction: ', test_rmse

























