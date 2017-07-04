from pyspark import SparkContext, SparkConf
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.feature import Normalizer
import matplotlib.pyplot as plt
import sys
import numpy as np


sparkConf = SparkConf().setAppName("Onramp Spark 2 - Assignment 2")
sc = SparkContext(conf=sparkConf)

fn = sys.argv[1]
input = sc.textFile(fn)

#Q.1
'''
Write a function that computes the value (wTx - y) x and test this function on two examples.

Args:
weights (DenseVector): An array of model weights (betas).
lp (LabeledPoint): The `LabeledPoint` for a single observation.
'''

def loss_function(w, lp):
    x = lp.features
    wTx = w.dot(x)
    y = lp.label
    return np.dot((wTx - y) , x)

#1st Test
w1 = Vectors.dense([2.0, 4.0, 7.0])
lp = LabeledPoint(2.0, [1.0, 1.0,1.0])
print 'Test loss function 1 : ', loss_function(w1, lp)

#2nd test
w2 = Vectors.dense([2.0, 1.0, 1.0])
lp = LabeledPoint(7.0, [4.0, 2.0,1.0])
print 'Test loss function 2 : ', loss_function(w2, lp)




#Q.2
'''
Implement a function that takes in weight and LabeledPoint instance and returns a <label, prediction tuple>

Args:
weights (np.ndarray): An array with one weight for each features in `trainData`.
observation (LabeledPoint): A `LabeledPoint` that contain the correct label and the
features for the data point.
'''
def actual_and_prediction(w, lp):
    return (lp.label, w.dot(lp.features))

#1st Test
w1 = Vectors.dense([2.0, 4.0, 7.0])
lp = LabeledPoint(2.0, [1.0, 1.0,1.0])
print 'Actual and Prediction 1 : ', actual_and_prediction(w1, lp)

#2nd test
w2 = Vectors.dense([2.0, 1.0, 1.0])
lp = LabeledPoint(7.0, [4.0, 2.0,1.0])
print 'Actual and Prediction 2 : ', actual_and_prediction(w2, lp)




#Q.3
'''
Implement a gradient descent function for linear regression. Test this function on an example.

Args:
trainData (RDD of LabeledPoint): The labeled data for use in training the model.
numIters (int): The number of iterations of gradient descent to perform.
'''

def rmse(actual_prediction):
    return np.sqrt(actual_prediction.map(lambda t: (t[0] - t[1]) ** 2).mean())

def grad_desc(train_data, itrs):
    features_count = len(train_data.take(1)[0].features)
    weights = np.zeros(features_count)
    n = train_data.count()
    learn_rate = 1.0
    train_error = [0] * itrs
    for i in range(itrs):
        print 'Iteration # ', i + 1
        train_label_and_pred = train_data.map(lambda lp: actual_and_prediction(weights, lp))
        train_error[i] = rmse(train_label_and_pred)
        grad = train_data.map(lambda lp: loss_function(weights, lp)).sum()
        learn_rate_temp = learn_rate / (n * np.sqrt(i + 1))
        weights -= learn_rate_temp * grad
    return weights, train_error


test_grad_desc_rdd = sc.parallelize([LabeledPoint(0.9, np.array([0.2, 0.1, 0.4])), \
                                     LabeledPoint(0.5, np.array([0.3, 0.2, 0.1])), \
                                     LabeledPoint(0.8, np.array([0.3, 0.2, 0.6]))])

#Test Gradient Descent
new_weigths, new_train_error = grad_desc(test_grad_desc_rdd, 50)
gd_actual_and_pred = test_grad_desc_rdd.map(lambda lp: actual_and_prediction(new_weigths, lp))
rmse_test = rmse(gd_actual_and_pred)
print 'Actual and Prediction on sample data : ', gd_actual_and_pred.collect()
print 'RMSE for sample data : ', rmse_test

'''
RMSE for sample data                 :  0.189028156723
'''



#Q.4
'''
Train our model on training data and evaluate the model based on validation set.
'''

#Converting data to LabeledPoint
def convert_to_labeled_point(line):
  values = [float(x) for x in line.split(',')]
  #return values
  return LabeledPoint(values[0], values[1:])

input_labeled = input.map(convert_to_labeled_point)
labels = input_labeled.map(lambda x : x.label)
features = input_labeled.map(lambda x: x.features)
min_label = labels.min()
scaled_label_input = input_labeled.map(lambda lp: LabeledPoint(lp.label - min_label, lp.features))
scaled_labels = scaled_label_input.map(lambda x: x.label)
norm = Normalizer()
normalized_input = scaled_labels.zip(norm.transform(features))
normalized_lp_input = normalized_input.map(lambda x: LabeledPoint(x[0], x[1]))

print 'Min year : ', min_label
print 'Scaled data         : ', normalized_lp_input.take(4)

#Split the dataset
train_data, validation_data, test_data = normalized_lp_input.randomSplit([.7, .2, .1], 50)
itr = 50
train_weights, train_error = grad_desc(train_data, itr)
train_actual_and_pred = train_data.map(lambda lp: actual_and_prediction(train_weights, lp))
train_rmse = rmse(train_actual_and_pred)
val_actual_and_pred = validation_data.map(lambda lp: actual_and_prediction(train_weights, lp))
val_rmse = rmse(val_actual_and_pred)

print 'RMSE for training data : ', train_rmse
print 'RMSE for validation data : ', val_rmse


#Q.5
'''
Visualize the log of the training error as a function of iteration. The scatter plot visualizes the logarithm of the
training error for all 50 iterations.
'''

plt.scatter(range(0, 50), np.log(train_error), alpha=0.5)
plt.title('Iteration and Training Error Plot')
plt.xlabel('Iteration #')
plt.ylabel('Training error')
plt.show()


#Q.6
'''
Use this model for prediction on test data. Calculate Root Mean Square Error of our model.
'''

actual_and_pred_test = test_data.map(lambda lp: actual_and_prediction(train_weights, lp))
rmse_test = rmse(actual_and_pred_test)
print 'RMSE for test data : ', rmse_test

'''
RMSE for training data :  13.5765998113
RMSE for validation data :  14.0917030378
RMSE for test data :  12.4618219227
'''

##################End of Script#####################























