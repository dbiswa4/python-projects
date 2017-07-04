import sys
import numpy as np
from pyspark import SparkContext, SparkConf
from pyspark.mllib.linalg import Vectors, DenseVector
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.feature import Normalizer
from matplotlib.cm import get_cmap
from matplotlib.colors import ListedColormap, Normalize
import matplotlib.pyplot as plt


#Create Spark Context
sparkConf = SparkConf().setAppName("DS OnRamp Spark 2 - hw2")
sc = SparkContext(conf=sparkConf)

filename = sys.argv[1]
msd = sc.textFile(filename)

###Q.1
#Write a function that computes the value (wTx - y) x and test this function on two examples.

def loss_fn(weights, lp):
    wTx = weights.dot(lp.features)
    return np.dot((wTx - lp.label) , lp.features)
#Test 1
weights = Vectors.dense([1.0, 2.0, 3.0])
lp = LabeledPoint(10.0, [2.0, 1.0,4.0])
print 'Test#1 : ', loss_fn(weights, lp)

#Test 2
weights = Vectors.dense([2.0, 1.0, 1.0])
lp = LabeledPoint(5.0, [3.0, 2.0,1.0])
print 'Test#2 : ', loss_fn(weights, lp)


###Q.2
#Implement a function that takes in weight and LabeledPoint instance and returns a <label, prediction tuple>
def get_actual_and_prediction(weights, lp):
    pred = weights.dot(lp.features)
    return (lp.label, pred)

#Test 1
weights = Vectors.dense([1.0, 2.0, 3.0])
lp = LabeledPoint(10.0, [2.0, 1.0,4.0])
print 'Prediction#1 : ', get_actual_and_prediction(weights, lp)

#Test 2
weights = Vectors.dense([2.0, 1.0, 1.0])
lp = LabeledPoint(5.0, [3.0, 2.0,1.0])
print 'Prediction#2 : ', get_actual_and_prediction(weights, lp)

#Test 3
weights = np.array([2.0, 1.0, 1.0])
lp = LabeledPoint(5.0, [3.0, 2.0,1.0])
print 'Prediction#3 : ', get_actual_and_prediction(weights, lp)




###Q.3
#Implement a gradient descent function for linear regression. Test this function on an example.
#Args:
#trainData (RDD of LabeledPoint): The labeled data for use in training the model.
#numIters (int): The number of iterations of gradient descent to perform.

def get_rmse(actual_prediction):
    return np.sqrt(actual_prediction.map(lambda t: (t[0] - t[1]) ** 2).mean())

#Args
#train_data (RDD of LabeledPoint)
#num_iters (int)
def gradient_descent(train_data, num_iters):
    n = train_data.count()
    print 'train_data.take(1)[0] : ', train_data.take(1)[0]
    print 'train_data.take(1)[0].features : ', train_data.take(1)[0].features
    features_count = len(train_data.take(1)[0].features)
    print 'features_count : ', features_count
    theta = np.zeros(features_count)
    learn_rate = 1.0
    train_error = [0] * num_iters
    for i in range(num_iters):
        train_label_and_pred = train_data.map(lambda lp: get_actual_and_prediction(theta, lp))
        train_error[i] = get_rmse(train_label_and_pred)
        grad = train_data.map(lambda lp: loss_fn(theta, lp)).sum()
        temp_learn_rate = learn_rate / (n * np.sqrt(i + 1))
        theta -= temp_learn_rate * grad
        print 'Iteration completed # ', i + 1
    return theta, train_error


test_lp_rdd = sc.parallelize([LabeledPoint(0.9, np.array([0.2, 0.1,0.4])),LabeledPoint(0.5, np.array([0.3, 0.2,0.1])), LabeledPoint(0.7, np.array([0.3, 0.2,0.6])), LabeledPoint(0.8, np.array([0.3, 0.2,0.6]))])

#Test Gradient Descent
new_weigths, new_train_error = gradient_descent(test_lp_rdd, 50)
print 'new_weigths : ', new_weigths
gd_actual_and_pred = test_lp_rdd.map(lambda lp: get_actual_and_prediction(new_weigths, lp))
test_rmse = get_rmse(gd_actual_and_pred)
print 'Actual and Prediction by Gradient Descent : ', gd_actual_and_pred.collect()
print 'Sample Test rmse     : ', test_rmse

#Output
#learn_rate = 1.0
#Sample Test rmse     :  0.199968420308
#learn_rate = 0.5
#Test rmse     :
#Sample Test rmse     :  0.220144043916



###Q.4
#Train our model on training data and evaluate the model based on validation set.

#Converting data to LabeledPoint
def transform_to_labeled_point(line):
  values = [float(x) for x in line.split(',')]
  #return values
  return LabeledPoint(values[0], values[1:])

msd_labeled = msd.map(transform_to_labeled_point)
labels = msd_labeled.map(lambda x : x.label)
features = msd_labeled.map(lambda x: x.features)
min_label = labels.min()
scaled_label_msd = msd_labeled.map(lambda lp: LabeledPoint(lp.label - min_label, lp.features))
scaled_labels = scaled_label_msd.map(lambda x: x.label)
scaled_min_label = scaled_labels.min()
norm = Normalizer()
normalized_msd = scaled_labels.zip(norm.transform(features))
normalized_lp_msd = normalized_msd.map(lambda x: LabeledPoint(x[0], x[1]))

print 'Min label                 : ', min_label
print 'Scaled Min label          : ', scaled_min_label
print 'Data with scaled label    : ', scaled_label_msd.take(2)
print 'Data with scaled features : ', normalized_msd.take(2)
print 'Normalized data           : ', normalized_lp_msd.take(2)

#Output
#Min label                 :  1930.0

#Split the dataset
train_data, validation_data, test_data = normalized_lp_msd.randomSplit([.7, .2, .1], 50)
itr = 50
train_weights, train_error = gradient_descent(train_data, itr)
train_actual_and_pred = train_data.map(lambda lp: get_actual_and_prediction(train_weights, lp))
train_rmse = get_rmse(train_actual_and_pred)
val_actual_and_pred = validation_data.map(lambda lp: get_actual_and_prediction(train_weights, lp))
val_rmse = get_rmse(val_actual_and_pred)

print 'Training rmse   : ', train_rmse
print 'Validation rmse : ', val_rmse


###Q.5
#Visualize the log of the training error as a function of iteration. The scatter plot visualizes the logarithm of the
#training error for all 50 iterations.

plt.scatter(range(0, 50), np.log(train_error), alpha=0.5)
plt.title('Iteration and Training Error')
plt.xlabel('Iteration')
plt.ylabel('Training Error')
plt.show()


###Q.6
#Use this model for prediction on test data. Calculate Root Mean Square Error of our model.

test_actual_and_pred = test_data.map(lambda lp: get_actual_and_prediction(train_weights, lp))
test_rmse = get_rmse(test_actual_and_pred)
print 'Test rmse : ', test_rmse

'''
With learn_rate = 1.0
Training rmse   :  13.5765998113
Validation rmse :  14.0917030378
Test rmse :  12.4618219227

With learn_rate = 0.5
Training rmse   :  13.6903332718
Validation rmse :  14.1841443527
Test rmse :  12.6113911096
'''

##################End of Script#####################























