import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.utils import shuffle
import sys

'''
1. Implement your own logistic regression and classify the iris data into setosa or non-setosa.
You are supposed to write your own logit function and implement gradient descent to learn optimal weights.
Then using this weight classify the entire data set as setosa or non-setosa.
You are not supposed to use logistic regression implementation of scikit learn package.
However, you are free to use SciPy package. Report how much accuracy you got.
You can try your logistic regression code on some other dataset as well for binary classification.

'''

#dataFilePath = '/Users/dbiswas/Documents/MS/datasets/iris.csv'
dataFilePath = sys.argv[1]
data = pd.read_csv(dataFilePath)

iris = datasets.load_iris()
label = [1 if i == 0 else 0 for i in iris.target]
data.insert(4,"target", label)

# learning rate
learningRate=0.001
# threshold to check condition of convergence
convergeThreshold=0.0001


# remove target value from the data frame
def filterData(data,targetColumn,columnName=[]):
    if len(columnName)==0:
        columnName= [col for col in data.columns if col not in targetColumn]
    data = shuffle(data)
    dataFrame= data[columnName]
    labelFrame=data[[targetColumn]]
    dataFrame.insert(0, "x0", [1] * len(dataFrame))
    size=len(dataFrame)
    trainingData= dataFrame[:int(size/2)]
    trainingLabel=labelFrame[:int(size/2)]
    testData=dataFrame[int(size/2):]
    testLabel=labelFrame[int(size/2):]
    return trainingData,trainingLabel,testData,testLabel

def meanNormalization(data,targetColName):
    for i in data.columns:
        if i==targetColName:
            continue
        mean=data[i].mean()
        min=data[i].min()
        max=data[i].max()
        data[i]=data[i].apply(lambda d : float(d-mean)/float(max-min))


def initializeWeightVector(allZeros,numberOfFeatures):
    if allZeros:
        return np.zeros((numberOfFeatures,1))
    else:
       return np.array(np.random.uniform(-2,2,size=numberOfFeatures)).reshape(numberOfFeatures,1)

# calculate cost after each iteration to look for convergance
def calCost(weight,data,target):
    m=len(data)
    predictedValue=1/(1+np.exp(-np.dot(data,weight)))
    return -(np.dot(target.T,np.log(predictedValue))+np.dot((1-target).T,np.log((1-predictedValue))))/m

def calAccuracy(testLabel,predictLable):
  count=0
  for i in range(len(testLabel)):
      if testLabel[i] == predictLable[i]:
          count += 1
  print("Accuracy :", (float(count) / len(testLabel))*100)


# calculate gradient
def calGradient(data,weights,target,column,learningRate):
    predition=1/(1+np.exp(-np.dot(data,weights)))
    return learningRate*((((predition - target)*(data[[column]].as_matrix())).sum()))

# second parameter is the name of the target column
def gradientDescent():
    cost_list=[]
    meanNormalization(data, "target")
    trainingData, trainingLabel, testData, testLabel=filterData(data, "target")
    weights=initializeWeightVector(True,len(trainingData.columns))
    cost=calCost(weights,trainingData,trainingLabel)
    old_cost=0
    iteration_count=0
    while abs(old_cost-cost[0])>convergeThreshold: # check for convergence
       iteration_count+=1
       old_cost=cost[0]
       grad=[]
       for i in range(len(data.columns)):
           grad.append(calGradient(trainingData, weights, trainingLabel,i, learningRate))
       weights=weights-grad
       cost=calCost(weights,trainingData,trainingLabel)
       cost_list.append(float(cost))

    print("Part 1 Starts")
    print ("Algo converged in ",iteration_count, "iteration")
    print ("final weights ",weights.T)
    prediction = 1/(1+np.exp(-np.dot(testData,weights)))
    for i in range(len(prediction)):
        prediction[i] = 1 if prediction[i]>=0.5 else 0
    print("Predicted Values: ",prediction.T)
    print("Actual Values: ",testLabel.T.values)
    calAccuracy(testLabel.values,prediction)
    return cost_list

def plotGraph(x,y):
    # Create a figure of size 8x6 inches, 80 dots per inch
    plt.figure(figsize=(8, 6), dpi=80)
    # Create a new subplot from a grid of 1x1
    plt.subplot(1, 1, 1)  # rows , column , serial number starting from 1
    # Plot cosine with a blue continuous line of width 1 (pixels)
    plt.plot(x, y, color="blue", linewidth=2.5, linestyle="-", label="cost")
    plt.show()

costList=gradientDescent()
plotGraph(range(1,len(costList)+1),costList)


'''
('Actual Values: ', array([[1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1,
        0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0,
        0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0,
        0, 0, 0, 0, 0, 1, 0, 1, 0]]))
('Accuracy :', 100.0)
'''



"""
2. You are familiar with the matrix notation and linear algebra.
Using the mathematical procedure (discussed on day 9 ) of feature selection,  implement your own feature selection algorithm to select 5 best features of OnlineNewsPopularity data set.

Report:
1. getfeatures function implements the feature selection algorithm to select n features or k features using which threshold is reached.
2. Used the algorithm mentioned in the notes of Day 9
3. With the current settings output produced is list of 5 features with minimum SSE.
"""


import pandas
import os
from numpy.linalg import inv
import numpy as np

print '\nPart2 begins'
#dataFilePath = '/Users/dbiswas/Documents/MS/datasets/OnlineNewsPopularity.csv'
dataFilePath = sys.argv[2]
data = pandas.read_csv(dataFilePath)


def Z_ScoreNormalization(data,targetColName): # we need to normalize the data so that each attribute is brought to same scale
    for i in data.columns:
        if i==targetColName: # do not modify target values
            continue
        mean=data[i].mean()
        std=data[i].std()
        data[i]=data[i].apply(lambda d : float(d-mean)/float(std)) #perform z-score normalization



def filterData(dataFrame,targetVarName,columnName=[]): # we need to remove target value from the data frame and need to put x0 values.
    if len(columnName)==0: # if column names are not given then create X with all the attributes
        columnName= [col for col in dataFrame.columns if col not in targetVarName]
    targetValues=dataFrame[targetVarName]
    data = dataFrame[columnName] # keep all column except 0th
    data.insert(0, "x0", [1] * len(dataFrame))# in 0th column set all 1 and name the column as X0 this is to accommodate biases weights i.e. w0
    return targetValues,data #return target value and modified dataframe


def calOptimalWeight(dataFrame,target):# this function calculates W'=(X^TX)^-1X^TY
    innerProduct=(dataFrame.T).dot(dataFrame) # calculating X^TX
    inverse=inv(innerProduct)# calculate (X^TX)^-1
    product=inverse.dot(dataFrame.T) # (X^TX)^-1X^T
    weight=product.dot(target)#(X^TX)^-1X^TY
    return weight #return W'


def predict(weights,X): # this function calculates Y'=W^TX or y'=XW
    predictedValue= X.dot(weights)
    return predictedValue

def calSSE(target,predicted):#calculate sum of squared error
    m=len(target)
    SSE=((np.asarray(target)-np.asarray(predicted))**2)/float(2*m)
    return sum(SSE)



Z_ScoreNormalization(data,'shares') # step 1 Normalize
label, X=filterData(data, 'shares', []) #Step 2 Filter the data i.e. seperate data from target value
threshold = 100
# columns = get_best_features(threshold, X, label, 5)
e_column = ["x0"];
ne_columns = [column for column in X.columns if column not in e_column]
minimum_sse = float('inf')
old_sse = float('inf')
while (len(ne_columns)!=0 and len(e_column) != 6):
    for col in ne_columns:
        df = X[e_column+[col]]
        weights = calOptimalWeight(df,label)
        prediction = predict(weights,df)
        temp_sse_value = calSSE(label,prediction)
        if temp_sse_value < minimum_sse:
            minimum_sse = temp_sse_value
            tempCol = col
    if old_sse - minimum_sse > threshold:
        e_column.append(tempCol)
        ne_columns = [column for column in X.columns if column not in e_column]
        old_sse = minimum_sse
    else:
        print(old_sse,minimum_sse,old_sse-minimum_sse)
        e_column.append(tempCol)
        break
print "\n****Part 2"
print "Total columns in dataset          : ",len(X.columns)
print "Total number of features selected : ", len(e_column)-1
print "Selected Features                 : ", e_column[1:]


'''
****Part 2
Total columns in dataset          :  59
Total number of features selected :  5
Selected Features                 :  ['kw_avg_avg', 'self_reference_min_shares', 'kw_max_avg', 'kw_min_avg', 'num_hrefs']
'''