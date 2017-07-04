
#Part 1: EM implementation

import os
import pandas
from scipy.stats import multivariate_normal as mvn
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris

def Z_ScoreNormalization(data): # we need to normalize the data so that each attribute is brought to same scale
    for i in data.columns:
        mean=data[i].mean()
        std=data[i].std()
        data[i]=data[i].apply(lambda d : float(d-mean)/float(std)) #perform z-score normalization

def Expectation_Maximization(data, gauss_contributions, means, deviations):

    #Getting the shape of the data
    no_of_records, m = data.shape
    #From pis, assigning total gaussian
    total_gaussians = len(gauss_contributions)
    ll_old = 0
    threshold=0.02
    total_iterations=150
    for y in range(total_iterations):
        exp_A = []
        exp_B = []
        ll_new = 0

        # E-step
        thetas = np.zeros((total_gaussians, no_of_records))
        for x in range(len(means)):
            for y in range(no_of_records):
                thetas[x, y] = gauss_contributions[x] * mvn(means[x], deviations[x]).pdf(data[y])
        thetas /= thetas.sum(0)

        # M-step
        gauss_contributions = np.zeros(total_gaussians)
        for x in range(len(means)):
            for y in range(no_of_records):
                gauss_contributions[x] = gauss_contributions[x] + thetas[x, y]
        gauss_contributions = gauss_contributions/no_of_records

        #Updating means based on new values
        means = np.zeros((total_gaussians, m))
        for x in range(total_gaussians):
            for y in range(no_of_records):
                means[x] = means[x] + thetas[x, y] * data[y]
            means[x] = means[x]/thetas[x, :].sum()

        #Updating deviations based on new values
        deviations = np.zeros((total_gaussians, m, m))
        for x in range(total_gaussians):
            for y in range(no_of_records):
                ys = np.reshape(data[y] - means[x], (2, 1))
                deviations[x] = deviations[x] + thetas[x, y] * np.dot(ys, ys.T)
            deviations[x] = deviations[x]/thetas[x, :].sum()

        # update complete log likelihoood
        ll_new = 0.0
        for y in range(no_of_records):
            s = 0
            for x in range(total_gaussians):
                s += gauss_contributions[x] * mvn(means[x], deviations[x]).pdf(data[y])
            ll_new += np.log(s)

        if np.abs(ll_new - ll_old) < threshold:
            break
        ll_old = ll_new

    datapoint_groups = {}
    for h in range(no_of_records):
        temp = 0
        for x in range(len(means)):
            if thetas[x][h]>temp:
                temp = thetas[x][h]
                index = x
        try:
            datapoint_groups[index].append(h)
        except:
            datapoint_groups[index] = []
            datapoint_groups[index].append(h)

    return ll_new, gauss_contributions, means, deviations, datapoint_groups

relativePath=os.getcwd()
dataFilePath=relativePath+"/Resources/oldFaithful.csv"
orig_data = pandas.read_csv(dataFilePath)
data = deepcopy(orig_data)
Z_ScoreNormalization(data)
data=data.values

# initial the parameters with random values
gaussian_contributions = np.random.random(4)
gaussian_contributions /= gaussian_contributions.sum()
means = np.random.random((4, 2))
deviations = np.array([np.eye(2)] * 4)

parameters = []
# log_likelihood, final_pis, final_mus, final_sigmas, groups = Expectation_Maximization(data, gaussian_contributions, means, deviations)
parameters = Expectation_Maximization(data, gaussian_contributions, means, deviations)

count=1
final_data = []
colors = iter(['b','r','g','m'])
markers = iter(['o','s','D','v'])
plt.figure(figsize=(8, 4))
for key in parameters[-1]:
    print("\n Mean of normalized data of Group ",count," is :", parameters[2][key - 1])
    print("Deviation of normalized data: ", parameters[3][key - 1])
    count+=1

for key in parameters[-1]:
    final_normalized_data = []
    for index in parameters[-1][key]:
        final_normalized_data.append(data[index].tolist())
    col = next(colors)
    mark = next(markers)
    X = [x for [x, y] in final_normalized_data]
    Y = [y for [x, y] in final_normalized_data]
    plt.plot(X, Y, marker=mark, color=col, ls = '')


#Part 2:
#PCA Implementation

X,y = load_iris(return_X_y=True)
pca = PCA(n_components=2)
Z = pca.fit_transform(X)
plt.figure(figsize=(8, 4))
for label, col in zip((0, 1, 2),
                      ('m', 'r', 'g')):
    plt.scatter(Z[y == label, 1],
                Z[y == label, 0],
                label=label,
                c=col)
plt.legend(loc='upper left')
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.show()
