from utils import *
import numpy as np
import matplotlib.pyplot as plt
import math

def augmentFeatureVector(X):
    columnOfOnes = np.zeros([len(X), 1]) + 1
    return np.hstack((columnOfOnes, X))

def computeProbabilities(X, theta):

    (num_rows, num_col) = X.shape
    n = num_rows
    (num_rows, num_col) = theta.shape
    k = num_rows
    H = np.zeros((k,n))

    for i in range(n):
         
        # calculate the constant, c
        c = -float("inf")
        for j in range(k):
            if c < np.dot(X[i,:],theta[j,:]):
                c = np.dot(X[i,:],theta[j,:])

        # calculate the summation
        summation = 0
        for j in range(k):
            summation += math.exp((np.dot(X[i,:],theta[j,:])-c))

        # calculate the vector 
        vector = np.zeros(k)
        for j in range(k):
            vector[j] += math.exp((np.dot(X[i,:],theta[j,:])-c)) 

        H[:,i] = (1.0/summation)*vector 
    return H 


def computeCostFunction(X, Y, theta, lambdaFactor):
    
    H = computeProbabilities(X, theta)


    (num_rows, num_col) = theta.shape
    k = num_rows

    (num_rows, num_col) = X.shape
    m = num_rows
    d = num_col
    summation = 0

    for i in range(m):
        for j in range(k):
            if Y[i] == j:
                summation += math.log(H[j,i])
               
    summation = -1.0/m * summation 

    newsum = 0

    for i in range(k):
        for j in range(d-1):
            newsum += theta[i,j]**2

    return summation + (lambdaFactor/2.0) * newsum


def runGradientDescentIteration(X, Y, theta, alpha, lambdaFactor):
    H = computeProbabilities(X, theta)

    (num_rows, num_col) = X.shape
    m = num_rows # number of data points
    d = num_col # number of features 

    (num_rows, num_col) = theta.shape
    k = num_rows # number of labels 

    for j in range(k):

        summation = np.zeros(d)
        for i in range(m):
            if Y[i] == j:
                correct_label = 1
            else:
                correct_label = 0

            summation += X[i,:]*(correct_label-H[Y[i],i])

        gradient = (-1.0/m*summation)+lambdaFactor*theta[j,:]
        theta[j,:] -= alpha*gradient

    return theta



def softmaxRegression(X, Y, alpha, lambdaFactor, k, numIterations):
    X = augmentFeatureVector(X)
    theta = np.zeros([k, X.shape[1]])
    costFunctionProgression = []
    for i in range(numIterations):
        costFunctionProgression.append(computeCostFunction(X, Y, theta, lambdaFactor))
        theta = runGradientDescentIteration(X, Y, theta, alpha, lambdaFactor)
    return theta, costFunctionProgression
    
def getClassification(X, theta):
    X = augmentFeatureVector(X)
    probabilities = computeProbabilities(X, theta)
    return np.argmax(probabilities, axis = 0)

def plotCostFunctionOverTime(costFunctionHistory):
    plt.plot(range(len(costFunctionHistory)), costFunctionHistory)
    plt.ylabel('Cost Function')
    plt.xlabel('Iteration number')
    plt.show()


def computeTestError(X, Y, theta):
    errorCount = 0.
    assignedLabels = getClassification(X, theta)
    return 1 - np.mean(assignedLabels == Y)