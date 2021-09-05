import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def logisticSigmoidFun(q, Theta):
    temp = np.dot(q, Theta)
    exp = (np.exp(-1 * temp)) + 1
    g = 1.0 / exp
    return g


def logisticCost(X, Theta, Y, m):
    h = logisticSigmoidFun(X, Theta)
    log1 = np.log10(h)
    temp = np.multiply(Y, log1)
    log2 = np.log10(1 - h)
    temp2 = np.multiply(1 - Y, log2)
    J = (np.sum(temp + temp2) * -1) / m
    return J

def MSE(X, Theta, Y,m):
    temp = logisticSigmoidFun(X, Theta) - Y
    squares = np.square(temp)
    MSE = np.sum(squares) / m
    return MSE

def logisticGradientDescent(initialTheta, X, Y, alpha, numOfIterations, m):
    costInIterations = []
    mseInIterations = []
    Theta = initialTheta
    for i in range(numOfIterations):
        costInIterations.append(logisticCost(X, Theta, Y, m))
        print('Cost in iteration ', i, ' = ', costInIterations[i])
        mseInIterations.append(MSE(X, Theta, Y, m))
        print('MSE in iteration ', i, ' = ', mseInIterations[i])
        num1 = logisticSigmoidFun(X, Theta)
        s = np.multiply(num1 - Y, X)
        tTheta = np.zeros((len(X[0]), 1))
        for j in range(len(X[0])):
            for k in range(len(X)):
                tTheta[j][0] += s[k][j]
        m = len(Y)
        num = alpha / m
        tempTheta = (Theta - (num * tTheta))
        Theta = tempTheta
    print('Theta= ',Theta)
    return Theta, costInIterations,mseInIterations


def runLogisticReg():
    size = 303
    numOfIterations = 1000
    dataFile = pd.read_csv("heart.csv", index_col=0)
    x = dataFile[["trestbps", "chol", "thalach", "oldpeak"]]
    y = dataFile[["target"]]
    #X = np.array(x[:size])
    Y = np.array(y[:size])
    XNormalized = (x - x.mean()) / x.std()
    X = XNormalized
    X = np.c_[np.ones(len(X)), X]
    Theta = np.zeros((len(X[0]), 1))
    m = len(Y)
    a=0
    b=0
    Theta, cost,mse = logisticGradientDescent(Theta, X, Y, 0.00005, numOfIterations, m)
    h = logisticSigmoidFun(X, Theta)
    for i in range(len(X)):
        if (h[i] >= 0.5):
            if(1==Y[i]):
                a += 1
            else:
                b += 1
            #print("have heart disease")
        else:
            if (0 == Y[i]):
                a += 1
            else:
                b += 1
            #print("not have heart disease")
    print("accuracy when alpha= ",0.00005," is: ",(a/303*100))
    a=0
    b=0
    Theta, cost,mse = logisticGradientDescent(Theta, X, Y, 0.0005, numOfIterations, m)
    h = logisticSigmoidFun(X, Theta)
    for i in range(len(X)):
        if (h[i] >= 0.5):
            if (1 == Y[i]):
                a += 1
            else:
                b += 1
            #print("have heart disease")
        else:
            if (0== Y[i]):
                a += 1
            else:
                b += 1
            #print("not have heart disease")
    print("accuracy when alpha= ",0.0005," is: ",(a/303*100))
    a = 0
    b = 0
    Theta,cost,mse = logisticGradientDescent(Theta, X, Y, 0.001, numOfIterations, m)
    h = logisticSigmoidFun(X, Theta)
    for i in range(len(X)):
        if (h[i] >= 0.5):
            if (1 == Y[i]):
                a += 1
            else:
                b += 1
            #print("have heart disease")
        else:
            if (0 == Y[i]):
                a += 1
            else:
                b += 1
            #print("not have heart disease")
    print("accuracy when alpha= ",0.005," is: ",(a/303*100))
    a = 0
    b = 0
    Theta,cost,mse = logisticGradientDescent(Theta, X, Y, 0.01, numOfIterations, m)
    h = logisticSigmoidFun(X, Theta)
    for i in range(len(X)):
        if (h[i] >= 0.5):
            if (1 == Y[i]):
                a += 1
            else:
                b += 1
            #print("have heart disease")
        else:
            if (0 == Y[i]):
                a += 1
            else:
                b += 1
            #print("not have heart disease")
    print("accuracy when alpha= ",0.05," is: ",(a/303*100))
    a = 0
    b = 0
    Theta, cost,mse= logisticGradientDescent(Theta, X, Y, 0.1, numOfIterations, m)
    h = logisticSigmoidFun(X, Theta)
    for i in range(len(X)):
        if (h[i] >= 0.5):
            if (1 == Y[i]):
                a += 1
            else:
                b += 1
            #print("have heart disease")
        else:
            if (0 == Y[i]):
                a += 1
            else:
                b += 1
            #print("not have heart disease")
    print("accuracy when alpha= ",0.5," is: ",(a/303*100))
    a = 0
    b = 0
    Theta, cost,mse= logisticGradientDescent(Theta, X, Y, 1, numOfIterations, m)
    h = logisticSigmoidFun(X, Theta)
    for i in range(len(X)):
        if (h[i] >= 0.5):
            if (1 == Y[i]):
                a += 1
            else:
                b += 1
            #print("have heart disease")
        else:
            if (0 == Y[i]):
                a += 1
            else:
                b += 1

            #print("not have heart disease")
    print("accuracy when alpha= ",1," is: ",(a/303*100))
    a = 0
    b = 0

runLogisticReg()

