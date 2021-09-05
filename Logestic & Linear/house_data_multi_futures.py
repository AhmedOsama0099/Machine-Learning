import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math


def getMeanandRange(x):
    meanAndRange = np.zeros((x.shape[0], 2))
    for i in range(x.shape[0]):
        if i == 0:
            continue
        min = 9999999
        max = -999999
        mean = 0
        for j in range(x.shape[1]):
            if x[i][j] < min:
                min = x[i][j]
            if x[i][j] > max:
                max = x[i][j]
            mean += x[i][j]
        meanAndRange[i][0] = mean / x.shape[1]
        meanAndRange[i][1] = max - min + 1
    return meanAndRange


def normalization(x, meanAndRange):
    for i in range(x.shape[0]):
        if i == 0:
            continue
        for j in range(x.shape[1]):
            x[i][j] = (x[i][j] - meanAndRange[i][0]) / meanAndRange[i][1]
    return x


futures = [
    'id', 'date', 'price', 'bedrooms', 'bathrooms',
    'sqft_living', 'sqft_lot', 'floors', 'waterfront', 'view', 'condition',
    'grade', 'sqft_above',
    'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode', 'lat', 'long',
    'sqft_living15',
    'sqft_lot15']
dataSet = pd.read_csv('house_data.csv', header=None, names=futures)

#             (index,"name","value")
dataSet.insert(0, "X0", 1)
#               rows,names
x = dataSet.loc[:, ['X0', 'grade', 'bathrooms', 'lat', 'sqft_living', 'view']]
y = dataSet.loc[:, ['price']]

# conver to matrix
x = x.to_numpy()
y = y.to_numpy()
x = np.delete(x, (0), axis=0)
y = np.delete(y, (0), axis=0)
x = x.astype('float64')

y = y.astype('float64')
x = x.T
y = y.T
meanAndRange = getMeanandRange(x)
x = normalization(x, meanAndRange)


def MSE(x, y, theta):
    h = np.dot(theta.T, x)
    cost = np.power(np.subtract(h, y), 2)
    j = (1 / (x.shape[1])) * np.sum(cost)
    return j


def costFun(x, y, theta):
    h = np.dot(theta.T, x)
    cost = np.power(np.subtract(h, y), 2)
    j = (1 / (2 * x.shape[1])) * np.sum(cost)
    return j


def gradientDescent(x, y, alpha, iteration):
    m = x.shape[1]
    theta = np.zeros((x.shape[0], 1))
    tmpTheta = np.zeros((x.shape[0], 1))
    mseArr = np.zeros(iteration)
    costArr = np.zeros(iteration)
    for i in range(iteration):
        cost = np.subtract(np.dot(theta.T, x), y)
        for j in range(theta.shape[0]):
            multi = np.multiply(cost, x[j, :])
            tmpTheta[j][0] = theta[j][0] - (alpha * (1 / m) * np.sum(multi))
        theta = tmpTheta
        mseArr[i] = MSE(x, y, theta)
        costArr[i] = costFun(x, y, theta)
    return mseArr, costArr, theta


alpha = [1, 0.1, 0.3, 0.01]
fig, axes = plt.subplots(2, 2)
fig.suptitle('MSE')

fig2, axes2 = plt.subplots(2, 2)
fig2.suptitle('Cost')

noItr = 1000
optimalTheta = np.zeros((x.shape[0], 1))
optimalMSE = 0
allThetas = []
for i in range(len(alpha)):
    mseArr, costArr, theta = gradientDescent(x, y, alpha[i], noItr)
    print("theta\n", theta)
    allThetas.append(theta)
    print("MSE\n", mseArr[len(mseArr) - 1])
    print("Cost\n", costArr[len(costArr) - 1])
    print("alpha= " + str(alpha[i]) + " --> accuracy= ", mseArr[len(mseArr) - 1])
    if i == 0:
        optimalMSE = mseArr[len(mseArr) - 1]
    if optimalMSE >= mseArr[len(mseArr) - 1]:
        optimalTheta = theta
        optimalMSE = mseArr[len(mseArr) - 1]
    axes[math.floor(i / 2), i % 2].plot(np.arange(noItr), mseArr)
    axes[math.floor(i / 2), i % 2].set_title("alpha= " + str(alpha[i]))

    axes2[math.floor(i / 2), i % 2].plot(np.arange(noItr), costArr)
    axes2[math.floor(i / 2), i % 2].set_title("alpha= " + str(alpha[i]))

print("optimal theta is \n", optimalTheta)
plt.show()
plt.show()
#
# for k in range(len(allThetas)):
#     match = 0
#     notMatch = 0
#     currTheta = allThetas[k]
#     for i in range(x.shape[1]):
#
#         f = currTheta[0, 0] + (currTheta[1, 0] * x[1][i]) + (
#                 currTheta[2, 0] * x[2][i]) + (
#                     currTheta[3, 0] * x[3][i]) + (
#                     currTheta[4, 0] * x[4][i]) + (
#                     currTheta[5, 0] * x[5][i])
#         if f == y[0][i]:
#             match = match + 1
#         else:
#             notMatch = notMatch + 1
#     print("alpha= " + str(alpha[k]) + " --> accuracy= ", (match / (match + notMatch)) * 100)
while True:
    grade = float(input("Enter grade: "))
    bathrooms = float(input("Enter bathrooms: "))
    lat = float(input("Enter lat: "))
    sqft_living = float(input("Enter sqft_living: "))
    view = float(input("Enter view: "))
    f = optimalTheta[0, 0] + (optimalTheta[1, 0] * (grade - meanAndRange[1][0]) / meanAndRange[1][1]) + (
            optimalTheta[2, 0] * (bathrooms - meanAndRange[2][0]) / meanAndRange[2][1]) + (
                optimalTheta[3, 0] * (lat - meanAndRange[3][0]) / meanAndRange[3][1]) + (
                optimalTheta[4, 0] * (sqft_living - meanAndRange[4][0]) / meanAndRange[4][1]) + (
                optimalTheta[5, 0] * (view - meanAndRange[5][0]) / meanAndRange[5][1])
    print("Price = ", f)
