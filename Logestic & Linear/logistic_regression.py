import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

from six import print_


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


futures = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak',
           'slope', 'ca', 'thal', 'target'
           ]
dataSet = pd.read_csv('heart.csv', header=None, names=futures)

#             (index,"name","value")
dataSet.insert(0, "X0", 1)
#               rows,names
x = dataSet.loc[:, ['X0', 'trestbps', 'chol', 'thalach', 'oldpeak']]
y = dataSet.loc[:, ['target']]

# conver to matrix
x = x.to_numpy()
y = y.to_numpy()
x = np.delete(x, 0, axis=0)
y = np.delete(y, 0, axis=0)
# print(y)
x = x.astype('float64')

y = y.astype('float64')
x = x.T
y = y.T
meanAndRange = getMeanandRange(x)
x = normalization(x, meanAndRange)

print_(x)


def costFun(x, y, theta):
    z = np.dot(theta.T, x)
    h = 1 / (1 + np.exp(-1 * z))
    log1 = np.log10(h)
    temp = np.multiply(y, log1)
    log2 = np.log10(1 - h)
    temp2 = np.multiply(1 - y, log2)
    J = (np.sum(temp + temp2) * -1) / x.shape[1]
    return J


def MSE(x, y, theta):
    z = np.dot(theta.T, x)
    h = 1 / (1 + np.exp(-1 * z))
    cost = np.power(np.subtract(h, y), 2)
    j = (1 / (x.shape[1])) * np.sum(cost)
    return j


def gradientDescent(x, y, alpha, iteration):
    m = x.shape[1]
    theta = np.zeros((x.shape[0], 1))
    tmpTheta = np.zeros((x.shape[0], 1))
    costArr = np.zeros(iteration)
    mseArr = np.zeros(iteration)
    for i in range(iteration):
        z = np.dot(theta.T, x)
        h = 1 / (1 + (np.exp(-1 * z)))
        cost = np.subtract(h, y)
        for j in range(theta.shape[0]):
            multi = np.multiply(cost, x[j, :])
            tmpTheta[j][0] = theta[j][0] - (alpha * (1 / m) * np.sum(multi))
        theta = tmpTheta
        costArr[i] = costFun(x, y, theta)
        mseArr[i] = MSE(x, y, theta)
    return costArr, mseArr, theta


alpha = [1, 0.1, 0.3, 0.05]

fig, axes = plt.subplots(2, 2)
fig.suptitle('MSE')

fig2, axes2 = plt.subplots(2, 2)
fig2.suptitle('Cost')

noItr = 1000
optimalTheta = np.zeros((x.shape[0], 1))
optimalMSE = 0
allThetas = []
for i in range(len(alpha)):

    costArr, mseArr, theta = gradientDescent(x, y, alpha[i], noItr)
    print("theta\n", theta)
    allThetas.append(theta)
    print("MSE\n", mseArr[len(mseArr) - 1])
    print("Cost\n", costArr[len(costArr) - 1])

    if i == 0:
        optimalMSE = mseArr[len(mseArr) - 1]
    if optimalMSE >= mseArr[len(mseArr) - 1]:
        optimalTheta = theta
        optimalMSE = mseArr[len(mseArr) - 1]
    axes[math.floor(i / 2), i % 2].plot(np.arange(noItr), mseArr)
    axes[math.floor(i / 2), i % 2].set_title("alpha= " + str(alpha[i]))

    axes2[math.floor(i / 2), i % 2].plot(np.arange(noItr), costArr)
    axes2[math.floor(i / 2), i % 2].set_title("alpha= " + str(alpha[i]))
plt.show()
plt.show()
print("Optimal Theta is \n", optimalTheta)

for k in range(len(allThetas)):
    match = 0
    notMatch = 0
    currTheta = allThetas[k]
    for i in range(x.shape[1]):
        z = currTheta[0, 0] + (currTheta[1, 0] * x[1][i]) + (
                currTheta[2, 0] * x[2][i]) + (
                    currTheta[3, 0] * x[3][i]) + (
                    currTheta[4, 0] * x[4][i])
        h = 1 / (1 + np.exp(-1 * z))
        if h >= 0.5:
            if y[0][i] == 1:
                match = match + 1
            else:
                notMatch = notMatch + 1
        else:
            if y[0][i] == 0:
                match = match + 1
            else:
                notMatch = notMatch + 1
    print("alpha= " + str(alpha[k]) + " --> accuracy= ", (match / (match + notMatch)) * 100)
while True:
    trestbps = float(input("Enter trestbps: "))
    chol = float(input("Enter chol: "))
    thalach = float(input("Enter thalach: "))
    oldpeak = float(input("Enter oldpeak: "))

    z = optimalTheta[0, 0] + (optimalTheta[1, 0] * (trestbps - meanAndRange[1][0]) / meanAndRange[1][1]) + (
            optimalTheta[2, 0] * (chol - meanAndRange[2][0]) / meanAndRange[2][1]) + (
                optimalTheta[3, 0] * (thalach - meanAndRange[3][0]) / meanAndRange[3][1]) + (
                optimalTheta[4, 0] * (oldpeak - meanAndRange[4][0]) / meanAndRange[4][1])
    h = 1 / (1 + np.exp(-1 * z))
    print("h= ", h)
    if h >= 0.5:
        print("target=1 (have heart disease)")
    else:
        print("target=0 (don't have heart disease)")
