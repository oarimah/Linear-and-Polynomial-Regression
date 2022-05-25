#Ositadinma Arimah 3a.py
#CS4442B Assignmnet 1

import numpy as np
from matplotlib import pyplot as plt
xTrain = np.loadtxt('hw1xtr.dat')
yTrain = np.loadtxt('hw1ytr.dat')
xTest = np.loadtxt("hw1xte.dat")
yTest = np.loadtxt("hw1yte.dat")

class Regularize:

    def __init__(self, lam):
        self.coeffW = None
        self._lambda = lam

    @staticmethod
    def _x_bar(x):
        return np.hstack((np.power(np.asarray(x), 4), np.power(np.asarray(x), 3), np.power(np.asarray(x), 2), x, [1.0]))

    def fit(self, xTrain, yTrain):
        X = np.vstack(([self._x_bar(x) for x in xTrain]))
        Y = np.vstack(([y for y in yTrain]))
        XT = np.transpose(X)
        XTX = np.matmul(XT, X) + self._lambda * np.identity(X.shape[1]) ## w = inv(xTx + lambda * I) * xTy
        self.coeffW = np.matmul(np.matmul(np.linalg.inv(XTX), XT), Y)

# create regularization objects, vary the regularization
# parameter λ ∈ {0.01, 0.1, 1, 10, 100, 1000}
regObj1 = Regularize(0.01)
regObj2 = Regularize(0.1)
regObj3 = Regularize(1)
regObj4 = Regularize(10)
regObj5 = Regularize(100)
regObj6 = Regularize(1000)
regObj7 = Regularize(10000)

# adjust the train data to fit
regObj1.fit(xTrain, yTrain)
regObj2.fit(xTrain, yTrain)
regObj3.fit(xTrain, yTrain)
regObj4.fit(xTrain, yTrain)
regObj5.fit(xTrain, yTrain)
regObj6.fit(xTrain, yTrain)
regObj7.fit(xTrain, yTrain)


x2 = np.power(xTrain, 2)
x3 = np.power(xTrain, 3)
x4 = np.power(xTrain, 4)
x2Test = np.power(xTest, 2)
x3Test = np.power(xTest, 3)
x4Test = np.power(xTest, 4)
lambdas = [0.01, 0.1, 1, 10, 100, 1000, 10000]

# Calculate fourth order polonomial regression
wList = []

l1 = regObj1.coeffW[0] * x4 + regObj1.coeffW[1] * x3 + regObj1.coeffW[2] * x2 + \
     regObj1.coeffW[
         3] * xTrain + [1.7552]
wList.append(l1)
l2 = regObj2.coeffW[0] * x4 + regObj2.coeffW[1] * x3 + regObj2.coeffW[2] * x2 + \
     regObj2.coeffW[
         3] * xTrain + [1.7552]
wList.append(l2)
l3 = regObj3.coeffW[0] * x4 + regObj3.coeffW[1] * x3 + regObj3.coeffW[2] * x2 + \
     regObj3.coeffW[
         3] * xTrain + [1.7552]
wList.append(l3)
l4 = regObj4.coeffW[0] ** 4 + regObj4.coeffW[1] * x3 + regObj4.coeffW[2] * x2 + \
     regObj4.coeffW[3] * xTrain + [1.7552]
wList.append(l4)
l5 = regObj5.coeffW[0] ** 4 + regObj5.coeffW[1] * x3 + regObj5.coeffW[2] * x2 + \
     regObj5.coeffW[
         3] * xTrain + [1.75552]
wList.append(l5)
l6 = regObj6.coeffW[0] * x4 + regObj6.coeffW[1] * x3 + regObj6.coeffW[2] * x2 + \
     regObj6.coeffW[
         3] * xTrain + [1.75552]
wList.append(l6)
l7 = regObj7.coeffW[0] * x4 + regObj7.coeffW[1] * x3 + regObj7.coeffW[2] * x2 + \
     regObj7.coeffW[
         3] * xTrain + [1.75552]
wList.append(l7)
trainingError = []
for i in range(len(wList)):
    trainingErrorIndex = (1 / 40) * np.sum((wList[i] - yTrain) ** 2)
    trainingError.append(trainingErrorIndex)

for i in range(len(trainingError)):
    print(print("The train error is: ", trainingError[i]))


plot1 = plt.figure(1)
plt.xlabel("Lambda")
plt.ylabel("yTrain")
plt.plot(lambdas, trainingError)
plt.xscale("log")
plt.ylabel('yTrain')
plt.title("3a.  xTrain vs. yTrain for error")
plt.show()


# Test Data
wListTest = []

l1Test = regObj1.coeffW[0] * x4Test + regObj1.coeffW[1] * x3Test + regObj1.coeffW[2] * x2Test + \
         regObj1.coeffW[3] * x4Test + [1.7552]

l2Test = regObj2.coeffW[0] * x4Test + regObj2.coeffW[1] * x3Test + regObj2.coeffW[2] * x2Test + \
         regObj2.coeffW[3] * xTest + [1.7552]

l3Test = regObj3.coeffW[0] * x4Test + regObj3.coeffW[1] * xTest + regObj3.coeffW[2] * x2Test + \
         regObj3.coeffW[3] * xTest + [1.7552]

l4Test = regObj4.coeffW[0] * x4Test + regObj4.coeffW[1] * x3Test + regObj4.coeffW[2] * x2Test + \
         regObj4.coeffW[3] * x4Test + [1.7552]

l5Test = regObj5.coeffW[0] * x4Test + regObj5.coeffW[1] * x3Test + regObj5.coeffW[2] * x2Test + \
         regObj5.coeffW[3] * xTest + [1.7552]

l6Test = regObj6.coeffW[0] * x4Test + regObj6.coeffW[1] * x3Test + regObj6.coeffW[2] * x2Test + \
         regObj6.coeffW[3] * xTest + [1.7552]

l7Test = regObj7.coeffW[0] * xTest + regObj7.coeffW[1] * x3Test + regObj7.coeffW[2] * x2Test + \
         regObj7.coeffW[3] * xTest + [1.7552]


testingError1 = (1 / 20) * np.sum((l1Test - yTest) ** 2)
testingError2 = (1 / 20) * np.sum((l2Test - yTest) ** 2)
testingError3 = (1 / 20) * np.sum((l3Test - yTest) ** 2)
testingError4 = (1 / 20) * np.sum((l4Test - yTest) ** 2)
testingError5 = (1 / 20) * np.sum((l5Test - yTest) ** 2)
testingError6 = (1 / 20) * np.sum((l6Test - yTest) ** 2)
testingError7 = (1 / 20) * np.sum((l7Test - yTest) ** 2)

testingError = [testingError1,testingError2,testingError3,testingError4,testingError5,testingError6,testingError7]
for i in range(len(testingError)):
    print(print("The testing error is: ", testingError[i]))


plt.scatter(xTest, yTest)
plt.xlabel('xTest')
plt.ylabel('yTest')
plt.title("3a. xTest vs. yTest for error")
plt.plot(lambdas, testingError)
plt.xscale("log")

plt.show()


#3b)

w4 = [regObj1.coeffW[0], regObj2.coeffW[0], regObj3.coeffW[0], regObj4.coeffW[0], regObj5.coeffW[0], regObj6.coeffW[0], regObj7.coeffW[0]]
w3 = [regObj1.coeffW[1], regObj2.coeffW[1], regObj3.coeffW[1], regObj4.coeffW[1], regObj5.coeffW[1], regObj6.coeffW[1], regObj7.coeffW[1]]
w2 = [regObj1.coeffW[2], regObj2.coeffW[2], regObj3.coeffW[2], regObj4.coeffW[2], regObj5.coeffW[2], regObj6.coeffW[2], regObj7.coeffW[2]]
w1 = [regObj1.coeffW[3], regObj2.coeffW[3], regObj3.coeffW[3], regObj4.coeffW[3], regObj5.coeffW[3], regObj6.coeffW[3], regObj7.coeffW[3]]
w0 = [1.7552, 1.7552, 1.7552, 1.7552, 1.7552, 1.7552, 1.7552]

plot3 = plt.figure(3)
plt.title("Weight Parameter as a Function of Lambda")
plt.xlabel("Lambda")
plt.ylabel("Weights")
plt.xscale("log")


plt.plot(lambdas, w0)
plt.plot(lambdas, w1)
plt.plot(lambdas, w2)
plt.plot(lambdas, w3)
plt.plot(lambdas, w4)

plt.show()

#3c)
