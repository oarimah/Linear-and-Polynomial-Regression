#Ositadinma Arimah 2b.py
#CS4442B Assignmnet 1
import matplotlib.pyplot as plt
import numpy as np

# import necessary files
xTrain = np.loadtxt('hw1xtr.dat')
yTrain = np.loadtxt('hw1ytr.dat')
xTest = np.loadtxt("hw1xte.dat")
yTest = np.loadtxt("hw1yte.dat")

tempArray = np.ones((len(xTrain), 2))  # Add a column vector of 1’s to the features
tempArray[:, 0] = np.asarray(xTrain)
x = tempArray
y = np.asarray(yTrain)

weight = np.matmul(np.matmul(np.linalg.inv(np.matmul(np.transpose(x), x)), np.transpose(x)),
                   y)  # obtain 2d weight vector

# calculate the function
h = weight[0] * xTrain + weight[1]


# Plot both the linear regression line and the training data on the same graph.
def plotData(x, h, xT, yT, data):
    if data == "Test Data":
        plt.plot(x, h, color='black')
        plt.title("2c. Linear Regression and Testing Data")
        plt.xlabel('xTestData')
        plt.ylabel('yTestData')
    else:
        plt.plot(x, h, color='red')
        plt.title("2b. Linear Regression and Training Data")
        plt.xlabel('xTrainData')
        plt.ylabel('yTrainData')
    plt.scatter(xT, yT)
    plt.show()


plotData(xTrain, h, xTrain, yTrain, "Train Data")

# Report the average error on the training set using Eq. (1).
trainingError = (1 / 40) * np.sum((h - y) ** 2)
print("The train error is: ", trainingError)
print("The weight is: ", weight)

# 2c.py

hTest = weight[0] * xTest + weight[1]  # calculate function
plotData(xTest, hTest, xTest, yTest, "Test Data")  # Plot both the regression line and the test data on the same graph.

# Report the average error on the training set using Eq. (1).
testingError = (1 / 20) * np.sum((hTest - yTest) ** 2)
print("The test error is: ", testingError)
