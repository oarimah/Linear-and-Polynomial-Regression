#Ositadinma Arimah 2d.py
#CS4442B Assignmnet 1
import matplotlib.pyplot as plt
import numpy as np

# import necessary files
from numpy import poly

# Implement the 2nd-order polynomial regression by adding new features x^2 to the inputs. Repeat (b)
xTrain = np.loadtxt('hw1xtr.dat')
yTrain = np.loadtxt('hw1ytr.dat')
xTest = np.loadtxt("hw1xte.dat")
yTest = np.loadtxt("hw1yte.dat")

# adding new features x^2 to the inputs
xTrainNew = np.ones((len(xTrain), 3))
xTrainNew[:, 0] = np.power(np.asarray(xTrain), 2)
xTrainNew[:, 1] = np.asarray(xTrain)
yTrainNew = np.asarray(yTrain)

# calculate weight
weight = np.matmul(np.matmul(np.linalg.inv(np.matmul(np.transpose(xTrainNew), xTrainNew)), np.transpose(xTrainNew)),
                   yTrainNew)

# obtain model with weights and calculate function
x2 = np.power(xTrain, 2)
h = weight[0] * x2 + weight[1] * xTrain + weight[2]

# calculate new values
x1 = np.linspace(xTrain.min(), xTrain.max())
poly = np.poly1d(np.polyfit(xTrain, h, 3))
y1 = poly(x1)


# Plot both the linear regression line and the training data on the same graph.
def plotData(x, h, xT, yT, data):
    if data == "Test Data":
        plt.plot(x, h, color='black')
        plt.title("2d. 2nd-Order Polynomial Regression and Testing Data")
        plt.xlabel('xTestData')
        plt.ylabel('yTestData')
    else:
        plt.plot(x, h, color='red')
        plt.title("2d. 2nd-Order Polynomial Regression and Training Data")
        plt.xlabel('xTrainData')
        plt.ylabel('yTrainData')
    plt.scatter(xT, yT)
    plt.show()


plotData(x1, y1, xTrain, yTrain, "Train Data")
print("The weight is: ", weight)

# calculate training error
trainingError = (1 / 40) * np.sum((h - yTrainNew) ** 2)
print("The train error is: ", trainingError)

# Implement the 2nd-order polynomial regression by adding new features x^2 to the inputs. Repeat (c)
x2_test = np.power(xTest, 2)

# obtain function for test error
hTest = weight[0] * x2_test + weight[1] * xTest + weight[2]

# calculate new values
polyTest = np.poly1d(np.polyfit(xTest, hTest, 3))
x1Test = np.linspace(xTest.min(), xTest.max())
y1Test = polyTest(x1Test)

# plotting linear regression line and test data
plotData(x1Test, y1Test, xTest, yTest, "Test Data")

# Finally, we are calculating our test error using the given formula
testingError = (1 / 20) * np.sum((hTest - yTest) ** 2)
print("The test error is: ", testingError)
