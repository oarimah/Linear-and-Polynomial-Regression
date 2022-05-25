#Ositadinma Arimah 2e.py
#CS4442B Assignmnet 1
import matplotlib.pyplot as plt
import numpy as np

# import necessary files
from numpy import poly

# Implement the 2nd-order polynomial regression by adding new features x^2 x^3 to the inputs. Repeat (b)
xTrain = np.loadtxt('hw1xtr.dat')
yTrain = np.loadtxt('hw1ytr.dat')
xTest = np.loadtxt("hw1xte.dat")
yTest = np.loadtxt("hw1yte.dat")

# adding new features x^3 to the inputs
xTrainNew = np.ones((len(xTrain), 4))
xTrainNew[:, 0] = np.power(np.asarray(xTrain), 3)
xTrainNew[:, 1] = np.power(np.asarray(xTrain), 2)
xTrainNew[:, 2] = np.asarray(xTrain)
yTrainNew = np.asarray(yTrain)

# obtain model with weights and calculate function
weight = np.matmul(np.matmul(np.linalg.inv(np.matmul(np.transpose(xTrainNew), xTrainNew)), np.transpose(xTrainNew)),
                   yTrainNew)
x2 = np.power(xTrain, 2)
x3 = np.power(xTrain, 3)
h = weight[0] * x3 + weight[1] * x2 + weight[2] * xTrain + weight[3]

# calculate new values
poly = np.poly1d(np.polyfit(xTrain, h, 4))
x1 = np.linspace(xTrain.min(), xTrain.max())
y1 = poly(x1)


# Plot both the linear regression line and the training data on the same graph.
def plotData(x, h, xT, yT, data):
    if data == "Test Data":
        plt.plot(x, h, color='black')
        plt.title("2e. 3rd-Order Polynomial Regression and Testing Data")
        plt.xlabel('xTestData')
        plt.ylabel('yTestData')
    else:
        plt.plot(x, h, color='red')
        plt.title("2e. 3rd-Order Polynomial Regression and Training Data")
        plt.xlabel('xTrainData')
        plt.ylabel('yTrainData')
    plt.scatter(xT, yT)
    plt.show()


plotData(x1, y1, xTrain, yTrain, "Train Data")
print("The weight is: ", weight)

# Finally, we are calculating our training error using the given formula
trainingError = (1 / 40) * np.sum((h - yTrain) ** 2)
print("The train error is: ", trainingError)

# Implement the 2nd-order polynomial regression by adding new features x^2 x^3 to the inputs. Repeat (c)
x2Test = np.power(xTest, 2)
x3Test = np.power(xTest, 3)
hTest = weight[0] * x3Test + weight[1] * x2Test + weight[2] * xTest + weight[3]

# calculate new values
polyTest = np.poly1d(np.polyfit(xTest, hTest, 4))
x1Test = np.linspace(xTest.min(), xTest.max())
y1Test = polyTest(x1Test)

#plot testing data
plotData(x1, y1, xTest, yTest, "Test Data")

# Finally, we are calculating our test error using the given formula
testingError = (1 / 20) * np.sum((hTest - yTest) ** 2)
print("The test error is: ", testingError)