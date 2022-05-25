#Ositadinma Arimah 2a.py
#CS4442B Assignmnet 1


# import statements

import matplotlib.pyplot as plt
import numpy as numpy

# load in the data files

xTrain = numpy.loadtxt("hw1xtr.dat")
yTrain = numpy.loadtxt("hw1ytr.dat")
xTest = numpy.loadtxt("hw1xte.dat")
yTest = numpy.loadtxt("hw1yte.dat")

#create function which plots data
def plotData(x, h, xT, yT, data):
    if data == "Test Data":
        plt.plot(x, h, color='black')
        plt.title("2a. Test Data vs. Train Data")
        plt.xlabel('xTestData')
        plt.ylabel('yTestData')
    else:
        plt.plot(x, h, color='red')
        plt.title("2a. Train Data vs. Test Data")
        plt.xlabel('xTrainData')
        plt.ylabel('yTrainData')
    plt.scatter(xT, yT)
    plt.show()


# plot Train Data
plotData(0, 0, xTrain, yTrain, "Train Data")

# plot Test Data
plotData(0, 0, xTest, yTest, "Test Data")

