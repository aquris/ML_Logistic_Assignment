# In this code I have implemented logistic regression using batch gradient, stochaistic gradient and mini batch gradient for feature scaled and unscaled data.

import numpy as np
import pandas as pd
import random
import math

def getdata():
	data = pd.read_csv("marks.csv")
	Y = data['selected']
	marks_1 = data['marks1']
	marks_2 = data['marks2']
	
	X_train_set = []
	X_test_set = []
	Y_train_set = []
	Y_test_set = []
	for i in range(70):
		X_train_set.append([1, marks_1[i], marks_2[i]])
		Y_train_set.append(Y[i])

	for i in range(71, 100):
		X_test_set.append([1, marks_1[i], marks_2[i]])
		Y_test_set.append(Y[i])
	return X_train_set, X_test_set, Y_train_set, Y_test_set

def getscaleddata():
	data = pd.read_csv("marks.csv")
	Y = data['selected']
	marks_1 = data['marks1']
	marks_2 = data['marks2']

	meanmarks_1 = np.mean(marks_1)
	maxmarks_1 = np.max(marks_1)
	minmarks_1 = np.min(marks_1)

	meanmarks_2 = np.mean(marks_2)
	maxmarks_2 = np.max(marks_2)
	minmarks_2 = np.min(marks_2)

	X_train_set = []
	X_test_set = []
	Y_train_set = []
	Y_test_set = []

	for i in range(70):
		X_train_set.append([1, (marks_1[i] - meanmarks_1) / (maxmarks_1 - minmarks_1), (marks_2[i] - meanmarks_2) / (maxmarks_2 - minmarks_2)])
		Y_train_set.append(Y[i])

	for i in range(70, 100):
		X_test_set.append([1, (marks_1[i] - meanmarks_1) / (maxmarks_1 - minmarks_1), (marks_2[i] - meanmarks_2) / (maxmarks_2 - minmarks_2)])
		Y_test_set.append(Y[i])
	return X_train_set, X_test_set, Y_train_set, Y_test_set

def sigmoid(z):
    return 1.0 / (1 + math.exp(-1 * z))

# Function to calculate Slope to find coefficients
def Slope(Coeff, X_train_set, Y_train_set, ind):
	diff = 0
	for i in range(len(X_train_set)):
		itr = 0
		for j in range(len(Coeff)):
			itr = itr + Coeff[j] * X_train_set[i][j]
		diff += (sigmoid(itr) - Y_train_set[i]) * X_train_set[i][ind]
	return diff

# Using batch gradient
def batchgra(X_train_set, Y_train_set, alpha = 0.00001, epochs = 50000):
	LearningRateNoScaling = alpha

	Coeff = [0, 0, 0]
	lis1 = []
	for i in range(epochs):
		TempCoeff = Coeff.copy()
		for j in range(len(Coeff)):
			TempCoeff[j] = TempCoeff[j] - ((LearningRateNoScaling / len(X_train_set)) * (Slope(Coeff, X_train_set, Y_train_set, j)))
		Coeff = TempCoeff.copy()
	return Coeff

# Finding Accuracy
def printaccuracy(X_test_set, Y_test_set, Coeff):
	count = 0
	for i in range(len(X_test_set)):
		predicted = 0
		for j in range(len(Coeff)):
		  	predicted = predicted + Coeff[j] * X_test_set[i][j]
		predicted = sigmoid(predicted)
		if predicted > 0.5:
			if Y_test_set[i] == 1:
				count += 1
		else:
			if Y_test_set[i] == 0:
				count += 1
	print("Accuracy : " + str(count / len(Y_test_set) * 100))

def SlopeStoch(Coeff, X_train_set, ActualVal, ind):
	itr = 0
	for j in range(len(Coeff)):
		itr = itr + Coeff[j] * X_train_set[j]
	return (sigmoid(itr) - ActualVal) * X_train_set[ind]

def stochgra(X_train_set, Y_train_set, alpha = 0.00001, epochs = 50000):
	LearningRateNoScaling = alpha
	Coeff = [0, 0, 0]
	for iter in range(epochs):
		for i in range(len(Y_train_set)):
			TempCoeff = Coeff.copy()
			for j in range(3):
				TempCoeff[j] = TempCoeff[j] - (LearningRateNoScaling * (SlopeStoch(Coeff, X_train_set[i], Y_train_set[i], j)))
			Coeff = TempCoeff.copy()
	return Coeff

def minibtchgra(X_train_set, Y_train_set, alpha = 0.000000001, epochs = 30, batch_size = 20):
	LearningRateScaling = alpha
	Coeff = [0, 0, 0]
	No_Of_Batches = math.ceil(len(Y_train_set) / batch_size)
	equally_div = False
	if (len(Y_train_set) % batch_size == 0):
		equally_div = True;

	for epoch in range(epochs):
		for batch in range(No_Of_Batches):
			Summation = [0, 0, 0]
			for j in range(len(Coeff)):
				for i in range(batch_size):
					if (batch * batch_size + i == len(X_train_set)):
						break
					value_pridicted = 0.0
					for wj in range(len(Coeff)):
						value_pridicted += Coeff[wj] * X_train_set[batch * batch_size + i][wj]
					value_pridicted = sigmoid(value_pridicted)
					value_pridicted -= Y_train_set[batch * batch_size + i]
					value_pridicted *= X_train_set[batch * batch_size + i][j]
					Summation[j] += value_pridicted;

			if (not equally_div and batch == No_Of_Batches - 1):
				for j in range(len(Summation)):
					Coeff[j] -= (Summation[j] / (len(Y_train_set) % batch_size)) * LearningRateScaling
			else:
				for j in range(len(Summation)):
					Coeff[j] -= (Summation[j] / batch_size) * LearningRateScaling
	return Coeff

# First doing batch gradient, stochaistic gradient and mini batch gradient without feature scaling.
X_train_set, X_test_set, Y_train_set, Y_test_set = getdata()

print("Doing batch gradient without feature scaling")
coeff = batchgra(X_train_set, Y_train_set, 0.00001, 5000)
print(coeff)
printaccuracy(X_test_set, Y_test_set, coeff)

print("Doing stochaistic gradient without feature scaling")
coeff = stochgra(X_train_set, Y_train_set, 0.001, 5000)
print(coeff)
printaccuracy(X_test_set, Y_test_set, coeff)

print("Doing Mini batch gradient without feature scaling")
coeff = minibtchgra(X_train_set, Y_train_set, 0.0001, 100, 20)
print(coeff)
printaccuracy(X_test_set, Y_test_set, coeff)

# Now doing batch gradient, stochaistic gradient and mini batch gradient with feature scaling.
X_train_set, X_test_set, Y_train_set, Y_test_set = getscaleddata()

print("Doing batch gradient with feature scaling")
coeff = batchgra(X_train_set, Y_train_set, 0.00001, 5000)
print(coeff)
printaccuracy(X_test_set, Y_test_set, coeff)

print("Doing stochaistic gradient with feature scaling")
coeff = stochgra(X_train_set, Y_train_set, 0.001, 5000)
print(coeff)
printaccuracy(X_test_set, Y_test_set, coeff)

print("Doing Mini batch gradient with feature scaling")
coeff = minibtchgra(X_train_set, Y_train_set, 0.0001, 100, 20)
print(coeff)
printaccuracy(X_test_set, Y_test_set, coeff)