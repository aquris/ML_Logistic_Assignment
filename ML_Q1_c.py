# In this code I have implemented feature scaled logistic regression using batch gradient, stochaistic gradient and mini batch gradient with and without regularisation.
# Here I have also used higher powers of the data to make more features.
# Now the hypothesis looks like h(x) = g(wx) where g(wx) = 1 / (1 + e^(-wx)) and wx = w0 + w1x + w2y + w3x^2 + w4y^2 + w5xy + w6x^3 + w7y^3 + w8x^2y + w9xy^2
import numpy as np
import pandas as pd
import random
import math

def getscaleddata():
	data = pd.read_csv("marks.csv")
	Y = data['selected']
	marks_1 = data['marks1']
	marks_2 = data['marks2']

	mean_marks_1 = np.mean(marks_1)
	max_marks_1 = np.max(marks_1)
	min_marks_1 = np.min(marks_1)

	mean_marks_2 = np.mean(marks_2)
	max_marks_2 = np.max(marks_2)
	min_marks_2 = np.min(marks_2)

	marks_1sq = []
	for i in marks_1:
		marks_1sq.append(i * i)
	mean_marks_1sq = np.mean(marks_1sq)
	max_marks_1sq = np.max(marks_1sq)
	min_marks_1sq = np.min(marks_1sq)
	for i in range(len(marks_1sq)):
		marks_1sq[i] = (marks_1sq[i] - mean_marks_1sq) / (max_marks_1sq - min_marks_1sq)

	marks_2sq = []
	for i in marks_2:
		marks_2sq.append(i * i)
	mean_marks_2sq = np.mean(marks_2sq)
	max_marks_2sq = np.max(marks_2sq)
	min_marks_2sq = np.min(marks_2sq)
	for i in range(len(marks_2sq)):
		marks_2sq[i] = (marks_2sq[i] - mean_marks_2sq) / (max_marks_2sq - min_marks_2sq)

	marks_1marks_2 = []
	for i in range(len(marks_1)):
		marks_1marks_2.append(marks_1[i] * marks_2[i])
	mean_marks_1marks_2 = np.mean(marks_1marks_2)
	max_marks_1marks_2 = np.max(marks_1marks_2)
	min_marks_1marks_2 = np.min(marks_1marks_2)
	for i in range(len(marks_1marks_2)):
		marks_1marks_2[i] = (marks_1marks_2[i] - mean_marks_1marks_2) / (max_marks_1marks_2 - min_marks_1marks_2)

	marks_1_cu = []
	for i in marks_1:
		marks_1_cu.append(i * i * i)
	mean_marks_1cu = np.mean(marks_1_cu)
	max_marks_1cu = np.max(marks_1_cu)
	min_marks_1cu = np.min(marks_1_cu)
	for i in range(len(marks_1_cu)):
		marks_1_cu[i] = (marks_1_cu[i] - mean_marks_1cu) / (max_marks_1cu - min_marks_1cu)

	marks_2_cu = []
	for i in marks_2:
		marks_2_cu.append(i * i * i)
	mean_marks_2cu = np.mean(marks_2_cu)
	max_marks_2cu = np.max(marks_2_cu)
	min_marks_2cu = np.min(marks_2_cu)
	for i in range(len(marks_2_cu)):
		marks_2_cu[i] = (marks_2_cu[i] - mean_marks_2cu) / (max_marks_2cu - min_marks_2cu)

	marks_1_sq_marks_2 = []
	for i in range(len(marks_1)):
		marks_1_sq_marks_2.append(marks_1[i] * marks_1[i] * marks_2[i])
	mean_marks_1sqmarks_2 = np.mean(marks_1_sq_marks_2)
	max_marks_1sqmarks_2 = np.max(marks_1_sq_marks_2)
	min_marks_1sqmarks_2 = np.min(marks_1_sq_marks_2)
	for i in range(len(marks_1_sq_marks_2)):
		marks_1_sq_marks_2[i] = (marks_1_sq_marks_2[i] - mean_marks_1sqmarks_2) / (max_marks_1sqmarks_2 - min_marks_1sqmarks_2)

	marks_2_sq_marks_1 = []
	for i in range(len(marks_1)):
		marks_2_sq_marks_1.append(marks_1[i] * marks_2[i] * marks_2[i])
	mean_marks_2sqmarks_1 = np.mean(marks_2_sq_marks_1)
	max_marks_2sqmarks_1 = np.max(marks_2_sq_marks_1)
	min_marks_2sqmarks_1 = np.min(marks_2_sq_marks_1)
	for i in range(len(marks_2_sq_marks_1)):
		marks_2_sq_marks_1[i] = (marks_2_sq_marks_1[i] - mean_marks_2sqmarks_1) / (max_marks_2sqmarks_1 - min_marks_2sqmarks_1)

	X_train_set = []
	X_test_set = []
	Y_train_set = []
	Y_test_set = []

	for i in range(70):
		X_train_set.append([1, (marks_1[i] - mean_marks_1) / (max_marks_1 - min_marks_1), (marks_2[i] - mean_marks_2) / (max_marks_2 - min_marks_2), marks_1sq[i], marks_2sq[i], marks_1marks_2[i], marks_1_cu[i], marks_2_cu[i], marks_1_sq_marks_2[i], marks_2_sq_marks_1[i]])
		Y_train_set.append(Y[i])

	for i in range(71, 100):
		X_test_set.append([1, (marks_1[i] - mean_marks_1) / (max_marks_1 - min_marks_1), (marks_2[i] - mean_marks_2) / (max_marks_2 - min_marks_2), marks_1sq[i], marks_2sq[i], marks_1marks_2[i], marks_1_cu[i], marks_2_cu[i], marks_1_sq_marks_2[i], marks_2_sq_marks_1[i]])
		Y_test_set.append(Y[i])
	return X_train_set, X_test_set, Y_train_set, Y_test_set

def sigmoid(z):
	try:
		ans = 1.0 / (1 + math.exp(-1 * z))
	except OverflowError:
		ans = 0
	return ans

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
	Coeff = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
	lis1 = []
	for i in range(epochs):
		Temp_Coeff = Coeff.copy()
		for j in range(len(Coeff)):
			Temp_Coeff[j] = Temp_Coeff[j] - ((LearningRateNoScaling / len(X_train_set)) * (Slope(Coeff, X_train_set, Y_train_set, j)))
		Coeff = Temp_Coeff.copy()
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
	Coeff = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
	for iter in range(epochs):
		for i in range(len(Y_train_set)):
			Temp_Coeff = Coeff.copy()
			for j in range(len(Coeff)):
				Temp_Coeff[j] = Temp_Coeff[j] - (LearningRateNoScaling * (SlopeStoch(Coeff, X_train_set[i], Y_train_set[i], j)))
			Coeff = Temp_Coeff.copy()
	return Coeff

def minibtchgra(X_train_set, Y_train_set, alpha = 0.000000001, epochs = 30, batch_size = 20):
	Scaling_Learning_Rate = alpha
	Coeff = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
	No_Of_Batches = math.ceil(len(Y_train_set) / batch_size)
	equally_div = False
	if (len(Y_train_set) % batch_size == 0):
		equally_div = True;

	for epoch in range(epochs):
		for batch in range(No_Of_Batches):
			Summ = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
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

					Summ[j] += value_pridicted;

			if (not equally_div and batch == No_Of_Batches - 1):
				for j in range(len(Summ)):
					Coeff[j] -= (Summ[j] / (len(Y_train_set) % batch_size)) * Scaling_Learning_Rate
			else:
				for j in range(len(Summ)):
					Coeff[j] -= (Summ[j] / batch_size) * Scaling_Learning_Rate
	return Coeff

# Using batch gradient
def batchgrareg(X_train_set, Y_train_set, alpha = 0.00001, epochs = 50000, lambdaparameter = -49):
	LearningRateNoScaling = alpha

	Coeff = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
	lis1 = []
	for i in range(epochs):
		Temp_Coeff = Coeff.copy()
		for j in range(len(Coeff)):
			if j == 0:
				Temp_Coeff[j] = Temp_Coeff[j] - ((LearningRateNoScaling / len(X_train_set)) * (Slope(Coeff, X_train_set, Y_train_set, j)))
			else:
				Temp_Coeff[j] = (1 - alpha * lambdaparameter / len(X_train_set)) * Temp_Coeff[j] - ((LearningRateNoScaling / len(X_train_set)) * (Slope(Coeff, X_train_set, Y_train_set, j)))
		Coeff = Temp_Coeff.copy()
	return Coeff

def stochgrareg(X_train_set, Y_train_set, alpha = 0.00001, epochs = 50000, lambdaparameter = 1000):
	LearningRateNoScaling = alpha
	Coeff = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
	for iter in range(epochs):
		for i in range(len(Y_train_set)):
			Temp_Coeff = Coeff.copy()
			for j in range(len(Coeff)):
				if j == 0:
					Temp_Coeff[j] = Temp_Coeff[j] - (LearningRateNoScaling * (SlopeStoch(Coeff, X_train_set[i], Y_train_set[i], j)))
				else:
					Temp_Coeff[j] = (1 - alpha * lambdaparameter) * Temp_Coeff[j] - (LearningRateNoScaling * (SlopeStoch(Coeff, X_train_set[i], Y_train_set[i], j)))
			Coeff = Temp_Coeff.copy()
	return Coeff

def minibtchgrareg(X_train_set, Y_train_set, alpha = 0.000000001, epochs = 30, batch_size = 20, LambdaParameter = 10):
	Scaling_Learning_Rate = alpha
	Coeff = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
	No_Of_Batches = math.ceil(len(Y_train_set) / batch_size)
	equally_div = False
	if (len(Y_train_set) % batch_size == 0):
		equally_div = True;

	for epoch in range(epochs):
		for batch in range(No_Of_Batches):
			Summ = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
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
					Summ[j] += value_pridicted;

			if (not equally_div and batch == No_Of_Batches - 1):
				for j in range(len(Summ)):
					if j == 0:
						Coeff[j] = Coeff[j] - (Summ[j] / (len(Y_train_set) % batch_size)) * Scaling_Learning_Rate
					else:
						Coeff[j] = (1 - Scaling_Learning_Rate * LambdaParameter / (len(Y_test_set) % batch_size)) * Coeff[j] - (Summ[j] / (len(Y_train_set) % batch_size)) * Scaling_Learning_Rate
			else:
				for j in range(len(Summ)):
					if j == 0:
						Coeff[j] = Coeff[j] - (Summ[j] / batch_size) * Scaling_Learning_Rate
					else:
						Coeff[j] = (1 - Scaling_Learning_Rate * LambdaParameter / batch_size) * Coeff[j] - (Summ[j] / batch_size) * Scaling_Learning_Rate
	return Coeff

# First doing batch gradient, stochaistic gradient and mini batch gradient without regularisation.
X_train_set, X_test_set, Y_train_set, Y_test_set = getscaleddata()

print("Batch gradient without regularisation : ")
coeff = batchgra(X_train_set, Y_train_set, 0.00001, 1000)
print(coeff)
printaccuracy(X_test_set, Y_test_set, coeff)

print("Stochaistic gradient without regularisation : ")
coeff = stochgra(X_train_set, Y_train_set, 0.0001, 5000)
print(coeff)
printaccuracy(X_test_set, Y_test_set, coeff)

print("Mini batch gradient without regularisation : ")
coeff = minibtchgra(X_train_set, Y_train_set, 0.0001, 100, 32)
print(coeff)
printaccuracy(X_test_set, Y_test_set, coeff)

#Now doing batch gradient, stochaistic gradient and mini batch gradient with regularisation.
print("Batch gradient with regularisation : ")
coeff = batchgrareg(X_train_set, Y_train_set, 0.0001, 5000, 1000)
print(coeff)
printaccuracy(X_test_set, Y_test_set, coeff)

print("Stochaistic gradient with regularisation : ")
coeff = stochgrareg(X_train_set, Y_train_set, 0.001, 500, 1000)
print(coeff)
printaccuracy(X_test_set, Y_test_set, coeff)

print("Mini batch gradient with regularisation : ")
coeff = minibtchgrareg(X_train_set, Y_train_set, 0.0001, 1000, 32, 1000)
print(coeff)
printaccuracy(X_test_set, Y_test_set, coeff)