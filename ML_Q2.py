# In this code I have predicted heart diseases using the clevland medical data. I have used mini batch GDA on regularised and feature scaled data.
import numpy as np
import pandas as pd
import random
import math

def getscaleddata():
	data = pd.read_csv("heart_diseases.csv")
	n = data.shape[0]
	Y = data['y']
	f1 = data['f1']
	f1 = (f1 - np.mean(f1)) / (np.max(f1) - np.min(f1))
	f2 = data['f2']
	f2 = (f2 - np.mean(f2)) / (np.max(f2) - np.min(f2))
	f3 = data['f3']
	f3 = (f3 - np.mean(f3)) / (np.max(f3) - np.min(f3))
	f4 = data['f4']
	f4 = (f4 - np.mean(f4)) / (np.max(f4) - np.min(f4))
	f5 = data['f5']
	f5 = (f5 - np.mean(f5)) / (np.max(f5) - np.min(f5))
	f6 = data['f6']
	f6 = (f6 - np.mean(f6)) / (np.max(f6) - np.min(f6))
	f7 = data['f7']
	f7 = (f7 - np.mean(f7)) / (np.max(f7) - np.min(f7))
	f8 = data['f8']
	f8 = (f8 - np.mean(f8)) / (np.max(f8) - np.min(f8))
	f9 = data['f9']
	f9 = (f9 - np.mean(f9)) / (np.max(f9) - np.min(f9))
	f10 = data['f10']
	f10 = (f10 - np.mean(f10)) / (np.max(f10) - np.min(f10))
	f11 = data['f11']
	f11 = (f11 - np.mean(f11)) / (np.max(f11) - np.min(f11))
	f12 = data['f12']
	f12 = (f12 - np.mean(f12)) / (np.max(f12) - np.min(f12))
	f13 = data['f13']
	f13 = (f13 - np.mean(f13)) / (np.max(f13) - np.min(f13))
	X_train_set = []
	X_test_set = []
	Y_train_set = []
	Y_test_set = []
	for i in range(int(0.7 * n)):
		X_train_set.append([1, f1[i], f2[i], f3[i], f4[i], f5[i], f6[i], f7[i], f8[i], f9[i], f10[i], f11[i], f12[i], f13[i]])
		Y_train_set.append(Y[i])

	for i in range(int(0.7 * n), n):
		X_test_set.append([1, f1[i], f2[i], f3[i], f4[i], f5[i], f6[i], f7[i], f8[i], f9[i], f10[i], f11[i], f12[i], f13[i]])
		Y_test_set.append(Y[i])
	return X_train_set, X_test_set, Y_train_set, Y_test_set

def sigmoid(z):
    return 1.0 / (1 + math.exp(-1 * z))

def minibtchgrareg(X_train_set, Y_train_set, alpha = 0.000000001, epochs = 30, batch_size = 20, Lambda = 10):
	Scaling_Learning_Rate = alpha
	Coeff = [0] * len(X_train_set[0])
	No_of_Batches = math.ceil(len(Y_train_set) / batch_size)
	equally_div = False
	if (len(Y_train_set) % batch_size == 0):
		equally_div = True;

	for epoch in range(epochs):
		for batch in range(No_of_Batches):
			summ = [0] * len(X_train_set[0])
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
					summ[j] += value_pridicted;

			if (not equally_div and batch == No_of_Batches - 1):
				for j in range(len(summ)):
					if j == 0:
						Coeff[j] = Coeff[j] - (summ[j] / (len(Y_train_set) % batch_size)) * Scaling_Learning_Rate
					else:
						Coeff[j] = (1 - Scaling_Learning_Rate * Lambda / (len(Y_test_set) % batch_size)) * Coeff[j] - (summ[j] / (len(Y_train_set) % batch_size)) * Scaling_Learning_Rate
			else:
				for j in range(len(summ)):
					if j == 0:
						Coeff[j] = Coeff[j] - (summ[j] / batch_size) * Scaling_Learning_Rate
					else:
						Coeff[j] = (1 - Scaling_Learning_Rate * Lambda / batch_size) * Coeff[j] - (summ[j] / batch_size) * Scaling_Learning_Rate
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
	print("Accuracy is : " + str(count / len(Y_test_set) * 100))

def getconfusionmat(X_test_set, Y_test_set, Coeff):
	true_positives = 0
	false_positives = 0
	true_negatives = 0
	false_negatives = 0
	for i in range(len(X_test_set)):
		predicted = 0
		for j in range(len(Coeff)):
		  	predicted = predicted + Coeff[j] * X_test_set[i][j]
		predicted = sigmoid(predicted)
		if predicted > 0.5:
			if Y_test_set[i] == 1:
				true_positives += 1
			else:
				false_positives += 1
		else:
			if Y_test_set[i] == 0:
				true_negatives += 1
			else:
				false_negatives += 1
	predicted_positive = []
	predicted_negative = []
	confustion_matrix = []
	predicted_positive.append(true_positives)
	predicted_positive.append(false_positives)
	predicted_negative.append(false_negatives)
	predicted_negative.append(true_negatives)
	confustion_matrix.append(predicted_positive)
	confustion_matrix.append(predicted_negative)
	return confustion_matrix

X_train_set, X_test_set, Y_train_set, Y_test_set = getscaleddata()
print("Mini batch gradient with feature scaling and regularised data : ")
coeff = minibtchgrareg(X_train_set, Y_train_set, 0.00001, 500, 64, 10)
print("Final coefficients : ")
print(coeff)
printaccuracy(X_test_set, Y_test_set, coeff)
print("Confustion matrix : ")
print(getconfusionmat(X_test_set, Y_test_set, coeff))