#!/usr/bin/python
#All the imports including numpy for processing array, pandas is for reading the csv file, and that imports classication
import numpy as np  
import pandas as pd
from sklearn.neural_network import MLPClassifier 

#Setting target vector: from 1-52 (1-9, a-z, A-Z)
targetVector = []
i = 1
j = 1
while i <= 62:
	while j <= 55:
		targetVector.append(i)
		j+=1
	i+=1
	j=1

#Since the number of the actual data vector is one smaller than the assumed target vector, reduce the last element
targetVector.pop(-1)

#Making the target vector as an numpy array
targetVector = np.array(targetVector)

#read the training set 
dataVector = pd.read_csv("allData.csv")

#import the neural network classifier 
clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(100,1000), random_state = 4)

#Print the shape of the dataVector to confirm the shape the vector have the same number of rows 
print(dataVector.shape)
print(targetVector.shape)

#load trainning data sets to two vectors X and y
#X,y = digits.data[:-10], digits.target[:-10]

#Apply the neural network to the data set 
clf.fit(dataVector, targetVector)

#Find the result of the training set 
result = clf.predict(dataVector)
#
##Find the length of the result, and compare how many of the result is correct by compare it to the test_data set
length = len(result)
#
##Setting the counter for correct result
counter = 0
i = 0
while i < length:
	if result[i] == targetVector[i]:
		counter+=1
	i+=1
#Print the accuracy of the model
print(str((counter+0.0)/length*100)+"%")

#Print the length of the result
print(str(len(targetVector))+" sets of test data")
