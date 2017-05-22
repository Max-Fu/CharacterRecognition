#!/usr/bin/python
#All the imports including numpy for processing array, scaling array from 0-255 to decimals, and packet that will convert image to numbers
import numpy as np 
from sklearn.preprocessing import StandardScaler 
from PIL import Image

#Function for decimalize the source (or the image)
def decimalize(source):
	#opening the image with normal RGB numbers 	
	f = Image.open(source).resize((5,5),Image.ANTIALIAS).convert('RGBA')
	#Conver this array into a numpy array 
	arr = np.array(f)
	#Conver the previous 2D array to 1D array 	
	flat_arr = arr.ravel()
	#Setting the scaling statistics by using the Standard Scaler for data 	
	scaler = StandardScaler()
	#Making the data into a 1D array	
	flat_arr=scaler.fit_transform(flat_arr)
	return flat_arr

#Put the data into the datavectors as a whole 
dataVector = []

file_object = open('all.txt', 'r')
for i in file_object:
	i = i.strip("\n")
	dataVector.append(decimalize(i))

#After this: the dataVector will be filled with list of 1D decimals of the images from input
dataVector = np.asarray(dataVector)
#Save the vector into a csv file so that it can be processed
np.savetxt("allData.csv", dataVector, delimiter=",")