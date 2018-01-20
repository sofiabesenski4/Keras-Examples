
#Mushroom Binary Classification: Edible vs Poisonous

#This version is intended to try to figure out what is going wrong with the code to make
#accuracy near 100% by fiddling around and trying to lower the accuracy by adjusting hyperparameters

#Written by Thomas Besenski
#Jan 15th 2018
from keras import models
from keras import layers
import matplotlib.pyplot as plt
import numpy as np

def interpret_sample_line(sample):
	for sample_num, sample in enumerate(fp):
		sample = sample.strip().split(",")
		for feature_num, feature_value in enumerate(sample):
				#print ("feature_num: ",feature_num, " ord(feature_val:) ", ord(feature_value))
				entire_dataset[sample_num, feature_num] = ord(feature_value)

def vectorize_features(entire_dataset):
	#trying to skip the first feature, which is the label/key for the samp
	partial_features_dataset = entire_dataset[:,1:]
	vectorized_features = np.zeros((partial_features_dataset.shape[0],partial_features_dataset.shape[1], 12))
	#print("len(entire_dataset[:] :",(entire_dataset.shape[1]))
	#print("np.unique(entire_dataset[:,i])",np.unique(entire_dataset[:,1]))
	print("the first 4 ascii values in the partial features dataset: ", str(partial_features_dataset[:4,0]))
	#outer for loop will iterate through every feature
	for i in range(partial_features_dataset.shape[1]):
			unique_feature_array = np.unique(partial_features_dataset[:,i])
			#inner for loop will iterate through every sample's feature determined by outer for loop
			for j in range(partial_features_dataset.shape[0]):
				vectorized_features[j,i, np.where(unique_feature_array == partial_features_dataset[j,i])] = 1
				
				
	return vectorized_features
	
	
def vectorize_labels(entire_dataset):
	labels = np.zeros(entire_dataset.shape[0])
	for i in range(entire_dataset.shape[0]):
		if entire_dataset[i,0] == 'e':
			labels[i]=1
			
	return labels
	

#creating a tensor to describe the dataset to hold all the values
entire_dataset = np.zeros((8123,23))

fp = open("mushrooms.csv")
#skipping the first line because those are the column titles
fp.readline()
for line in fp:
	interpret_sample_line(line)

training_data= vectorize_features(entire_dataset[:5000])
training_labels = vectorize_labels(entire_dataset[:5000])

x_train = training_data[:100,:22,:]
print("x_train.shape: ",x_train.shape)
x_test = training_data[100:,:22,:]
y_train = training_labels[:100]
print("y_train.shape: ",y_train.shape)
y_test = training_labels[100:]


holdout_val_x = training_data[100:,:22,:]
holdout_val_y = training_labels[100:]

# now to build the model
model = models.Sequential()
model.add(layers.Dense(32, activation = "relu", input_shape = (22,12)))

model.add(layers.Flatten())
model.add(layers.Dense(1, activation = "sigmoid"))

model.compile(optimizer = "rmsprop", loss = "binary_crossentropy", metrics = ['accuracy'])

history = model.fit(x_train, y_train, epochs = 1, batch_size = 4, validation_data= (holdout_val_x, holdout_val_y))
