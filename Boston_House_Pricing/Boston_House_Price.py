#Boston house pricing example, Deep Learning with Keras
#Jan 10th 2018
import matplotlib.pyplot as plt
from keras.datasets import boston_housing
from keras import models
from keras import layers
import numpy as np
##importing both the training sets and the testing sets which are already divided by the keras dataset
(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()

##Since the features all have a different range of values they can take on, it is 
##crucial that we "normalize" the features based on the other samples in the training set

##the axis parameter determines which axis from the multidimensional array/tensor we are finding the average of
mean = train_data.mean(axis=0)
print(mean)
train_data -=mean
print("train_data - mean",train_data)
std = train_data.std(axis=0)
train_data /= std
print("normalized training data",train_data)
test_data -=mean
test_data /=std

def build_model():
	model = models.Sequential()
	##the input shape below is representing the 2nd axis "shape[1]" since the first axis is simply references to the different datapoints
	model.add(layers.Dense(64, activation = "relu", input_shape=(train_data.shape[1],)))
	model.add(layers.Dense(64, activation = "relu"))
	model.add(layers.Dense(1))
	model.compile(optimizer = "rmsprop", loss= "mse", metrics=['mae'])
	return model


##now setting up K-FOLD VALIDATION for training, since there is not a large number of samples
## Note that K =4 in this example

k =4
num_val_samples = len(train_data)//k
num_epochs = 100
all_mae_histories = []
val_mae_histories = []
for i in range(k):
	print('processing fold #', i)
	val_data = train_data[i*num_val_samples: (i+1)*num_val_samples]
	val_targets = train_targets[i*num_val_samples: (i+1)*num_val_samples]
	
	partial_train_data = np.concatenate([train_data[:i*num_val_samples] 
										, train_data[(i+1)*num_val_samples:]],
										axis = 0)
	partial_train_targets = np.concatenate([train_targets[:i*num_val_samples] 
										, train_targets[(i+1)*num_val_samples:]]
										, axis = 0)
	model = build_model()
	history = model.fit(partial_train_data, partial_train_targets, validation_data = (val_data,val_targets), epochs = num_epochs, batch_size = 1, verbose = 2)
	val_mae_history = history.history['val_mean_absolute_error']
	val_mae_histories.append(val_mae_history)
	all_mae_history = history.history['mean_absolute_error']
	all_mae_histories.append(all_mae_history)
average_val_mae_history = [np.mean([x[i] for x in val_mae_histories]) for i in range(num_epochs)]
average_all_mae_history = [np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)]
plt.plot(range(1,len(average_all_mae_history)+1), average_all_mae_history, "bo", label="average_all_mae")
plt.plot(range(1,len(average_val_mae_history)+1), average_val_mae_history, "b", label = "average_val_mae")
plt.xlabel('Epochs')
plt.ylabel('MAE')
plt.show()

	
	
	
