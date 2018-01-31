#Jena_Climate.py Jan 29th 2018
"""
Example from the book, Deep Learning with Python by Francois Challet
Data Accessed Jan 29 2018
Once again, comments in triple quotes are my side notes, hashtag comments are generally something coming from the book
"""

#PROBLEM: 
#given data going back as "lookback" timesteps, and sampled every "steps" timesteps, can you predict the temperature in "delay" timesteps?


import os
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop
import numpy as np

"""
NOTE: there is a good way to find filepathes in python here, that I am not using because my data is in the same folder as my python script

data_dir= '/users/fchollet/Downloads/jena_climate.csv'
fname = os.path.join()data_dir, "jena_climate_2009_2016.csv"

f =open(fname)
data = f.read()
f.close()

lines = data.split('\n')
head = lines[0].split(',')
lines = lines[:1]

print(header)
print(len(lines))

"""


#inspecting the data
fname = r'jena_climate_2009_2016.csv'

f =open(fname)
data = f.read()
f.close()

lines = data.split('\n')
header = lines[0].split(',')
lines = lines[1:]

print(header)
print(len(lines))

#now print all 420,551 lines into a Numpy array:
#putting the data into np arrays
float_data = np.zeros((len(lines), len(header)-1))
#store each line into the 2nd dimension of a tensor / the columns of a numpyarray
for i, line in enumerate(lines):
	values = [float(x) for x in line.split(',')[1:]]
	float_data[i, :] = values
	
temp = float_data[:,1]
#plt.plot(range(len(temp)), temp)
#the chart will be super dense since there is a weather reading every 10 mins for 7 yrs
#plt.show()
#if we were to round each month to an average, the data would be more consistent and readable,
# it would then be much easier to predict the monthly average temp pretty easily.

#must now do 2 things: preprocess data for NN, and write a python generator which takes the current array of float data
#						and yields batches of data from the past, along with the target temperature in the future.
#						Since the samples in the dataset are highly redundant (ie: sample N and sample N+1 will have most of their
#						timeteps in common), we should generate samples on the fly givena couple parameters, to save space and time



#preprocess and std normalize features

#all elementwise operatorss
mean = float_data[:2000].mean(axis=0)
float_data -= mean
std = float_data[:200000].std(axis=0)
float_data /= std
#args to generator function
#data: the original orray of float data, which is normalized
#lookback: how many timesteps bath the input data should go
#delay: how many timesteps in the future the target should be
#min_index and max_index: indices in teh data array that delimit which time steps to draw from . this is useful for keeping train vs val data separate
#shuffle: whether to shuffly the samples or draw them in chronological order
#batch_size: The number of samples per batch
#step: the period, in timesteps, at which you sample data. set it to 6 to draw data at every hour

#next use the abstract function
def generator(data, lookback, delay, min_index, max_index,
			shuffle= False,batch_size = 128, step= 6,name = "unnamed"):
	if max_index is None:
		max_index = len(data) - delay -1
	i = min_index +lookback
	while 1:
		if shuffle:
			rows = np.random.randint(min_index + lookback, max_index , size = batch_size)
		else:
			if i + batch_size >= max_index:
				i = min_index + lookback
			rows = np.arange(i, min(i+batch_size,max_index))
			i += len(rows)
		samples = np.zeros((len(rows), lookback //step, data.shape[-1]))
		targets = np.zeros((len(rows),))
		for j, row in enumerate(rows):
			indices = range(rows[j] - lookback, rows[j], step)
			samples[j] = data[indices]
			targets[j] = data[rows[j]+delay][1]
		#print(name)
		yield samples, targets


#now lets use the abstract generator function to instantiate 3 generators: training, val, testing. Each with a different temporal segment

lookback = 1440
step = 6
delay = 144
batch_size = 128


train_gen = generator(float_data, lookback = lookback, delay = delay, min_index = 0, max_index = 200000,shuffle = True,	 batch_size = batch_size,step = step, name = "train")
val_gen = generator(float_data, lookback = lookback, delay = delay, min_index = 200001, max_index = 300000, batch_size = batch_size,step = step, name = "val")
test_gen = generator(float_data, lookback = lookback, delay = delay, min_index = 300001, max_index = None,  batch_size = batch_size,step = step, name = "test")

val_steps = 300000 - 200001 - lookback
test_steps = len(float_data) - 300001 - lookback

# Take it for granted that a baseline of this, guessing that the temperature the next day will be the same as this day,
# the mean absolute error is 2.57 degrees celsius
 
#building a model: 

model = Sequential()
model.add(layers.GRU(32, input_shape=(None, float_data.shape[-1])))
model.add(layers.Dense(1))
#note the last layer does not have an activation function, because you use the mean absolute error as the loss,
# therefore, the the baseline's output will be directly comparable to the regression output of mae

model.compile(optimizer = RMSprop(), loss = 'mae')
history = model.fit_generator(train_gen,
							steps_per_epoch = 500,
							epochs=20,
							validation_data=val_gen,
							validation_steps = 250)
	
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1,len(loss)+1)

plt.figure()

plt.plot(epochs, loss, 'bo', label = "Training Loss")
plt.plot(epochs, val_loss, 'b', label = "Validation loss")
plt.title("Training and validation loss")
plt.legend()
plt.show()






