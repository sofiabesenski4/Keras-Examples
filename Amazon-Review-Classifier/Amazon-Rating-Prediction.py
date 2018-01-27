#Rating predictor, metric= mean squared error, approach = scalar regression Amazon Reviews
#Author: Thomas Besenski
#Date: Jan 24th, 2018


#General Commenting format: #'d comments indicate comments specific to the following line of code
# triple quoted comments are talking about the general ideas/processes

"""
Taken from Deep Learning with Python: Universal Workflow of Machine Learning

1) Define the problem: 
	-dataset consists of 500,000 english language reviews from food on amazon, annotated with their 1-5 star rating
	-trying to predict the rating of a review, given the text sequence
	-scalar regression model: finding a continuous value 1-5 closest to the review's true rating
	Hypothesis: given the text sequences, there will be enough information and patterns in the data to accurately predict
				future reviews.
2) Measure of Success:
	-recommend using: Mean squared error
3) Evaluation Protocol:
	-hold out validation set: because we have lots of data (500,000)
4) Preparing Data: This is what is gonna be changed/tested
	-This model will be used as a baseline, just using one-hot encoding (not using hashing trick) on all the words and evaluating
		the sentiment based purely on the presence of certain words in the review.
	-With that in mind, keep the top 10,000 most used words, and vectorize each sequence using one-hot encoding 
	-No feature engineering here, except considering only the 1000 most used words
	-considering all words contained in each sample
5) Developing a model that works better than baseline:
	-last layer activation = sigmoid to get values between 0-1
	-loss function =  mean squared error
	-optimizer = standard RMSProp algorithm
6) Scaling up and developing a model that overfits
	-going to use 2 intermediate layers with an additional sigmoid layer for output
	-16 units per layer, mimicing the IMDBReviews example
	-10 epochs just to see the progress
7) Regularizing the model and finetuning the hyper parameters
	-Try adding dropout
	-try adding l2 regularization
"""
from keras import models
from keras import layers 
import numpy as np
from keras.preprocessing.text import Tokenizer
import matplotlib.pyplot as plt
import csv
import re
#csv has the format: Id,ProductId,UserId,ProfileName,HelpfulnessNumerator,HelpfulnessDenominator,Score,Time,Summary,Text
#we are only interested in  Score and Text
#technique for parsing through the csv's while they contain commas is found from https://stackoverflow.com/questions/21527057/python-parse-csv-ignoring-comma-with-double-quotes
"""
for l in  csv.reader(lines, quotechar='"', delimiter=',',
                     quoting=csv.QUOTE_ALL, skipinitialspace=True):
"""
num_of_training_samples = 10000
num_of_testing_samples = 5000
num_of_words_in_dict = 1000
#we are going to turn the first 10,000 samples into a list containing the binary values to represent the  
# 1000 most used words from all reviews vectorized with one-hot encoding

train_labels = []
train_samples = []

fp = open("Good_Samples.csv", "r")
fp.readline()
fp_csv = csv.reader(fp)
"""

/////////////////////////////////////////////////////////////////////////////////////////////////////////
currently stuck on removing the commas from within the csv file
trying to build  regex that can do it and replace them with spaces
"""


#pattern = re.compile(r"((?:\"[^\"]*\"),|(?:[^,\"]+))")
	

for i,line in enumerate(fp_csv):
#	parsed_csv_line = [element.replace(",", " ") for element in re.findall(pattern, line)] 
	#print (line)
	#print("sample num: ",line[0])
	#print("score: ", line[6])
	#print("text: ", line[9])
	#divide by 5 to get a value 0.2-1.0
	train_labels.append(int(line[6])/5)
	train_samples.append(line[9])
#	if i== 5: break


#directly from the textbook: One-hot encoding Keras library function:
tokenizer = Tokenizer(num_words = num_of_words_in_dict)
#builds the word index
tokenizer.fit_on_texts(train_samples)

#turns strings into lists of integer indices: isn't this useless in our case???
sequences = tokenizer.texts_to_sequences(train_samples)

train_vectors = tokenizer.texts_to_matrix(train_samples, mode = 'binary')

"""
Okay so now we have the first 10,000 samples vectorized into binary one-hot encoding, considering the most used 1000 words
Time to turn it into a tensor of shape (num of samples, 1000 possible words per sample) 
"""

x_train = np.asarray(train_vectors[:8000])

y_train = np.asarray(train_labels[:8000])
x_val= np.asarray(train_vectors[8000:10000])
y_val=np.asarray(train_labels[8000:10000])
print("size x_train: {}	, size y_train: {}, size x_val: {}, size y_val: {}".format(x_train.shape,y_train.shape,x_val.shape,y_val.shape))

"""
Now to build our model
"""

model = models.Sequential()
model.add(layers.Dense(16, activation ="relu", input_shape=(1000,)))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(32, activation="relu"))
model.add(layers.Dense(1, activation = "sigmoid"))

model.compile(optimizer = "rmsprop", loss = "mse" , metrics = ['mae'])

history  = model.fit(x_train, y_train, epochs= 6, batch_size = 32, validation_data = (x_val, y_val))


"""
The following code is simply plotting the progress of training in a few ways:
"""
history_dict = history.history

loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
mean_absolute_error = history_dict['mean_absolute_error']
val_mean_absolute_error = history_dict['val_mean_absolute_error']
print(history_dict.keys())
epochs = range(1, len(history_dict['val_mean_absolute_error'])+1)

plt.plot(epochs, loss_values, 'bo', label = "Training loss")
plt.plot(epochs, val_loss_values, 'b', label = 'Validation Loss')
plt.title('Training and Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.figure()

plt.plot(epochs, mean_absolute_error, 'bo', label = "training mae")
plt.plot(epochs, val_mean_absolute_error, 'b', label = "validation mae")
plt.title("training mae vs val mae")
plt.xlabel('Epochs')
plt.ylabel('mae')
plt.legend()

plt.show()

plt.show()

model.save("amazon-rating-predicton-ffwd-dropout-twodense-6epoch.h5")
