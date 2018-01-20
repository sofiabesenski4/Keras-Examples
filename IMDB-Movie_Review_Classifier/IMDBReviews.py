#Classifying movie Reviews, p. 69 in Deep learning with Python
from keras.datasets import imdb
import numpy as np
from keras import models
from keras import layers
import matplotlib.pyplot as plt


#loading the data into 2 tuples containing lists
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
#print(train_data[0], train_labels[0])

#This example is simply turning the words in the review into a 10,000D vector which contains binary
#values which represents if a sample review contains a given word or not. Somewhat naive set up here
#because it does not account for sentences with opposite meanings, but identical word composition.

##   following code will decode a review, 1st one in this case, from it's dictionary key representation to english
##   new variable representing the imdb.get_word_index() function
#word_index = imdb.get_word_index()
#reverse_word_index = dict([value,key] for (key,value) in word_index.items())
#print (reverse_word_index)
#decoded_review = ' '.join([reverse_word_index.get(i-3, '?') for i in train_data[0]])
#print (decoded_review)

##Preparing data for feeding:

##Vectorize sequences will turn the sequences of integer digits, corresponding to words in the dictionary stored in imdb
def vectorize_sequences(sequences,dimension=10000):
	results = np.zeros((len(sequences),dimension))
	for i, sequence in enumerate(sequences):
		results[i,sequence]=1.
#	print (results)
	return results
x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

##Vectorizing the labels for automated hypothesis testing
xy_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')


##Defining the model to be used, setting up the network

model = models.Sequential()
#first layer must specify the shape of the data coming in, (blank used to denote variable number of reviews coming in)
model.add(layers.Dense(16, activation="relu", input_shape = (10000,)))
model.add(layers.Dense(16, activation = 'relu'))
model.add(layers.Dense(1, activation = "sigmoid"))

##now we must define what Loss function and optimizer to use when adjusting the weight matrix at every update step
## also defining what we are trying to maximize in the experimental learning process
model.compile(optimizer = "rmsprop", loss = "binary_crossentropy", metrics = ['binary_accuracy'])
##the string "rmsprop" represents a default optimizer stored in the Keras library, but you can also explicitly define your own
##optimizers or loss functions using methods within the keras stdlib and pass them in as function objects

##must set aside some samples for testing how well our training is done, by testing out the network on an unencountered set of 10,000 samples

x_val  = x_train[:10000]
partial_x_train = x_train[10000:]
y_val = y_train[:10000]
partial_y_train= y_train[10000:]


##Now we are set to start "training" our model on the training and test cases...

history = model.fit(partial_x_train, partial_y_train, epochs=20, batch_size=512, validation_data = (x_val,y_val))




###The following code is simply plotting the progress of training in a few ways:

history_dict = history.history

loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']

epochs = range(1, len(history_dict['acc'])+1)

plt.plot(epochs, loss_values, 'bo', label = "Training loss")
plt.plot(epochs, val_loss_values, 'b', label = 'Validation Loss')
plt.title('Training and Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
 
