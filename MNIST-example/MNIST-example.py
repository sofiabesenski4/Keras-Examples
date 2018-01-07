from keras.datasets import mnist
from keras import models
from keras import layers
from keras.utils import to_categorical


#grabbing the test/training data
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

#Defining a new model
network = models.Sequential()

#Adding hidden layers to the network
#First layer

network.add(layers.Dense(512, activation = "relu", input_shape = (28*28,)))
#network.add(layers.Dense(512, activation = "relu", input_shape = (28*28,)))
#Last layer which will end up classifying which number the image represents
network.add(layers.Dense(10, activation = "softmax"))

#Compilation step, choosing the loss function, optimizer, and the way that we will monitor correctness
network.compile(optimizer = "rmsprop", loss = "categorical_crossentropy", metrics = ['accuracy']) 

#Now to prepare the image data:

#reshaping the data to fit 60000 samples into image sizes of 28*28
train_images = train_images.reshape((60000, 28*28))
#reforming the value stored in each pixel by normalizing it between 0-255 to define how dark the grey is
train_images= train_images.astype('float32')/255

#same goes for 
test_images = test_images.reshape((10000, 28*28))
test_images = test_images.astype('float32')/255


#preparing the labels and categorizing them
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

#now to run the training(fit) function

network.fit(train_images, train_labels, epochs=5, batch_size=128)


#now to test how well our model learned!

test_loss, test_acc = network.evaluate(test_images, test_labels)
print('test_accuracy:', test_acc)
