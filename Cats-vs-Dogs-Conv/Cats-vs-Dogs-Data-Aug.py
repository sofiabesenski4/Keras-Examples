"""
Cats-vs-Dogs.py , Convolutional Neural Network

"""
import os, shutil
from keras import models
from keras import layers
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

original_dataset_dir = '/home/teb8/School/Research/Keras-Examples/Cats-vs-Dogs-Conv/train'
base_dir = '/home/teb8/School/Research/Keras-Examples/Cats-vs-Dogs-Conv/Training-Subset'

#defining the names of where to get the image samples from
train_dir = os.path.join(base_dir, "train")
validation_dir = os.path.join(base_dir, "validation")
test_dir = os.path.join(base_dir, "test")

train_cats_dir = os.path.join(train_dir,'cats')
train_dogs_dir = os.path.join(train_dir,'dogs')

validation_cats_dir = os.path.join(validation_dir, "cats")
validation_dogs_dir = os.path.join(validation_dir, "dogs")

test_cats_dir = os.path.join(test_dir, "cats")
test_dogs_dir = os.path.join(test_dir, "dogs")



"""
PREPROCESSING IMAGE DATA INTO SCALAR TENSORS 0-1
	general process:
	1) read picture files
	2) Decode the JPEG into a pixel grid of RBG colors
	3) Convert these into floating point tensors, one for each R B and G channels
	4) Rescale the pixel values in between 0-1 to represent how much of a color is present
"""

#Rescales all images by 1/255 to get the % saturation of color, since there are a possible 256 values of saturation
"""


Data Augmentation: This is the step where we would like to randomly transform our datapoints, in hopes of achieving higher
validation accuracy.

"""
train_datagen = ImageDataGenerator(rescale=1./255,
									fill_mode = "nearest",
									rotation_range = 15,
									shear_range = 15
									
									)
test_datagen = ImageDataGenerator(rescale=1./255)

#these generators will pull out the images from the directories, while rescaling their RBG channels, resizing the image,
#	 delivering the images in batches of 20, and classifying them binarily, all with the purpose of being able to 
#	iterate through the items while using the model.fit function 
train_generator = train_datagen.flow_from_directory(
									train_dir,
									
									target_size = (150,150),
									batch_size = 20,
									class_mode= 'binary')
validation_generator= test_datagen.flow_from_directory(
									validation_dir,
									target_size= (150,150),
									batch_size= 20,
									class_mode= "binary")
		
"""
BUILDING THE MODEL
"""
model = models.Sequential()
model.add(layers.Conv2D(32, (3,3), activation = 'relu', input_shape=(150,150,3)))
#explanation for line: 32 filters/output channels, 3X3 convolution frame, relu activation function, input_shape=(150x150px, RBG so 3 channels)
model.add(layers.MaxPooling2D((2,2)))
# 2x2 max pooling window
model.add(layers.Conv2D(64, (3,3), activation = 'relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(128, (3,3), activation = 'relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(128, (3,3), activation = 'relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Flatten())
model.add(layers.Dense(512, activation = 'relu'))
model.add(layers.Dense(1, activation = "sigmoid"))
print(model.summary())

model.compile(loss = 'binary_crossentropy', metrics = ['accuracy'], optimizer= optimizers.RMSprop(lr = 1e-5))


"""
FITTING THE MODEL USING THE GENERATOR
you are able to pass a generator instead of a NumPy array into Keras as both the input set, and as the validation set,
but since the generator will run endlessly, you need to specify when it should stop.
In this case, you would not want to train it on pictures/samples it has already seen, so to utilize all 2000 samples we have accumulated,
you would specify that 20 samples per batch, and 100 batches per epoch will equal 2000 samples trained on per epoch.
Similarly for the validation test set
"""

history = model.fit_generator(
			train_generator,
			steps_per_epoch = 100,
			epochs= 20,
			validation_data = validation_generator,
			validation_steps = 50)
"""
SAVING THE MODEL AND INTERPRETTING TRAINING RESULTS
"""
model.save('cats-vs-dogs-data-aug-convn.h5')

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc)+1)
plt.plot(epochs, acc, 'bo', label= "Training acc")
plt.plot(epochs, val_acc, 'b', label= "Validation acc")
plt.title('Training and Validation Accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label= "Training loss")
plt.plot(epochs, val_loss, 'b', label= "Validation loss")
plt.title('Training and Validation Loss')
plt.legend()

plt.show()

