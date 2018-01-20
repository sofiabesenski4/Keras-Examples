"""
Cats vs. Dogs Image preparation
jan 16th 2018
Example from Deep Learning with Python, Francois Challet
Thomas Besenski
"""
import os, shutil

#Initially, we have a zip file with 25,000 images in it, and we want to divide that into separate training,
# validation and testing sets of images for both cats and dogs.

original_dataset_dir = '/home/teb8/School/Research/Keras-Examples/Cats-vs-Dogs-Conv/train'
base_dir = '/home/teb8/School/Research/Keras-Examples/Cats-vs-Dogs-Conv/Training-Subset'
os.mkdir(base_dir)

#making parent folders
train_dir = os.path.join(base_dir, "train")
os.mkdir(train_dir)
validation_dir = os.path.join(base_dir, "validation")
os.mkdir(validation_dir)
test_dir = os.path.join(base_dir, "test")
os.mkdir(test_dir)

#making specific folders
train_cats_dir = os.path.join(train_dir,'cats')
os.mkdir(train_cats_dir)
train_dogs_dir = os.path.join(train_dir,'dogs')
os.mkdir(train_dogs_dir)

validation_cats_dir = os.path.join(validation_dir, "cats")
os.mkdir(validation_cats_dir)
validation_dogs_dir = os.path.join(validation_dir, "dogs")
os.mkdir(validation_dogs_dir)

test_cats_dir = os.path.join(test_dir, "cats")
os.mkdir(test_cats_dir)
test_dogs_dir = os.path.join(test_dir, "dogs")
os.mkdir(test_dogs_dir)

fnames = ['cat.{}.jpg'.format(i) for i in range(1000)]
for fname in fnames:
	src = os.path.join(original_dataset_dir,fname)
	dst = os.path.join(train_cats_dir,fname)
	shutil.copyfile(src,dst)

fnames = ['cat.{}.jpg'.format(i) for i in range(1000,1500)]
for fname in fnames:
	src = os.path.join(original_dataset_dir,fname)
	dst = os.path.join(validation_cats_dir,fname)
	shutil.copyfile(src,dst)

fnames = ['cat.{}.jpg'.format(i) for i in range(1500,2000)]
for fname in fnames:
	src = os.path.join(original_dataset_dir,fname)
	dst = os.path.join(test_cats_dir,fname)
	shutil.copyfile(src,dst)

fnames = ['dog.{}.jpg'.format(i) for i in range(1000)]
for fname in fnames:
	src = os.path.join(original_dataset_dir,fname)
	dst = os.path.join(train_dogs_dir,fname)
	shutil.copyfile(src,dst)

fnames = ['dog.{}.jpg'.format(i) for i in range(1000,1500)]
for fname in fnames:
	src = os.path.join(original_dataset_dir,fname)
	dst = os.path.join(validation_dogs_dir,fname)
	shutil.copyfile(src,dst)

fnames = ['dog.{}.jpg'.format(i) for i in range(1500,2000)]
for fname in fnames:
	src = os.path.join(original_dataset_dir,fname)
	dst = os.path.join(test_dogs_dir,fname)
	shutil.copyfile(src,dst)


#Now lets just doublecheck how many we got in each


