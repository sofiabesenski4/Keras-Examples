#comparing the cats vs dogs non augmented model's history and the augmented one's history

import os, shutil
from keras import models
from keras import layers
from keras import optimizers
import matplotlib as plt

model_data_aug = models.load_model("models/cats-vs-dogs-data-aug-convn.h5")
model_without_data_aug = models.load_model("models/cats-vs-dogs-basic-convn.h5")

"""
Turns out, the history isn't saved within the model, but within another "history" object which is generated
when you tell the model to fit a certain set of data
"""
