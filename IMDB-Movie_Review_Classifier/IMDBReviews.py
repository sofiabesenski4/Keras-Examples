#Classifying movie Reviews, p. 69 in Deep learning with Python
from keras.datasets import imdb
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)


#print(train_data[0], train_labels[0])
