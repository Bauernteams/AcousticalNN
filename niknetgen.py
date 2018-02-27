# Nikki's Net Generator

import keras as kr

# generate a sequential model object
model = kr.Sequential()


###
from keras.datasets import mnist

(X_train, y_train), (X_test, y_test) = mnist.load_data()