from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten

#Create the Sequential model
model = Sequential()

#1st Layer - Add an input layer of 32 nodes
model.add(Dense, input_dim=32)

#2nd Layer - Add a fully connected layer of 128 nodes
model.add(Dense(128))

#3rd Layer - Add a softmax activation layer
model.add(Activation('softmax'))

#4th Layer - Add a fully connected layer
model.add(Dense(10))

#5th Layer - Add a Sigmoid activation layer
model.add(Activation('sigmoid'))
