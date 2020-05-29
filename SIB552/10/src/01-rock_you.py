# Small LSTM Network to Generate Text for Alice in Wonderland
import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
import pandas as pd
# load ascii text and covert to lowercase

filename = "rockyou_simple.txt"
raw_text = open(filename).read().replace("\r", " ").replace("\n"," ")[0:50000]
#raw_text = raw_text.lower()

# create mapping of unique chars to integers
chars = sorted(list(set(raw_text)))
char_to_int = dict((c, i) for i, c in enumerate(chars))
# summarize the loaded data
n_chars = len(raw_text)
n_vocab = len(chars)
print( "Total Characters: ", n_chars)
print( "Total Vocab: ", n_vocab)
# prepare the dataset of input to output pairs encoded as integers
seq_length = 100
dataX = []
dataY = []
for i in range(0, n_chars - seq_length, 1):
	seq_in = raw_text[i:i + seq_length]
	seq_out = raw_text[i + seq_length]
	dataX.append([char_to_int[char] for char in seq_in])
	dataY.append(char_to_int[seq_out])
n_patterns = len(dataX)
print( "Total Patterns: ", n_patterns)
# reshape X to be [samples, time steps, features]
X = numpy.reshape(dataX, (n_patterns, seq_length, 1))
# normalize
X = X / float(n_vocab)
# one hot encode the output variable
y = np_utils.to_categorical(dataY)

print("y",y.shape)
print("x",X.shape)


# define the LSTM model
model = Sequential()
model.add(LSTM(128, input_shape=(X.shape[1], X.shape[2]), activation='tanh', return_sequences=True))
model.add(Dropout(0.1))
model.add(LSTM(128, activation='tanh'))
model.add(Dropout(0.1))
model.add(Dense(y.shape[1], activation='softmax'))


filename = "01-rockyou.hdf5"
model.load_weights(filename)


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# define the checkpoint
checkpoint = ModelCheckpoint(filename, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]
# fit the model
model.fit(X, y, epochs=2000, batch_size=1000, callbacks=callbacks_list, verbose=1)
