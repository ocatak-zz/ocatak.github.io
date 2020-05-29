from keras.models import Sequential
from keras.layers import Convolution2D
import numpy as np
from keras.preprocessing.image import array_to_img, img_to_array, load_img
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
import cv2
from sklearn.model_selection import train_test_split


from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras import callbacks

# https://www.pyimagesearch.com/2017/12/11/image-classification-with-keras-and-deep-learning/

data = []
labels = []

class LeNet:
    @staticmethod
    def build(width, height, depth, classes):
        # initialize the model
        model = Sequential()
        inputShape = (height, width, depth)
        
        '''
        # if we are using "channels first", update the input shape
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
        '''
        # first set of CONV => RELU => POOL layers
        model.add(Conv2D(10, (5, 5), padding="same",
			input_shape=inputShape))
        model.add(Activation("tanh"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        
        # second set of CONV => RELU => POOL layers
        model.add(Conv2D(20, (5, 5), padding="same"))
        model.add(Activation("tanh"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        
        # first (and only) set of FC => RELU layers
        model.add(Flatten())
        model.add(Dense(30))
        model.add(Activation("sigmoid"))

        # softmax classifier
        model.add(Dense(classes))
        model.add(Activation("softmax"))

        # return the constructed network architecture
        return model
        

f = open("results.txt","r")
iter = 0
for line in f:
    iter += 1
    if iter > 30:
        continue
    c,fname = line.split(",")
    fpath = "/home/ozgur/Documents/dataset/pngs/"+ fname.strip()
    
    print(fpath.strip())
    i = cv2.imread(fpath)
    #i = cv2.cvtColor(i, cv2.COLOR_BGR2GRAY)
    #print(i.shape)
    image = cv2.resize(i, (2048,512))
    image = img_to_array(image)
    data.append(image)

    labels.append(c)

print(len(data))

data = np.array(data, dtype="float")
labels = np.array(labels)

print('data.shape',data.shape)
print('labels.shape',labels.shape)

encoder = LabelEncoder()
encoder.fit(labels)
encoded_Y = encoder.transform(labels)
dummy_Y = np_utils.to_categorical(encoded_Y)
y = encoded_Y
print("encoded_Y",encoded_Y)

model = LeNet.build(width=2048, height=512, depth=3, classes=dummy_Y.shape[1])
model.summary()

#filepath="malware-detection-{epoch:02d}-{loss:.4f}.hdf5"
filepath="03-malware-detection.hdf5"

#model.load_weights(filepath)

checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=["accuracy"])

model.fit(data, dummy_Y, verbose=1, epochs=200, batch_size=5, validation_split=0.1 , callbacks=callbacks_list )
