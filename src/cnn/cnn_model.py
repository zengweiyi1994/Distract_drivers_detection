from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD, Adam
from keras import backend as T
from keras.preprocessing.image import ImageDataGenerator
import cv2, numpy as np
import h5py
from cnn_input import load_data
from keras.utils.np_utils import to_categorical
import tensorflow as tf

def data_generator():
    return ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True)

def load_model():
    inputs = tf.placeholder(tf.float32, shape=(None, 3, 120, 160))
    x = BatchNormalization()(inputs)

    x = ZeroPadding2D((1,1))(x)
    x = Convolution2D(64, 3, 3, activation='relu')(x)
    x = ZeroPadding2D((1,1))(x)
    x = Convolution2D(64, 3, 3, activation='relu')(x)
    x = MaxPooling2D((2,2), strides=(2,2))(x)

    x = ZeroPadding2D((1,1))(x)
    x = Convolution2D(128, 3, 3, activation='relu')(x)
    x = ZeroPadding2D((1,1))(x)
    x = Convolution2D(128, 3, 3, activation='relu')(x)
    x = MaxPooling2D((2,2), strides=(2,2))(x)

    x = Flatten()(x)
    x = Dense(4096, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(4096, activation='relu')(x)
    x = Dropout(0.5)(x)
    preds = Dense(10, activation='softmax')(x)

    return preds

if __name__ == "__main__":
    print "Loading Data..."
    X_train, Y_train = load_data()
    Y_train = to_categorical(Y_train)
    print '%d training files loaded.' % X_train.shape[0]

#    data_gen = data_generator()
#    data_gen.fit(X)

    print "Loading Model...\n"
    model = load_model()

    print "Compiling Model...\n"
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    #adam = Adam(lr=0.01)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=["accuracy"])

    print "Training Model...\n"
    model.fit(X_train, Y_train, batch_size=32, nb_epoch=5)

    print "Saving Weight...\n"
    model.save_weights('cnn4_weight.h5')
