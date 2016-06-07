from keras.models import Model
from keras.layers import Input
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD, Adam
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
import cv2, numpy as np
import h5py
import os
from cnn_input import load_all_training_images
from keras.utils.np_utils import to_categorical
import tensorflow as tf

PRETRAINED_WEIGHT_PATH = './models/cnn4_weight.h5'

def load_model(inputs):
    x = ZeroPadding2D((1,1), dim_ordering='tf')(inputs)
    x = Convolution2D(32, 3, 3, activation='relu', dim_ordering='tf')(x)
    x = BatchNormalization(axis=3)(inputs)
    x = ZeroPadding2D((1,1), dim_ordering='tf')(x)
    x = Convolution2D(32, 3, 3, activation='relu', dim_ordering='tf')(x)
    x = BatchNormalization(axis=3)(inputs)
    x = MaxPooling2D((2,2), strides=(2,2), dim_ordering='tf')(x)

    x = ZeroPadding2D((1,1), dim_ordering='tf')(x)
    x = Convolution2D(64, 3, 3, activation='relu', dim_ordering='tf')(x)
    x = BatchNormalization(axis=3)(inputs)
    x = ZeroPadding2D((1,1), dim_ordering='tf')(x)
    x = Convolution2D(64, 3, 3, activation='relu', dim_ordering='tf')(x)
    x = BatchNormalization(axis=3)(inputs)
    x = MaxPooling2D((2,2), strides=(2,2), dim_ordering='tf')(x)

    x = Flatten()(x)
    x = Dense(128, activation='relu', init='he_normal')(x)
    x = Dropout(0.5)(x)
    preds = Dense(10, activation='softmax')(x)

    return preds

if __name__ == "__main__":
    sess = tf.Session()
    K.set_session(sess)
    init = tf.initialize_all_variables()
    sess.run(init)

    with sess.as_default():
        print "Loading Model...\n"
        inputs = Input(shape=(120, 160, 3))
        model = Model(input=inputs, output=load_model(inputs))
        if os.path.isfile(PRETRAINED_WEIGHT_PATH):
            print "Loading Pretrained Weights...\n"
            model.load_weights(PRETRAINED_WEIGHT_PATH)

        print "Compiling Model...\n"
        sgd = SGD(lr=0.001, decay=1e-4, momentum=0.9, nesterov=True)
        #adam = Adam(lr=0.01)
        model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=["accuracy"])

        print "Loading Data..."
        X_train, Y_train = load_all_training_images()

        print "Training Model...\n"
        model.fit(X_train, Y_train, batch_size=128, nb_epoch=200)

        print "Saving Weight...\n"
        model.save_weights('cnn4_weight.h5')
