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

def cnn_model(inputs):
    inputs = tf.placeholder(tf.float32, shape=(None, 3, 224, 224))

    x = ZeroPadding2D((1,1))(inputs)
    x = Convolution2D(64, 3, 3, activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2,2), strides=(2,2))(x)

    x = ZeroPadding2D((1,1))(x)
    x = Convolution2D(128, 3, 3, activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2,2), strides=(2,2))(x)

    x = Flatten()(x)
    x = Dense(4096, activation='relu')(x)
    x = Dropout(0.5)(x)
    preds = Dense(10, activation='softmax')(x)

    return preds

if __name__ == "__main__":
    print "Loading Data..."
    X_train, Y_train, X_test, Y_test = load_data()
    Y_train = to_categorical(Y_train)
    Y_test = to_categorical(Y_test)
    print '%d training files loaded.' % X_train.shape[0]
    print '%d testing files loaded.\n' % X_test.shape[0]

#    data_gen = data_generator()
#    data_gen.fit(X)

    print "Loading Model...\n"
    model = cnn_model()

    print "Compiling Model...\n"
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    #adam = Adam(lr=0.01)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=["accuracy"])

    print "Training Model...\n"
    model.fit(X_train, Y_train, batch_size=32, nb_epoch=5)
#    model.fit_generator(datagen.flow(X, Y, batch_size=32), samples_per_epoch=X.shape[0], nb_epoch=1)

    print "Saving Weight...\n"
    model.save_weights('vgg16_weights_new.h5')

    print '%d files loaded.\n' % X.shape[0]

#    data_gen = data_generator()
#    data_gen.fit(X)

    print "Loading Model...\n"
    model = CNN()

    print "Compiling Model...\n"
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    #adam = Adam(lr=0.01)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=["accuracy"])

    print "Training Model...\n"
    model.fit(X_train, Y_train, batch_size=32, nb_epoch=5)
#    model.fit_generator(datagen.flow(X, Y, batch_size=32), samples_per_epoch=X.shape[0], nb_epoch=1)

    print "Saving Weight...\n"
    model.save_weights('vgg16_weights_new.h5')

    print "Predicting...\n"
    out = model.predict(X_test)
    print np.argmax(out)

