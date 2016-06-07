from keras.utils.np_utils import to_categorical
import tensorflow as tf
import numpy as np
import random
import csv
import cv2
import os
import tensorflow as tf
from keras import backend as K

BATCH_SIZE = 64
IMAGE_HEIGHT = 120
IMAGE_WIDTH = 160
NUM_EPOCHS_PER_DECAY = 5
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 1000

TRAIN_DIR = './dataset/train'
TEST_DIR = './dataset/test'
META_FILE = './dataset/driver_imgs_list.csv'
SAMPLE_RATE = 0.9

def file_list(input_dir, extension):
    flist = []
    for root, dirs, files in os.walk(input_dir):
        for f in files:
            if f[-4:] == '.' + extension:
                flist.append(os.path.join(root, f))
    return flist

def load_file_to_label_map():
    label_map = {}
    csv_data = csv.reader(open(META_FILE))
    row_num = 0
    encoded_label = []
    for row in csv_data:
        if row_num == 0:
            tags = row
        else:
            label_map[row[2]] = int(row[1][-1:])
        row_num = row_num + 1
    raw_labels = list(set(label_map.values()))
    encoded_labels = to_categorical(raw_labels)
    raw_encoded_label_map = {r:e for r,e in zip(raw_labels, encoded_labels)}
    return { f:raw_encoded_label_map[l] for f, l in label_map.iteritems() }

def load_label_to_file_map(flist, label_map):
    inv_map = {}
    for f in flist:
        label = label_map[os.path.basename(f)]
        key = np.where(label==1)[0][0]
        inv_map.setdefault(key, []).append(f)
    return inv_map

def split(label2file, sample_rate):
    train_data = []
    validation_data = []
    for k in label2file:
        random.shuffle(label2file[k])
        TRAIN_NUM = int(sample_rate * len(label2file[k]))
        train_data.extend(label2file[k][0:TRAIN_NUM])
        validation_data.extend(label2file[k][TRAIN_NUM:])
    return train_data, validation_data

def prepare_files():
    flist = file_list(TRAIN_DIR, 'jpg')
    file2label = load_file_to_label_map()
    label2file = load_label_to_file_map(flist, file2label)
    train_files, validation_files = split(label2file, SAMPLE_RATE)
    train_labels = [file2label[os.path.basename(f)] for f in train_files]
    validation_labels = [file2label[os.path.basename(f)] for f in validation_files]
    return train_files, train_labels, validation_files, validation_labels

def read_image(filename):
    contents = tf.read_file(filename)
    image = tf.image.decode_jpeg(contents, channels=3)
    image.set_shape([480, 640, 3])
    reshaped_image = tf.cast(image, tf.float32)
    resized_image = tf.image.resize_image_with_crop_or_pad(reshaped_image,
                                                          IMAGE_WIDTH, IMAGE_HEIGHT)
    #   float_image = tf.image.per_image_whitening(resized_image)
    distorted_image = tf.random_crop(reshaped_image, [IMAGE_HEIGHT, IMAGE_WIDTH, 3])
    distorted_image = tf.image.random_brightness(distorted_image,
                                                 max_delta=63)
    distorted_image = tf.image.random_contrast(distorted_image,
                                               lower=0.2, upper=1.8)
    return distorted_image.eval()

def read_images(filenames):
    images = []
    total = len(filenames)
    for i, filename in enumerate(filenames):
        images.append(read_image(filename))
        if i % 100 == 0:
            print '    %%%.2f completed' % (i/float(total))
    return images

def load_train_data():
    if os.path.isfile('train_data.npz') and os.path.isfile('validation_data.npz'):
        print 'Loading training files from file'
        npzfile = np.load('train_data.npz')
        X_train, Y_train = npzfile['arr_0'], npzfile['arr_1']
        print '(%d, %d) files/labels loaded' % (len(X_train), len(Y_train))

        print 'Loading validation files from file'
        npzfile = np.load('validation_data.npz')
        X_validation, Y_validation = npzfile['arr_0'], npzfile['arr_1']
        print '(%d, %d) files/labels loaded' % (len(X_validation), len(Y_validation))
        return X_train, Y_train, X_validation, Y_validation

    train_files, train_labels, validation_files, validation_labels = prepare_files()
    print 'Loading training files, %d in total' % len(train_files)
    train_data = read_images(train_files)
    print 'Loading validation files, %d in total' % len(validation_files)
    validation_data = read_images(validation_files)

    X_train = np.asarray(train_data)
    Y_train = np.asarray(train_labels)
    X_validation = np.asarray(validation_data)
    Y_validation = np.asarray(validation_labels)

    with open('train_data.npz', 'w+') as f:
        np.savez_compressed(f, X_train, Y_train)
    with open('validation_data.npz', 'w+') as f:
        np.savez_compressed(f, X_validation, Y_validation)

    return X_train, Y_train, X_validation, Y_validation

if __name__ == '__main__':
    sess = tf.Session()
    K.set_session(sess)
    init = tf.initialize_all_variables()
    sess.run(init)

    with sess.as_default():
        load_train_data()
