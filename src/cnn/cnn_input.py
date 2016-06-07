from keras.utils.np_utils import to_categorical
import tensorflow as tf
import numpy as np
import random
import pickle
import csv
import cv2
import os

BATCH_SIZE = 64
IMAGE_HEIGHT = 120
IMAGE_WIDTH = 160
NUM_EPOCHS_PER_DECAY = 5
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 1000
num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / BATCH_SIZE
decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

TRAIN_DIR = './dataset/train'
TEST_DIR = './dataset/test'
META_FILE = './dataset/driver_imgs_list.csv'
SAMPLE_SIZE = 800

def file_list(input_dir, extension):
    flist = []
    for root, dirs, files in os.walk(input_dir):
        for f in files:
            if f[-4:] == '.' + extension:
                flist.append(os.path.join(root, f))
    return flist

def encode_label(labels):
    return {l:i for i,l in enumerate(labels)}

def load_label_map():
    label_map = {}
    csv_data = csv.reader(open(META_FILE))
    row_num = 0
    encoded_label = []
    for row in csv_data:
        if row_num == 0:
            tags = row
        else:
            label_map[row[2]] = row[1]
        row_num = row_num + 1
    encoded_label = encode_label(set(label_map.values()))
    print "Encoded label: %s" % str(encoded_label)
    return { f:encoded_label[l] for f, l in label_map.iteritems() }

def select(flist, label_map, n):
    inv_map = {}
    for f in flist:
        k = label_map[os.path.basename(f)]
        inv_map.setdefault(k, []).append(f)
    return [v for k in inv_map for v in random.sample(inv_map[k], n) ]

def prepare_train_file_and_labels(sample=False, shuffle=True):
    label_map = load_label_map()
    flist = file_list(TRAIN_DIR, 'jpg')
    if sample:
        flist = select(flist, label_map, SAMPLE_SIZE)
    if shuffle:
        random.shuffle(flist)
    labels = [label_map[os.path.basename(f)] for f in flist]
    labels = to_categorical(labels)
    return [(f,l) for f,l in zip(flist, labels)]

def epochs(inputs):
    for i in range(NUM_EPOCHS_PER_DECAY):
        print 'Epoch %d/%d' % (i, NUM_EPOCHS_PER_DECAY)
        images = np.asarray([read_image(f) for (f,l) in inputs])
        labels = np.asarray([l for (f,l) in inputs])
        yield (images, label)

def batches(epoch):
    for (images, labels) in range(num_batches_per_epoch):
        start = i * num_batches_per_epoch
        end = start + num_batches_per_epoch
        if cur >= len(epoch):
            return
        yield images[start:end], labels[start:end]

def prepare_test_files(sample=False, shuffle=True):
    flist = file_list(TEST_DIR, 'jpg')
    if sample:
        flist = random.sample(flist, SAMPLE_SIZE*3)
    if shuffle:
        random.shuffle(flist)
    return flist

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

def load_all_training_images():
    if os.path.isfile('sample.pickle'):
        with open('sample.pickle') as f:
            images, labels = pickle.load(f)
            return np.asarray(images), np.asarray(labels)

    train_inputs = prepare_train_file_and_labels(sample=True)
    print 'Reading from %d files...' % len(train_inputs)
    images = [read_image(f) for (f,l) in train_inputs]
    labels = [l for (f,l) in train_inputs]
    with open('sample.pickle', 'w+') as f:
        pickle.dump([images, labels], f)
    return images, labels
