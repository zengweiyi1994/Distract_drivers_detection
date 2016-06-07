from cnn_prepare import prepare_train_files, prepare_test_files
import tensorflow as tf
import cv2

BATCH_SIZE = 64
IMAGE_HEIGHT = 120
IMAGE_WIDTH = 160
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 1000
num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / BATCH_SIZE

def read_image(filename):
    contents = tf.read_file(filename)
    image = tf.image.decode_png(contents, channels=3)
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
    return distorted_image

def load_training_inputs():
    flist, labels = prepare_train_files(sample=True)
    train_inputs = [(f,l) for f,l in zip(flist, labels)]
    return train_inputs

def epochs(inputs):
    for i in range(NUM_EPOCHS_PER_DECAY):
        cur = i * NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
        yield i, inputs[cur:cur + NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN]

def batches(epoch):
    for i in range(num_batches_per_epoch):
        cur = i * num_batches_per_epoch
        batch = epoch[cur:cur + num_batches_per_epoch]
        images = [read_image(f) for (f,l) in batch]
        labels = [l for (f,l) in batch]
        return images, labels

