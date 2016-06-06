from cnn_prepare import prepare_train_files, prepare_test_files
import tensorflow as tf
import cv2

BATCH_SIZE = 32
IMAGE_HEIGHT = 120
IMAGE_WIDTH = 160
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 1000

def read_images(input_queue):
    label = input_queue[1]
    contents = tf.read_file(input_queue[0])
    image = tf.image.decode_png(contents, channels=3)
    image.set_shape([480, 640, 3])
    return image, label

def _generate_image_and_label_batch(image, label, min_queue_examples,
                                    batch_size):
    num_preprocess_threads = 16
    images, label_batch = tf.train.batch(
        [image, label],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * batch_size)

    tf.image_summary('images', images)
    return images, tf.reshape(label_batch, [batch_size])

def load_training_images():
    flist, labels = prepare_train_files(sample=True, n=1000)
    images = tf.convert_to_tensor(flist, dtype=dtypes.string)
    labels = tf.convert_to_tensor(labels, dtype=dtypes.int32)
    input_queue = tf.train.slice_input_producer([images, labels], shuffle=True)

    image, label = read_images(input_queue)
    reshaped_image = tf.cast(image, tf.float32)
    resized_image = tf.image.resize_image_with_crop_or_pad(reshaped_image,
                                                          IMAGE_WIDTH, IMAGE_HEIGHT)
    distorted_image = tf.random_crop(reshaped_image, [IMAGE_HEIGHT, IMAGE_WIDTH, 3])
    distorted_image = tf.image.random_brightness(distorted_image,
                                                 max_delta=63)
    distorted_image = tf.image.random_contrast(distorted_image,
                                               lower=0.2, upper=1.8)
#    float_image = tf.image.per_image_whitening(resized_image)

    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
                               min_fraction_of_examples_in_queue)

    return _generate_image_and_label_batch(distorted_image, label,
                                          min_queue_examples, BATCH_SIZE)

def load_data():
    X_train, Y_train = load_training_images()
    return X_train, Y_train

if __name__ == '__main__':
    X_train, Y_train = load_training_images()
    #X_train, Y_train, X_test, Y_test = load_data()
    #print X_train.get_shape(), Y_train.get_shape()
    #print X_test.get_shape(), Y_test.shape.get_shape()
