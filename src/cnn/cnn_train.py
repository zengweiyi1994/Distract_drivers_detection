import cnn_input
import cnn_model
import tensorflow as tf
from keras import backend as K
from keras.objectives import categorical_crossentropy

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('train_dir', '/tmp/cnn_train',
    """Directory where to write event logs and checkpoints""")

INITIAL_LEARNING_RATE = 0.005
LEARNING_RATE_DECAY_FACTOR = 0.1

num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / BATCH_SIZE
decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

def batches(epoch):
    for (images, labels) in range(num_batches_per_epoch):
        start = i * num_batches_per_epoch
        end = start + num_batches_per_epoch
        if cur >= len(epoch):
            return
        yield images[start:end], labels[start:end]

def train():
    global_step = tf.Variable(0, trainable=False)
    img = tf.placeholder(tf.float32, shape=(None, 120, 160, 3))
    lbs = tf.placeholder(tf.float32, shape=(None, 10))

    preds = cnn_model.load_model(img)
    loss = tf.reduce_mean(categorical_crossentropy(lbs, preds))
    tf.scalar_summary('loss', loss)
    lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                global_step,
                                cnn_input.decay_steps,
                                LEARNING_RATE_DECAY_FACTOR,
                                staircase=True)
    tf.scalar_summary('learning_rate', lr)
    train_step = tf.train.GradientDescentOptimizer(lr).minimize(loss, global_step=global_step)

    sess = tf.Session()
    K.set_session(sess)
    init = tf.initialize_all_variables()
    sess.run(init)
    summary_writer = tf.train.SummaryWriter(FLAGS.train_dir, sess.graph)

    with sess.as_default():
        train_data = cnn_input.load_train_data()
        for epoch in cnn_input.epochs(train_data):
            for batch in cnn_input.batches(epoch):
                train_step.run(feed_dict={img: batch[0],
                                          lbs: batch[1],
                                          K.learning_phase(): 1})

if __name__ == '__main__':
    if tf.gfile.Exists(FLAGS.train_dir):
        tf.gfile.DeleteRecursively(FLAGS.train_dir)
    tf.gfile.MakeDirs(FLAGS.train_dir)
    train()
