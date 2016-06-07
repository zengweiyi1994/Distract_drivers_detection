import cnn_input
import cnn_model
from keras import backend as K
from keras.objectives import categorical_crossentropy
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('train_dir', '/tmp/cnn_train',
    """Directory where to write event logs and checkpoints""")

INITIAL_LEARNING_RATE = 0.1
LEARNING_RATE_DECAY_FACTOR = 0.1
NUM_EPOCHS_PER_DECAY = 4
decay_steps = int(cnn.num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

def train():
    global_step = tf.Variable(0, trainable=False)
    preds = cnn_model.load_model()
    loss = tf.reduce_mean(categorical_crossentropy(labels, preds))
    lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                global_step,
                                decay_steps,
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
        train_inputs = cnn_input.load_training_inputs()
        for i, epoch in cnn_input.epochs(train_inputs):
            print 'Processing epoch %d' i
            for images, labels in cnn_input.batches(epoch)
                train_step.run(feed_dict={img: images,
                               labels: labels,
                               K.learning_phase(): 1})

if __name__ == '__main__':
    if tf.gfile.Exists(FLAGS.train_dir):
        tf.gfile.DeleteRecursively(FLAGS.train_dir)
    tf.gfile.MakeDirs(FLAGS.train_dir)
    train()
