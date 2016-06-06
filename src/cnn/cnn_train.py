import cnn_input
import cnn_model
from keras import backend as K
from keras.objectives import categorical_crossentropy
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('max_steps', 1000000,
                             """Number of batches to run.""")

NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = cnn_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
INITIAL_LEARNING_RATE = 0.1
LEARNING_RATE_DECAY_FACTOR = 0.1
NUM_EPOCHS_PER_DECAY = 200
num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

with tf.Graph.as_default():
    global_step = tf.Variable(0, trainable=False)

    X_train, Y_train, X_test, Y_test = cnn_input.load_data()
    preds = load_model()
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
    tf.train.start_queue_runners(sess=sess)

    with sess.as_default():
        for i in range(FLAGS.max_steps):
            start_time = time.time()
            assert not np.isnan(loss_value), 'Model diverged with loss = NaN'
            train_step.run(feed_dict={img: X_train,
                                  labels: Y_train,
                                  K.learning_phase(): 1})

    labels = tf.placeholder(tf.float32, shape=(None, 10))
    acc_value = accuracy(labels, preds)
    with sess.as_default():
        print acc_value.eval(feed_dict={img: X_test,
                                    labels: Y_test,
                                    K.learning_phase(): 0})
