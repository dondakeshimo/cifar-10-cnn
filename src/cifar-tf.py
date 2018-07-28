import pickle
import tensorflow as tf


'''
Global Parameters
'''
n_epochs = 11
batch_size = 1
g_lr = 0.0025
d_lr = 0.00001
beta = 0.5
d_thresh = 0.8
z_size = 200
leak_value = 0.2
cube_len = 64
obj_ratio = 0.7
obj = "chair"

train_sample_directory = "./train_sample/"
model_directory = "./models/"
is_local = True


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def cnn(x):
    xavier_init = tf.contrib.layers.xavier_initializer()

    with tf.variable_scope("cifar"):
        with tf.variable_scope("input"):
            z = tf.reshape(x, (batch_size, 32, 32, 3))

        with tf.variable_scope("conv1"):
            wg_1 = tf.get_variable("w1",
                                   shape=[4, 4, 4, 512, 200],
                                   initializer=xavier_init)
            g_1 = tf.nn.conv3d_transpose(z,
                                         wg_1,
                                         (batch_size, 4, 4, 4, 512),
                                         strides=[1, 1, 1, 1, 1],
                                         padding="VALID")
            g_1 = tf.contrib.layers.batch_norm(g_1, is_training=True)
            g_1 = tf.nn.relu(g_1)

        with tf.variable_scope("layer2"):
            wg_2 = tf.get_variable("w2",
                                   shape=[4, 4, 4, 256, 512],
                                   initializer=xavier_init)
            g_2 = tf.nn.conv3d_transpose(g_1,
                                         wg_2,
                                         (batch_size, 8, 8, 8, 256),
                                         strides=strides,
                                         padding="SAME")
            g_2 = tf.contrib.layers.batch_norm(g_2, is_training=phase_train)
            g_2 = tf.nn.relu(g_2)

        with tf.variable_scope("layer3"):
            wg_3 = tf.get_variable("w3",
                                   shape=[4, 4, 4, 128, 256],
                                   initializer=xavier_init)
            g_3 = tf.nn.conv3d_transpose(g_2,
                                         wg_3,
                                         (batch_size, 16, 16, 16, 128),
                                         strides=strides,
                                         padding="SAME")
            g_3 = tf.contrib.layers.batch_norm(g_3, is_training=phase_train)
            g_3 = tf.nn.relu(g_3)

        with tf.variable_scope("layer4"):
            wg_4 = tf.get_variable("w4",
                                   shape=[4, 4, 4, 64, 128],
                                   initializer=xavier_init)
            g_4 = tf.nn.conv3d_transpose(g_3,
                                         wg_4,
                                         (batch_size, 32, 32, 32, 64),
                                         strides=strides,
                                         padding="SAME")
            g_4 = tf.contrib.layers.batch_norm(g_4, is_training=phase_train)
            g_4 = tf.nn.relu(g_4)

        with tf.variable_scope("layer5"):
            wg_5 = tf.get_variable("w5",
                                   shape=[4, 4, 4, 1, 64],
                                   initializer=xavier_init)
            g_5 = tf.nn.conv3d_transpose(g_4,
                                         wg_5,
                                         (batch_size, 64, 64, 64, 1),
                                         strides=strides,
                                         padding="SAME")
            # g_5 = tf.nn.sigmoid(g_5)
            g_5 = tf.nn.tanh(g_5)

    print("g1: ", g_1)
    print("g2: ", g_2)
    print("g3: ", g_3)
    print("g4: ", g_4)
    print("g5: ", g_5)

    return g_5
    return out

data_batch_1 = unpickle("data/cifar-10-batches-py/data_batch_1")
data_batch_1
data_batch_1[b"data"].shape
len(unpickle("data/cifar-10-batches-py/data_batch_1")[b"labels"])
len(unpickle("data/cifar-10-batches-py/data_batch_2")[b"labels"])
len(unpickle("data/cifar-10-batches-py/data_batch_3")[b"labels"])
len(unpickle("data/cifar-10-batches-py/data_batch_4")[b"labels"])
len(unpickle("data/cifar-10-batches-py/data_batch_5")[b"labels"])
