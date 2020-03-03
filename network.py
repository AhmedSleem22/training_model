import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# for input image
width = 352
hight = 288
channels = 3


learning_rate =0.0002
training_epochs = 100
tf.reset_default_graph()


layer_1 = 32    # number of filter of first layer
layer_2 = 32    # number of filter of second layer
layer_3 = 64    # number of filter of third layer
layer_4 = 64    # number of filter of fourth layer
layer_5 = 128   # number of filter of fifth layer
layer_6 = 128   # number of filter of sixth layer
layer_7 = 256   # number of filter of seventh layer
layer_8 = 256   # number of filter of eighth layer

def model(input_layer, num_classes=3):
    #with tf.variable_scope('input'):
        #X = tf.placeholder(tf.float32, shape=(None, hight, width, channels ))

    with tf.variable_scope('layer_1'):
        weights = tf.get_variable(name="weights1", shape=[5, 5, 3, layer_1], initializer=tf.contrib.layers.xavier_initializer())
        biases = tf.get_variable(name="biases1", shape=[layer_1], initializer=tf.zeros_initializer())
        layer_1_output = tf.nn.conv2d(input_layer, weights, strides=[1, 2, 2, 1], padding="VALID")
        layer_1_output = tf.nn.bias_add(layer_1_output, biases)
        layer_1_output = tf.nn.elu(layer_1_output)

    print("con1 = {}".format(layer_1_output.shape))


    with tf.variable_scope('layer_2'):
        weights = tf.get_variable(name = "weights2", shape=[3, 3, layer_1, layer_2], initializer=tf.contrib.layers.xavier_initializer())
        biases = tf.get_variable(name = "biases2", shape = [layer_2], initializer = tf.zeros_initializer())
        layer_2_output = tf.nn.conv2d(layer_1_output, weights,strides=[1, 1, 1, 1], padding="VALID")
        layer_2_output = tf.nn.bias_add(layer_2_output, biases)
        layer_2_output = tf.nn.elu(layer_2_output)

    print("con2 = {}".format(layer_2_output.shape))


    with tf.variable_scope('layer_3'):
        weights = tf.get_variable(name="weights3", shape=[3, 3,layer_2 , layer_3], initializer=tf.contrib.layers.xavier_initializer())
        biases = tf.get_variable(name="biases3", shape=[layer_3], initializer=tf.zeros_initializer())
        layer_3_output = tf.nn.conv2d(layer_2_output, weights,strides=[1, 2, 2, 1], padding="VALID")
        layer_3_output = tf.nn.bias_add(layer_3_output, biases)
        layer_3_output = tf.nn.elu(layer_3_output)

    print("con3 = {}".format(layer_3_output.shape))


    with tf.variable_scope('layer_4'):
        weights = tf.get_variable(name="weights4", shape=[3, 3,layer_3 , layer_4], initializer=tf.contrib.layers.xavier_initializer())
        biases = tf.get_variable(name="biases4", shape=[layer_4], initializer=tf.zeros_initializer())
        layer_4_output = tf.nn.conv2d(layer_3_output, weights,strides=[1, 1, 1, 1], padding="VALID")
        layer_4_output = tf.nn.bias_add(layer_4_output, biases)
        layer_4_output = tf.nn.elu(layer_4_output)

    print("con4 = {}".format(layer_4_output.shape))


    with tf.variable_scope('layer_5'):
        weights = tf.get_variable(name="weights5", shape=[3, 3,layer_4 , layer_5], initializer=tf.contrib.layers.xavier_initializer())
        biases = tf.get_variable(name="biases5", shape=[layer_5], initializer=tf.zeros_initializer())
        layer_5_output = tf.nn.conv2d(layer_4_output, weights,strides=[1, 2, 2, 1], padding="VALID")
        layer_5_output = tf.nn.bias_add(layer_5_output, biases)
        layer_5_output = tf.nn.elu(layer_5_output)

    print("con5 = {}".format(layer_5_output.shape))


    with tf.variable_scope('layer_6'):
        weights = tf.get_variable(name="weights6", shape=[3, 3,layer_5 , layer_6], initializer=tf.contrib.layers.xavier_initializer())
        biases = tf.get_variable(name="biases6", shape=[layer_6], initializer=tf.zeros_initializer())
        layer_6_output = tf.nn.conv2d(layer_5_output, weights,strides=[1, 1, 1, 1], padding="VALID")
        layer_6_output = tf.nn.bias_add(layer_6_output, biases)
        layer_6_output = tf.nn.elu(layer_6_output)

    print("con6 = {}".format(layer_6_output.shape))


    with tf.variable_scope('layer_7'):
        weights = tf.get_variable(name="weights7", shape=[3, 3,layer_6 , layer_7], initializer=tf.contrib.layers.xavier_initializer())
        biases = tf.get_variable(name="biases7", shape=[layer_7], initializer=tf.zeros_initializer())
        layer_7_output = tf.nn.conv2d(layer_6_output, weights,strides=[1, 1, 1, 1], padding="VALID")
        layer_7_output = tf.nn.bias_add(layer_7_output, biases)
        layer_7_output = tf.nn.elu(layer_7_output)

    print("con7 = {}".format(layer_7_output.shape))


    with tf.variable_scope('layer_8'):
        weights = tf.get_variable(name="weights8", shape=[3, 3,layer_7 , layer_8], initializer=tf.contrib.layers.xavier_initializer())
        biases = tf.get_variable(name="biases8", shape=[layer_8], initializer=tf.zeros_initializer())
        layer_8_output = tf.nn.conv2d(layer_7_output, weights,strides=[1, 1, 1, 1], padding="VALID")
        layer_8_output = tf.nn.bias_add(layer_8_output, biases)
        layer_8_output = tf.nn.elu(layer_8_output)

    print("con8 = {}".format(layer_8_output.shape))

    #flaten layer
    with tf.variable_scope('layer_9'):
        layer_9_output=tf.layers.flatten(layer_8_output)

    print("flaten = {}".format(layer_9_output.shape))

    #dense layer
    with tf.variable_scope('layer_10'):
        layer_10_output = tf.layers.dense(layer_9_output,512)
        layer_10_output = tf.nn.elu(layer_10_output)
    print("dense1 = {}".format(layer_10_output.shape))

    with tf.variable_scope('layer_11'):
        layer_11_output=tf.layers.dense(layer_10_output,512)
        layer_11_output = tf.nn.elu(layer_11_output)

    print("dense2 = {}".format(layer_11_output.shape))


    #dropout layers
    with tf.variable_scope('layer_12'):
        layer_12_output = tf.layers.dropout(layer_11_output)

    print("dropout = {}".format(layer_12_output.shape))

    with tf.variable_scope('output'):
        prediction = tf.layers.dense(layer_12_output, 3)
        prediction = tf.nn.softmax(prediction)

    print("output= {}".format(prediction.shape))
    return prediction



# with tf.variable_scope('cost'):
#     Y = tf.placeholder(tf.float32, shape=(1,))
#     cost = tf.reduce_mean(tf.squared_difference(prediction, Y))
#
# with tf.variable_scope('train'):
#     optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
