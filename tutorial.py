import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib import rnn

class NeuralNetwork(object):

    def __init__(self, timesteps=28, num_units=128, n_input=28, learning_rate=0.00025, n_classes=10, batch_size=128):
        self.input_data = input_data.read_data_sets("/tmp/data", one_hot=True)
        self.train = self.input_data.train
        self.test = self.input_data.test
        self.timesteps = timesteps
        self.num_units = num_units
        self.n_input = n_input
        self.learning_rate = learning_rate
        self.n_classes = n_classes
        self.batch_size = batch_size

        # initialize weights and biases
        self.out_weights = tf.Variable(tf.random_normal([self.num_units, self.n_classes]))
        self.out_bias = tf.Variable(tf.random_normal([self.n_classes]))

        self.x = tf.placeholder(tf.float32, [None, self.timesteps, self.n_input])
        self.y = tf.placeholder(tf.float32, [None, self.n_classes])
        self.input = tf.unstack(self.x, self.timesteps, 1)

        # define network topology
        self.lstm_layer = rnn.BasicLSTMCell(self.num_units, forget_bias=1)
        self.outputs, _ = rnn.static_rnn(self.lstm_layer, self.input, dtype=tf.float32)

        self.prediction = tf.matmul(self.outputs[-1], self.out_weights) + self.out_bias

        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.prediction, labels=self.y))
        self.opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

        self.correct_prediction = tf.equal(tf.argmax(self.prediction, 1), tf.argmax(self.y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))


    def session(self):
        initializer = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(initializer)
            i = 1
            while i < 3000:
                batch_x, batch_y = self.train.next_batch(batch_size=self.batch_size)
                batch_x = batch_x.reshape((self.batch_size, self.timesteps, self.n_input))
                sess.run(self.opt, feed_dict={self.x: batch_x, self.y: batch_y})

                if i % 50 == 0:
                    acc = sess.run(self.accuracy, feed_dict={self.x: batch_x, self.y: batch_y})
                    loss = sess.run(self.loss, feed_dict={self.x: batch_x, self.y: batch_y})
                    print(i, acc, loss)
                    #print()
                i += 1


    def inspect(self):
        print("Training images: ", self.train.images.shape, "labels: ", self.train.labels.shape)
        print("Test images: ", self.test.images.shape, "labels: ", self.test.labels.shape)
        print(type(self.input_data))

nn = NeuralNetwork(timesteps=28, num_units=128, n_input=28, learning_rate=0.001, n_classes=10, batch_size=128)
nn.session()
nn.inspect()