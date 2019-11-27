import tensorflow as tf
from tensorflow import keras
import os
import csv
import os.path
import glob
import random




class LstmReg(keras.Model):
    def __init__(self, input_shape):
        super(LstmReg, self).__init__()
        self.s = input_shape
        self.lstm = tf.keras.layers.LSTM(2048, return_sequences=False, dropout=0.5)
        self.dense512 = tf.keras.layers.Dense(512, activation=tf.nn.relu)
        self.dropout = tf.keras.layers.Dropout(0.5)
        self.dense1 = tf.keras.layers.Dense(1, activation=tf.nn.relu)

    def call(self, inputs, training=True):
        x = self.lstm(inputs, training=training)
        x = self.dense512(x)
        if training:
            x = self.dropout(x, training=training)
        x = self.dense1(x)
        return x

    def model(self):
        """
        this is a expicitly method to use subclass model
        """
        i = self.keras.Input(self.s)

        return keras.Model(inputs=[i], outputs=self.call(i))


class Lstm(keras.Model):
    def __init__(self, num_class,input_shape):
        super(Lstm, self).__init__()
        self.s = input_shape
        self.lstm = tf.keras.layers.LSTM(2048, return_sequences=False, dropout=0.5)
        self.dense512 = tf.keras.layers.Dense(512, activation=tf.nn.relu)
        self.dropout = tf.keras.layers.Dropout(0.5)
        self.dense1 = tf.keras.layers.Dense(num_class, activation=tf.nn.softmax)


    def call(self, inputs, training=True):
        # print("input shape:", inputs.shape)

        x = self.lstm(inputs, training=training)
        x = self.dense512(x)
        if training:
            x = self.dropout(x, training=training)
        x = self.dense1(x)
        return x

    def model(self):
        """
        this is a expicitly method to use subclass model
        """

        # write summaries

        i = keras.Input(self.s)


        return keras.Model(inputs=[i], outputs=self.call(i))

class Lstm_signal_record_regression(keras.Model):
    def __init__(self,input_shape):
        super(Lstm_signal_record_regression, self).__init__()
        self.s = input_shape
        self.lstm = tf.keras.layers.LSTM(2048, return_sequences=False, dropout=0.5)
        self.dense512 = tf.keras.layers.Dense(512, activation=tf.nn.relu)
        self.dropout = tf.keras.layers.Dropout(0.5)
        self.dense1 = tf.keras.layers.Dense(1, activation=tf.nn.softmax)


    def call(self, inputs, training=True):
        # print("input shape:", inputs.shape)

        x = self.lstm(inputs, training=training)
        x = self.dense512(x)
        if training:
            x = self.dropout(x, training=training)
        x = self.dense1(x)
        return x

    def model(self):
        """
        this is a expicitly method to use subclass model
        """

        # write summaries

        i = keras.Input(self.s)


        return keras.Model(inputs=[i], outputs=self.call(i))