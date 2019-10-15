import tensorflow as tf
from tensorflow import keras
import os
import csv
import os.path
import glob
import random




class LstmReg(keras.Model):
    def __init__(self):
        super(LstmReg, self).__init__()
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

class Lstm(keras.Model):
    def __init__(self, num_class):
        super(Lstm, self).__init__()
        self.lstm = tf.keras.layers.LSTM(2048, return_sequences=False, dropout=0.5)
        self.dense512 = tf.keras.layers.Dense(512, activation=tf.nn.relu)
        self.dropout = tf.keras.layers.Dropout(0.5)
        self.dense1 = tf.keras.layers.Dense(num_class, activation=tf.nn.softmax)


    def call(self, inputs, training=True):
        x = self.lstm(inputs, training=training)
        x = self.dense512(x)
        if training:
            x = self.dropout(x, training=training)
        x = self.dense1(x)
        return x