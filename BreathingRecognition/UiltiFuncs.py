import tensorflow as tf
from tensorflow import keras
import os
import csv

def schedule(epoch):
    BASE_LEARNING_RATE = 0.1
    BS_PER_GPU =128
    LR_SCHEDULE = [(0.1, 30), (0.01, 45)]
    initial_learning_rate = BASE_LEARNING_RATE * BS_PER_GPU / 128
    learning_rate = initial_learning_rate
    for mult, start_epoch in LR_SCHEDULE:
        if epoch >= start_epoch:
          learning_rate = initial_learning_rate * mult
        else:
          break
    tf.summary.scalar('learning rate', data=learning_rate, step=epoch)
    return learning_rate


def get_compilied_model(model, opt, loss, metrics):
    model.compile(optimizer=opt,
                  loss=loss,
                  metrics=metrics)
    return model





