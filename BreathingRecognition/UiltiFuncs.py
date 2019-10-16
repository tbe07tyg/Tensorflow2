import tensorflow as tf
from tensorflow import keras
import os
import csv
import re
import glob

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

def compare(x, y):
    stat_x = os.stat(x)
    stat_y = os.stat(y)
    if stat_x.st_ctime < stat_y.st_ctime:
        return -1
    elif stat_x.st_ctime > stat_y.st_ctime:
        return 1
    else:
        return 0

class Clean_CheckpointsCaches(keras.callbacks.Callback):
    """
    this callback will delete all the saved checkpoints except the last 5 savings
    """
    def __init__(self, model_type, feature_type, folder_path):
        self.type = model_type+ "-" +feature_type
        self.folder_path =  folder_path

    def on_epoch_end(self, epoch, logs=None):
        # select =  [f for f in os.listdir(self.folder_path) if self.type in f]
        files = list(filter(os.path.isfile, glob.glob(self.folder_path + "/*")))
        # print(files)
        files.sort(key=lambda x: os.path.getmtime(x))
        # print(files)
        if len(files)>5:
            del_items =  files[5:]
            # print(del_items)
            for file in del_items:
                os.remove(file)
        # print("after delete:", os.listdir(self.folder_path))









