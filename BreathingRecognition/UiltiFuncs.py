import tensorflow as tf
from tensorflow import keras
import os
import csv

def schedule(epoch):
  initial_learning_rate = BASE_LEARNING_RATE * BS_PER_GPU / 128
  learning_rate = initial_learning_rate
  for mult, start_epoch in LR_SCHEDULE:
    if epoch >= start_epoch:
      learning_rate = initial_learning_rate * mult
    else:
      break
  tf.summary.scalar('learning rate', data=learning_rate, step=epoch)
  return learning_rate


def get_compilied_model(model, opt, loss, metric):
    model.compile(optimizer=keras.optimizers.RMSprop(learning_rate=1e-3),
                  loss='sparse_categorical_crossentropy',
                  metrics=['sparse_categorical_accuracy'])
    return model


# for data generator =========================>
def get_data(csv_path_root):
    """Load our data from file."""
    if csv_path_root == None:
        with open(os.path.join("data", 'data_file.csv'), 'r') as fin:
            reader = csv.reader(fin)
            data = list(reader)
    else:
        with open(os.path.join(csv_path_root, 'data_file.csv'), 'r') as fin:
            reader = csv.reader(fin)
            data = list(reader)

    return data


def split_train_test():


def our_generator(train_test, data_type):
    """Return a generator that we can use to train on. There are
           a couple different things we can return:

           data_type: 'features', 'images'
           """
    # Get the right dataset for the generator.
    train, test = self.split_train_test()
    data = train if train_test == 'train' else test

    print("Creating %s generator with %d samples." % (train_test, len(data)))