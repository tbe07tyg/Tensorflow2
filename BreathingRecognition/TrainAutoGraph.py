"""
Train our RNN on extracted features or images in Tensorflow 2.0 with autograph conception.
"""
import re
import os
import csv
from tensorflow.python.keras.utils.data_utils import Sequence
from numpy import math
import numpy as np
import glob
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from ModelZoo import LstmReg, Lstm




# load data -------------------------------------------------->
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

# get classes
def get_classes(data, class_limit):
    """Extract the classes from our data. If we want to limit them,
    only return the classes we need."""
    classes = []
    for item in data:
        if item[1] not in classes:
            classes.append(item[1])

    # Sort them.
    classes = sorted(classes)

    # Return.
    if class_limit is not None:
        return classes[:class_limit]
    else:
        return classes

def clean_data(data, seq_length, max_frames, classes):
    """Limit samples to greater than the sequence length and fewer
    than N frames. Also limit it to classes we want to use."""
    data_clean = []
    for item in data:
        if int(item[3]) >= seq_length and int(item[3]) <= max_frames \
                and item[1] in classes:
            data_clean.append(item)

    return data_clean

def split_train_test(data):
    """Split the data into train and test groups."""
    train = []
    test = []
    for item in data:
        if item[0] == 'train':
            train.append(item)
        else:
            test.append(item)
    return train, test

# preprocessing ------------------------------------------------------------->
def get_class_one_hot(class_names, class_str):
    """Given a class as a string, return its number in the classes
    list. This lets us encode and one-hot it for training."""
    # Encode it first.
    label_encoded = class_names.index(class_str)

    # Now one-hot it.
    label_hot = to_categorical(label_encoded, len(class_names))

    assert len(label_hot) == len(class_names)

    return label_hot

def get_target_value_regression(target_str):
    """
    Given a target breathing per minute string value which contains value. Thie extract number in the string only
    for regression training process
    :param target_str: bpm str
    :return:  number in target str
    """
    return re.findall(r"\d+\.?\d*", target_str)[0]

def get_extracted_batch_sequence(batch_records, sequence_path, seq_length,data_type, task_type, class_names):
    """Get the saved extracted features."""
    batch_x_list = []
    batch_y_list = []
    # print("batch sample", batch_records)
    for each in batch_records.numpy():
        filename = each[2].decode("utf-8")
        # print(filename)
        path = os.path.join(sequence_path, filename + '-' + str(seq_length) + \
                            '-' + data_type + '.npy')
        # print(path)
        if os.path.isfile(path):
            batch_x_list.append(np.load(path))
        else:
            raise Exception("please check the npy path")

        if task_type == "classification":
            batch_y_list.append(get_class_one_hot(class_names=class_names, class_str=each[1].decode("utf-8")))
            # print(y)
        elif task_type == "regression":
            # print("here")
            # print(sample[1])
            batch_y_list.append(float(get_target_value_regression(each[1].decode("utf-8"))))
            # print(y)
        else:
            pass
    return np.array(batch_x_list), np.array(batch_y_list)

# use tf.GradientTape TO TRAIN model


# evaluation -------------------------------------------->
 # define average evaluation metrics
train_avg_loss = tf.keras.metrics.Mean(name='train_avg_loss')
train_avg_acc = tf.keras.metrics.Mean(name='train_avg_acc')
train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')

test_avg_loss = tf.keras.metrics.Mean(name='test_avg_loss')
test_avg_acc = tf.keras.metrics.Mean(name='test_avg_acc')
test_accuracy = tf.keras.metrics.CategoricalAccuracy(name='test_accuracy')

@tf.function
def train_step(input_feature, labels, model, optimizer, loss_object):
    with tf.GradientTape() as tape:
        predictions = model(input_feature)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_avg_loss(loss)
    train_avg_acc(train_accuracy(labels, predictions))


# test model:
@tf.function
def test_step(input_feature, labels):
    predictions = model(input_feature)
    t_loss = loss_object(labels, predictions)

    test_avg_loss(t_loss)
    test_avg_acc(test_accuracy(labels, predictions))

def write_tb_logs_scaler(writer, name_list, value_list, step):
    with writer.as_default():
        # optimizer.iterations is actually the entire counter from step 1 to step total batch
        for i in range(len(name_list)):
            tf.summary.scalar(name_list[i], value_list[i].result(), step=step)
            value_list[i].reset_states()  # Clear accumulated values with .reset_states()
        writer.flush()

def write_tb_logs_image(writer, name_list, value_list, step,max_outs):
    with writer.as_default():
        # optimizer.iterations is actually the entire counter from step 1 to step total batch
        for i in range(len(name_list)):
            tf.summary.image(name_list[i], value_list[i].result(), step=step, max_outputs=max_outs)
            value_list[i].reset_states()  # Clear accumulated values with .reset_states()
        writer.flush()

if __name__ == '__main__':
    # hyper parramters
    class_limit = None  # int, can be 1-101 or None
    batch_size = 32
    nb_epoch = 1000
    lr = 2e-6
    model_type = 'lstm'
    seq_length = 40
    max_frames = 300
    data_type = 'features'
    task_type = "classification"
    csv_path_root = "I:\\DeepLearning\\BreathingRecognition\\five-video-classification-methods-master\\data"
    sequence_path = "I:\\DeepLearning\\BreathingRecognition\\five-video-classification-methods-master\\data\\sequences"
    tb_log_root = "I:\\DeepLearning\\TensorflowV2\\BreathingRecognition\\logs"

    # image shape
    feature_length =2048
    image_shape = (seq_length, feature_length)


    data = get_data(csv_path_root)  # get full data list
    print("data：", data)
    print("data_length:", data.__len__())
    class_names = get_classes(data, class_limit)  # get class names
    print("classNames:", class_names)
    print("len of classes:", len(class_names))

    cleaned_data = clean_data(data, seq_length=seq_length, max_frames=max_frames,
                              classes=class_names)  # clean the data with only frame
    print("cleaned_data:", cleaned_data)
    print("len of cleaned data:", len(cleaned_data))
    print("cleaned # of data:", len(data) - len(cleaned_data))

    train_list, test_list = split_train_test(cleaned_data)  # split train and test
    print("len of train:", len(train_list))
    print("len of test:", len(test_list))
    # mySeqGen_Train = mySeqFeatureRegGenerator(batch_size=batch_size, train_test="train", data_type=data_type, train_list=train_list,
    #                                           test_list=test_list, task_type="classification")
    # dataset ---------------------------------------->
    # train_dataset = tf.data.Dataset.from_tensor_slices(train_list).batch(batch_size).map(lambda item: tuple(tf.py_function(get_extracted_sequence, [item], [tf.float32,])))
    train_dataset = tf.data.Dataset.from_tensor_slices(train_list).shuffle(10000).batch(batch_size)
    test_dataset = tf.data.Dataset.from_tensor_slices(test_list).shuffle(10000).batch(batch_size)

    # model design =====================================>
    num_classes = len(class_names)

    model = Lstm(num_classes, image_shape).model()

    # train optimizer and loss
    loss_object = tf.keras.losses.CategoricalCrossentropy()
    optimizer = tf.keras.optimizers.Adam()



    # if log root dir does not exist, the writer will not write log files
    if not os.path.exists(tb_log_root):
        os.makedirs(tb_log_root)
    train_summary_writer = tf.summary.create_file_writer(os.path.join(tb_log_root, 'train')) # tensorboard --logdir /tmp/summaries
    test_summary_writer = tf.summary.create_file_writer(os.path.join(tb_log_root, 'test'))


    # train loop:
    EPOCHS = 5
    total_num_Batchs= math.ceil(len(train_list)/batch_size)
    log_freq = total_num_Batchs
    for epoch in range(EPOCHS):
        for (batch, each_batch) in enumerate(train_dataset):
            # print(each_batch)
            # load input batch features
            train_batch_x, train_batch_y = get_extracted_batch_sequence(batch_records=each_batch, seq_length=seq_length, sequence_path=sequence_path,
                                         data_type=data_type, task_type=task_type, class_names=class_names)
            # print("x.shape:", train_batch_x.shape)
            # print("y.shape:", batch_y.shape)
            train_step(train_batch_x, train_batch_y, model, optimizer, loss_object)

            batch_template = 'Epoch {} - Batch[{}/{}], Loss: {}, Accuracy: {}'

            print(batch_template.format(epoch + 1,
                                        batch + 1,
                                        total_num_Batchs,
                                        train_avg_loss.result(),
                                        train_avg_acc.result() * 100))
            # train_avg_loss.update_state(loss)  # udpate_state use for accumulate values like append?
            # train_avg_acc.update_state(train_accuracy(prediction, batch_y))

        for each_batch in test_dataset:  # validation after one epoch training
            # load input batch features
            test_batch_x, test_batch_y = get_extracted_batch_sequence(batch_records=each_batch, seq_length=seq_length, sequence_path=sequence_path,
                                         data_type=data_type, task_type=task_type, class_names=class_names)

            test_step(test_batch_x, test_batch_y)


        template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
        print(template.format(epoch + 1,
                              train_avg_loss.result(),
                              train_avg_acc.result() * 100,
                              test_avg_loss.result(),
                              test_avg_acc.result() * 100))


        if tf.equal(optimizer.iterations % log_freq, 0):
            print("writing logs to tensorboard")
            # write train logs # with the same name for train and test write will write multiple curves into one plot
            write_tb_logs_scaler(train_summary_writer, ["avg_loss", "avg_acc"],
                                 [train_avg_loss, train_avg_acc], optimizer.iterations//log_freq)

            write_tb_logs_scaler(test_summary_writer, ["avg_loss", "avg_acc"],
                                 [test_avg_loss, test_avg_acc], optimizer.iterations // log_freq)




