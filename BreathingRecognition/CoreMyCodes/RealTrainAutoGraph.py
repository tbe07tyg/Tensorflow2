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
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import TensorBoard, LearningRateScheduler
from CustomGenerator import GetBatchGenerator
from ModelZoo import LstmReg, Lstm, Lstm_signal_record_regression
from sklearn.externals import joblib
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
cpus = tf.config.experimental.list_physical_devices(device_type='CPU')
print(gpus, cpus)
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# load data -------------------------------------------------->
def get_data(csv_path):
    """Load our data records from file."""
    with open(csv_path, 'r') as fin:
        reader = csv.reader(fin)
        data = list(reader)

    return data


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

def get_extracted_batch_sequence(batch_records, class_names=None):
    """Get the saved extracted features."""
    batch_x_list = []
    batch_y_list = []

    # print("batch_records", batch_records)
    # print("batch sample", batch_records)
    for each in batch_records.numpy():
        # batch record is as formart: [train or test, target value, npy path without suffix, # of frames]
        # print(filename)

        path = each[2].decode("utf-8") + '.npy'
        # print(path)
        if os.path.isfile(path):
            batch_x_list.append(np.load(path))
        else:
            raise Exception("please check the npy path")

        if not class_names == None:
            batch_y_list.append(get_class_one_hot(class_names=class_names, class_str=each[1].decode("utf-8")))
            # print(y)
        else:
            # print("here")
            # print(sample[1])
            batch_y_list.append(float(get_target_value_regression(each[1].decode("utf-8"))))
            # print(y)

    return np.array(batch_x_list), np.array(batch_y_list)

# use tf.GradientTape TO TRAIN model


# evaluation -------------------------------------------->
train_avg_loss = tf.keras.metrics.Mean(name='train_avg_loss')
train_avg_metric = tf.keras.metrics.Mean(name='train_avg_acc')
test_avg_loss = tf.keras.metrics.Mean(name='test_avg_metric')
test_avg_metric = tf.keras.metrics.Mean(name='test_avg_metric')


@tf.function
def train_step(input_feature, labels, model, optimizer):

    with tf.GradientTape() as tape:
        predictions = model(input_feature)
        loss = tf.keras.losses.mean_squared_error(labels, predictions)
        mae =  tf.keras.losses.mae(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_avg_loss(loss)
    train_avg_metric(mae)


# test model: # tf.function will build graph when loading the script
@tf.function
def test_step(input_feature, labels):
    predictions = model(input_feature)
    t_loss =  tf.keras.losses.mean_squared_error(labels, predictions)
    t_mae = tf.keras.losses.mae(labels, predictions)
    test_avg_loss(t_loss)
    test_avg_metric(t_mae)

def write_tb_logs_scaler(writer, name_list, value_list, step):
    with writer.as_default():
        # optimizer.iterations is actually the entire counter from step 1 to step total batch
        for i in range(len(name_list)):
            tf.summary.scalar(name_list[i], value_list[i].result(), step=step)
            # print(value_list[i].result())
            value_list[i].reset_states()  # Clear accumulated values with .reset_states()
            # print(value_list[i].result())
        writer.flush()

def write_tb_logs_image(writer, name_list, value_list, step,max_outs):
    with writer.as_default():
        # optimizer.iterations is actually the entire counter from step 1 to step total batch
        for i in range(len(name_list)):
            # print(value_list[i].shape)
            batch_images = np.expand_dims(value_list[i], -1)
            # print(batch_images.shape)
            tf.summary.image(name_list[i], batch_images, step=step, max_outputs=max_outs)
            # value_list[i].reset_states()  # Clear accumulated values with .reset_states()
        writer.flush()

def write_tb_model_graph(writer, name, step, logdir):
    with writer.as_default():
        tf.summary.trace_export(
            name=name,
            step=step,
            profiler_outdir=logdir)

def train_and_checkpoint(model, manager, EPOCHS,log_freq, ckpt_freq):
    temp_mae = 100 # mae the less the better

    ckpt.restore(manager.latest_checkpoint)

    if manager.latest_checkpoint:
        print("Restored from {}".format(manager.latest_checkpoint))
    else:
        print("Initializing from scratch.")


    for epoch in range(EPOCHS):
        test_avg_metric_list = []
        for (batch, each_batch) in enumerate(train_dataset):
            # print("each_batch", each_batch)

            # load input batch features
            train_batch_x, train_batch_y = get_extracted_batch_sequence(batch_records=each_batch)
            # print("train_batch_x.shape:", train_batch_x.shape)
            # print("train_batch_y.shape:", train_batch_y.shape)
            # print(type(train_batch_x))
            # print(type(train_batch_y))
            # write_tb_logs_image(train_summary_writer, ["input_features"], [train_batch_x], optimizer.iterations, batch_size)


            train_step(train_batch_x, train_batch_y, model, optimizer)
        #
            batch_template = 'Epoch {} - Batch[{}/{}], Train Avg Loss: {}, Train Avg MAE: {}'
        #
            print(batch_template.format(int(ckpt.step),
                                        batch + 1,
                                        train_total_Batches,
                                        train_avg_loss.result(),
                                        train_avg_metric.result()))


            if batch==0:
                print("write model graph")
                tf.summary.trace_on(graph=True, profiler=True)
                write_tb_model_graph(train_summary_writer, "trainGraph", 0, tb_log_root)
        for (test_batch, each_batch) in enumerate(test_dataset):  # validation after one epoch training
            # load input batch features
            test_batch_x, test_batch_y = get_extracted_batch_sequence(batch_records=each_batch)

            test_step(test_batch_x, test_batch_y)
            batch_template = 'Epoch {} - Batch[{}/{}], test Avg Loss: {}, test Avg MAE: {}'
            test_avg_metric_list.append(test_avg_metric.result())
            print(batch_template.format(int(ckpt.step),
                                        test_batch + 1,
                                        test_total_Batches,
                                        test_avg_loss.result(),
                                        test_avg_metric.result()))
        test_avg_metric_e=  sum(test_avg_metric_list)/len(test_avg_metric_list)
        template = 'Validation Epoch {}, Train Avg Loss: {}, Train Avg MAE: {}, Test Avg Loss: {}, Test Avg MAE: {}'
        print(template.format(int(ckpt.step),
                              train_avg_loss.result(),
                              train_avg_metric.result() ,
                              test_avg_loss.result(),
                              test_avg_metric_e))
        #

        #
        if int(ckpt.step) % ckpt_freq == 0 and test_avg_metric_e <temp_mae:
            print("save model...")
            temp_mae = test_avg_metric.result()
            save_path = manager.save()
            print("Saved checkpoint for epoch {}: {}".format(int(ckpt.step), save_path))
            print("Where test mae {:1.2f} ".format(test_avg_metric_e))


        if tf.equal(optimizer.iterations % log_freq, 0):
            print("writing logs to tensorboard")
            # write train logs # with the same name for train and test write will write multiple curves into one plot
            write_tb_logs_scaler(train_summary_writer, ["avg_loss", "avg_MAE"],
                                 [train_avg_loss, train_avg_metric], optimizer.iterations // log_freq)

            write_tb_logs_scaler(test_summary_writer, ["avg_loss", "avg_MAE"],
                                 [test_avg_loss, test_avg_metric], optimizer.iterations // log_freq)

        ckpt.step.assign_add(1)


#  Sequence generator




if __name__ == '__main__':


    # hyper parramters
    class_limit = None  # int, can be 1-101 or None
    batch_size = 32

    lr = 5e-6
    model_type = 'lstm'
    seq_length = 40
    max_frames = 300
    data_type = 'features'
    task_type = "classification"
    my_training_pairs_path =  "my_training_pairs.csv"

    tb_log_root = "logs"
    ckpt_log_root = "ckpt"

    # image shape
    feature_length =2048
    image_shape = (seq_length, feature_length)


    data = get_data(my_training_pairs_path)  # get full data list
    # print("data：", data)
    print("data_length:", data.__len__())

    #

    #
    train_list, test_list = split_train_test(data)  # split train and test
    print("len of train samples:", len(train_list))
    print("len of test samples:", len(test_list))
    # print("train_list:", train_list)  # train_list: [['train', '-0.6708105123895995', 'I:\\dataset\\BreathingData_16_29\\sequences\\Rec20191108_014327_21.13-40-features_0-40', '320'], ['train', '-0.520847926910924', 'I:\\dataset\\BreathingData_16_29\\sequences\\Rec20191108_014327_21.13-40-features_1-41', '320'],
    train_dataset = tf.data.Dataset.from_tensor_slices(train_list).shuffle(10000).batch(batch_size)
    test_dataset = tf.data.Dataset.from_tensor_slices(test_list).shuffle(10000).batch(batch_size)
    # print("train_dataset", train_dataset)
    # print("train_dataset", test_dataset)
    #
    # model design =====================================>

    model = Lstm_signal_record_regression(image_shape).model()
    #
    print(model.summary())
    # train optimizer and loss

    # define evaluation metrics
    optimizer = tf.keras.optimizers.Adam(lr)

    ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, net=model)
    #
    #
    # if log root dir does not exist, the writer will not write log files
    if not os.path.exists(ckpt_log_root):
        print("build ckpt folder")
        os.makedirs(ckpt_log_root)

    if not os.path.exists(tb_log_root):
        print("build tensorboard log folder")
        os.makedirs(tb_log_root)
    train_summary_writer = tf.summary.create_file_writer(os.path.join(tb_log_root, 'train')) # tensorboard --logdir /tmp/summaries
    test_summary_writer = tf.summary.create_file_writer(os.path.join(tb_log_root, 'test'))
    manager = tf.train.CheckpointManager(ckpt, ckpt_log_root, max_to_keep=3)
    #
    # train loop:
    EPOCHS = 1000
    train_total_Batches= math.ceil(len(train_list)/batch_size)
    test_total_Batches =  math.ceil(len(test_list)/batch_size)
    log_freq = train_total_Batches
    ckpt_freq  = 1 # 1 epoch




    train_and_checkpoint(model, manager, EPOCHS, log_freq, ckpt_freq)
    #
    #
    #
