from tensorflow.keras.callbacks import TensorBoard, LearningRateScheduler
import tensorflow as tf
from CustomGenerator import GetBatchGenerator
import os
import re
import csv
import pickle

def schedule_train(model, EPOCHS,  sd_tb_log_path, train_list, test_list, batch_size):

    opt = tf.keras.optimizers.Adam(lr=1e-8)  # for learning rate schedular
    lr_schedule = LearningRateScheduler(
        lambda epoch: 1e-8 * 10 ** (epoch / 20))

    tb_callback = TensorBoard(log_dir=sd_tb_log_path, update_freq='epoch', profile_batch=0)
    calls = [lr_schedule, tb_callback]

    # generator
    train_Generator = GetBatchGenerator(data_list=train_list, batch_size=batch_size, classNames=None)
    test_Generator = GetBatchGenerator(data_list=test_list, batch_size=batch_size, classNames=None)

    # build model
    model.compile(optimizer=opt,
                  loss="mse",
                  metrics=["mse", "mae"])


    # fit the custom generator
    history = model.fit_generator(generator=train_Generator,
                                  validation_data=test_Generator,
                                  use_multiprocessing=True,
                                  callbacks=calls,
                                  epochs=EPOCHS,
                                  workers=4,
                                  shuffle=True)
    return history



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

def storeTree(inputTree, dir, filename):
    """
    :param inputTree: the data object need to be written
    :param filename:  the saved filename
    :return:
    """
    if os.path.exists(dir):
        pass
    else:
        os.makedirs(dir)
    filename=dir+filename
    print(filename)
    fw = open(filename, 'wb') #以二进制读写方式打开文件
    pickle.dump(inputTree, fw)  #pickle.dump(对象, 文件，[使用协议])。序列化对象
    # 将要持久化的数据“对象”，保存到“文件”中，使用有3种，索引0为ASCII，1是旧式2进制，2是新式2进制协议，不同之处在于后者更高效一些。
    #默认的话dump方法使用0做协议
    fw.close() #关闭文件

def grabTree_full_address(full_address):
    """
    :param filename: the file data to be read
    :return:
    """

    fr = open(full_address, 'rb')
    return pickle.load(fr) #读取文件，反序列化