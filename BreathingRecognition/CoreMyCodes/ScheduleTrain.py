from UiltilFunctions import schedule_train, get_data, split_train_test, storeTree
from ModelZoo import Lstm_signal_record_regression
import os
import matplotlib.pyplot as plt


if __name__ == '__main__':


    # hyper parramters
    class_limit = None  # int, can be 1-101 or None
    batch_size = 32

    lr = 2e-6
    model_type = 'lstm'
    seq_length = 40

    data_type = 'features'
    task_type = "classification"
    my_training_pairs_path =  "my_training_pairs.csv"

    tb_log_root = ".\\logs"
    ckpt_log_root = ".\\ckpt"

    # image shape
    feature_length =2048
    image_shape = (seq_length, feature_length)

    # get csv path
    data = get_data(my_training_pairs_path)  # get full data list
    # print("data：", data)
    print("data_length:", data.__len__())

    train_list, test_list = split_train_test(data)  # split train and test

    # initialize model
    model = Lstm_signal_record_regression(image_shape).model()

    EPOCHS = 100
    sd_tb_log_path = "..\\sheduler_tb_path"
    if not os.path.exists(sd_tb_log_path):
        print("build schedule_train_tensorboard folder")
        os.makedirs(sd_tb_log_path)
    history = schedule_train(model=model, EPOCHS=EPOCHS,
                             train_list=train_list, test_list= test_list,
                             sd_tb_log_path=sd_tb_log_path,
                             batch_size=batch_size)

    # save history in disk for letter use
    # save history
    tracking_saved_dir = "./IOU_tracking_Data/"  # "Directory for test data storeage"
    storeTree(history, tracking_saved_dir, "history")
    # # load history
    #         # history = joblib.load('history.pkl')

    plt.semilogx(history.history["lr"], history.history["loss"])
    plt.axis([1e-8, 1e-4, 0.2, 0.4])
    plt.savefig('schedule_lr.png')  #

