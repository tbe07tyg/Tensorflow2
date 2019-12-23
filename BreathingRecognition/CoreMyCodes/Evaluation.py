import os

import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt
from scipy.fftpack import fft,ifft
import re
from sklearn.preprocessing import StandardScaler
from pandas import Series
from math import sqrt
import tensorflow as tf
from ModelZoo import LstmReg, Lstm, Lstm_signal_record_regression

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

def read_csv(full_path):
    """
    full path: full_path of csv file and
    return: raw_signal_no_mean (0 mean ) and time values
    """
    data_time_wave_raw = pd.read_csv(full_path)
    csv_values = data_time_wave_raw.iloc[:, 1]
    time_values = data_time_wave_raw.iloc[:, 0]
    data_value = csv_values.values
    raw_mean = np.mean(data_value)
    raw_signal_no_mean = data_value - raw_mean
    time_value = time_values.values
    return raw_signal_no_mean, time_value

def plot_time_signal(y_value, x_value, Td, axes, legendText):
    """
    y_value: y magnitude with respect to x_value
    x_value: time axis in time domain
    Td: total time period in secs
    axes: array axes to make plot
    """
    # axes.plot(x_value, y_value, 'b', marker='*',
    #                 label= legendText)
    axes.plot(x_value, y_value, 'b', marker='*')
    # axes[3, 0].set_xlabel("time(s) for 14.15 bpm")
    axes.set_xlabel("time(s)")
    axes.set_ylabel("removed DC resp_wave")
    axes.set_xlim([0, Td])
    show_end = '[' + str(round(x_value[-1],3)) + ", " + str(round(y_value[-1], 3)) + ']'
    axes.annotate(show_end, xy=(x_value[-1], y_value[-1]),
                          xytext=(x_value[-1] - 2.5, y_value[-1] -2.5),
                          arrowprops=dict(facecolor='black', shrink=-0.05),
                          )
    axes.set_title(legendText)
    # axes.legend(loc="best")

def plot_fft(y_value, Td, axes_list, half_plot=True, zoom=True):
    """
    y_value: y magnitude with respect to x_value
    x_value: time axis in time domain
    Td: total time period in secs
    axes: array axes to make plot
    """
    N = len(y_value)
    fs =  N/Td
    delta_f = fs / N
    f_axis = np.arange(0, fs, delta_f)
    yf_norm = abs(fft(y_value)) / N  # 归一化处理
    axes_list[0].plot(f_axis, yf_norm)
    axes_list[0].set_xlabel("frequency(Hz[0->fs])")
    axes_list[0].set_ylabel("magnitude(abs(fft(y))/N)")
    axes_list[0].set_xlim([0, fs])

    if half_plot==True:
        f_axis_half = np.arange(0, fs // 2, delta_f)
        yf_half = yf_norm[0: len(f_axis_half)]
        # print("y_half shape", yf_half.shape)
        # print("f_axis_half shape:", f_axis_half.shape)
        # print("f_axis_half:", f_axis_half)
        axes_list[1].plot(f_axis_half, yf_half, marker="+")
        axes_list[1].set_xlabel("frequency(Hz)[0->fs/2]")
        axes_list[1].set_ylabel("magnitude(abs(fft(y))/N)")
        axes_list[1].set_xlim([0, fs // 2])

    if zoom == True:
        f_axis_half = np.arange(0, fs // 2, delta_f)
        yf_half = yf_norm[0: len(f_axis_half)]
        axes_list[2].plot(f_axis_half, yf_half, marker="+")
        axes_list[2].set_xlabel("zoom frequency(Hz)[0->2s]")
        axes_list[2].set_ylabel("magnitude(abs(fft(y))/N)")
        axes_list[2].set_xlim([0, 2])
        max_indx = np.argmax(yf_half)
        bpm = round(f_axis_half[max_indx],4) * 60
        show_max = '[' + str(round(f_axis_half[max_indx], 4)) + '(' + str(round(bpm, 4)) + 'bpm)' + ' ' + str(
            round(yf_half[max_indx], 4)) + ']'
        axes_list[2].annotate(show_max, xy=(f_axis_half[max_indx], yf_half[max_indx]),
                            xytext=(f_axis_half[max_indx] + 0.3, yf_half[max_indx] + 0.025),
                            arrowprops=dict(facecolor='black', shrink=-0.05),
                            )

def LinearUpsampleInterp(x, y, Td, desiredN):
    """
    (x, y) is used for define the original cordinates
    """
    from scipy.interpolate import interp1d
    f = interp1d(x, y) # define x, y pair object
    print(x[-1])
    assert  Td == x[-1]
    # define xnew for new upsampled x
    xnew = np.linspace(0, Td, num=desiredN, endpoint=True)

    return xnew, f(xnew)

def UP_Down_interp1d(x, y, Td, desiredN):
    """
    (x, y) is used for define the original cordinates
    """
    from scipy.interpolate import interp1d
    f = interp1d(x, y)  # define x, y pair object
    print(x[-1])
    assert Td == x[-1]
    # define xnew for new upsampled x
    xnew = np.linspace(0, Td, num=desiredN, endpoint=True)

    return xnew, f(xnew)

def get_target_value_regression(target_str):
    """
    Given a target breathing per minute string value which contains value. Thie extract number in the string only
    for regression training process
    :param target_str: bpm str
    :return:  number in target str
    """
    return re.findall(r"\d+\.?\d*", target_str)[0]

def standardization(x):
    """
    mean = sum(x) / count(x)
    standard_deviation = sqrt( sum( (x - mean)^2 ) / count(x))
    y = (x - mean) / standard_deviation

    return y


    """
    # define contrived series
    series =  Series(x)
    print("prepare series for standardization:", series)
    values =  series.values
    values =  values.reshape((len(values),1))
    # train the normalization
    scaler = StandardScaler()
    scaler = scaler.fit(values)
    print('Mean: %f, StandardDeviation: %f' % (scaler.mean_, sqrt(scaler.var_)))
    # normalize the dataset and print
    standardized = scaler.transform(values)
    print("initial standized shape", standardized.shape)
    #
    standardized = standardized.reshape(1, standardized.shape[0]).tolist()[0]
    print(standardized)

    return standardized

def get_extracted_batch_sequence(self, idx, class_names=None):
    """Get the saved extracted features."""
    # fetch the batch of data by indexes
    indexes = self.indexes[idx * self.batch_size:(idx + 1) * self.batch_size]

    # Find the batch list of records
    batch_list = [self.used_data[k] for k in indexes]

    batch_x_list = []
    batch_y_list = []
    # print("batch_list", batch_list)
    # print("batch_list len:", len(batch_list))
    self.counter +=1
    print(self.counter)
    for sample in batch_list:
        path = sample[2] + '.npy'
        # print(path)
        if os.path.isfile(path):
            batch_x_list.append(np.load(path))
        else:
            raise Exception("please check the npy path")

        if not self.classNames == None:
            batch_y_list.append(self.get_class_one_hot(class_names=class_names, class_str=sample[1]))
            # print(y)
        else:
            # print("here")
            # print(sample[1])
            batch_y_list.append(float(self.get_target_value_regression(sample[1])))
            # print(y)

        # print("batch_x_list",batch_x_list, len(batch_x_list))
        # print("batch_y_list",batch_y_list, len(batch_y_list))
    # print("batch_x_len",len(batch_x_list))
    # print("batch_y_len", len(batch_y_list))
    print(np.array(batch_x_list).shape)
    print(np.array(batch_y_list).shape)
    # print(np.array(batch_x_list))
    return np.array(batch_x_list), np.array(batch_y_list)



def predict_for_test(test_list, model):
    predictions_list = []
    count =0
    for sample in test_list:
        path = sample[2] + '.npy'
        features = np.load(path)
        # print("test features", features.shape)
        features =  np.expand_dims(features, axis=0)
        # print("expand input shape:", features.shape)
        prediction = model(features,training=False)
        count+=1

        predictions_list.append(prediction.numpy()[0][0])
        print("predicted num:", count)
        print("prediction：", prediction)
        print("prediction.numpy():", prediction.numpy())
    return predictions_list

if __name__ == '__main__':
    my_training_pairs_path = "my_training_pairs.csv"

    data = get_data(my_training_pairs_path)  # get full data list

    # # CSV path
    csv_root_path = "I:\\dataset\\BreathingData_16_29\\test_csv\\"
    # csv_root_path2 = "I:\\dataset\\BreathingData_16_29\\train_csv\\"
    filename1 = "Rec20191108_043542_14.15.csv"
    filename_no_suffix = filename1.split(os.path.sep)
    print("filename_no_suffix:", filename_no_suffix)
    full_path_wave_time = os.path.join(csv_root_path, filename1)
    raw_signal_no_mean, time_value = read_csv(full_path_wave_time)
    raw_signal_no_mean = standardization(raw_signal_no_mean)
    print("time_list csv:", time_value)

    #
    # data_list, time_list = read_csv(full_path_wave_time2)
    train_list, test_list = split_train_test(data)  # split train and test
    print("We have training samples:", len(train_list))
    print("We have test samples:", len(test_list))

    # visualize
    initial_40 =  [0] * 40
    full_target_curve = initial_40.copy()
    print("initial_40:", initial_40)
    print("visualize test samples")
    for sample in test_list:
        print("sample:", sample)
        if "Rec20191108_043542_14.15" in sample[2]:
            print("find target csv")
            csv_target =  sample[1]
            full_target_curve.append(float(csv_target)) # change str values into float
        else:
            print("do not find target csv")
    print("full_generated_target_curve:", full_target_curve)

    print("raw csv len:", len(time_value))
    print("full_generated_target_curve len:", len(full_target_curve))

    # up-down sampling
    desiredN = 320
    time320, y320 = UP_Down_interp1d(x=time_value, y=raw_signal_no_mean, Td=19.98, desiredN=desiredN)
    y320 = standardization(y320)

    # my model predictions:
    # image shape
    seq_length = 40
    feature_length = 2048
    image_shape = (seq_length, feature_length)
    model = Lstm_signal_record_regression(image_shape).model()
    ckpt_log_root = "E:\\logs\\breathing_project\\20191223\\ckpt"
    optimizer = tf.keras.optimizers.Adam(lr= 5e-6)
    ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, net=model)
    manager = tf.train.CheckpointManager(ckpt, ckpt_log_root, max_to_keep=3)
    # restore the model
    ckpt.restore(manager.latest_checkpoint)

    predicts = predict_for_test(test_list, model)

    full_predict_for_test_list =  initial_40 + predicts
    print("we have predicted # of samples:", len(initial_40))
    print("we have padded initial # of zeros:", len(predicts))
    print("predicts:", predicts)
    print("full_predict_for_test_list", full_predict_for_test_list)
    print("len of full_predict_for_test_list", len(full_predict_for_test_list))


    # plot compare:
    bpm = 14.15
    fig, axes = plt.subplots(4, 4, figsize=(10, 4))
    plot_time_signal(y_value=raw_signal_no_mean, x_value=time_value, Td=time_value[-1], axes=axes[0, 0],
                     legendText="raw " + str(len(time_value)) + " samples for " + str(bpm) + " bpm")
    plot_fft(y_value=raw_signal_no_mean, Td=time_value[-1], axes_list=[axes[0, 1], axes[0, 2], axes[0, 3]], half_plot=True,
             zoom=True)

    # down-sample plot
    plot_time_signal(y_value=y320, x_value=time320, Td=time320[-1], axes=axes[1, 0],
                     legendText=str(len(time320)) + " down sampled for " + str(bpm) + " bpm")
    plot_fft(y_value=y320, Td=time320[-1], axes_list=[axes[1, 1], axes[1, 2], axes[1, 3]], half_plot=True,
             zoom=True)


    # plot catched target csv values
    plot_time_signal(y_value=full_predict_for_test_list, x_value=time320, Td=time320[-1], axes=axes[3, 0],
                     legendText=str(len(time320)) + " target values for " + str(bpm) + " bpm")
    plot_fft(y_value=full_predict_for_test_list, Td=time320[-1], axes_list=[axes[3, 1], axes[3, 2], axes[3, 3]], half_plot=True,
             zoom=True)

    # plot predicted csv values
    plot_time_signal(y_value=full_target_curve, x_value=time320, Td=time320[-1], axes=axes[2, 0],
                     legendText=str(len(time320)) + " target values for " + str(bpm) + " bpm")
    plot_fft(y_value=full_target_curve, Td=time320[-1], axes_list=[axes[2, 1], axes[2, 2], axes[2, 3]], half_plot=True,
             zoom=True)

    print("y320 last：", y320[-1])
    print("full target last:", full_target_curve[-1])

    plt.subplots_adjust(hspace=0.47)
    plt.show()
