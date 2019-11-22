import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import fft,ifft
import matplotlib
def red_csv(full_path):
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
                          xytext=(x_value[-1] - 5, y_value[-1] -11),
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
        print("y_half shape", yf_half.shape)
        print("f_axis_half shape:", f_axis_half.shape)
        print("f_axis_half:", f_axis_half)
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
                            xytext=(f_axis_half[max_indx] + 0.3, yf_half[max_indx] + 0.3),
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

def EquallyDownSample(targetInput, dersiredN):
    """
    targetInput: the signal is going to be downsampled
    desiredN: The target length of down-sampled signal
    return: down-sampled signal
    """
    # assert len(targetInput) % dersiredN ==0
    skip = len(targetInput) // dersiredN
    print("skip:", skip)
    # Build our new output.
    output = [targetInput[i] for i in range(0, len(targetInput), skip)]
    return output





if __name__ == '__main__':
    # CSV path
    csv_root_path = "I:\\dataset\\BreathingData_16_29\\test_csv\\"
    csv_root_path2 = "I:\\dataset\\BreathingData_16_29\\train_csv\\"
    filename1 = "Rec20191108_043542_14.15.csv"
    filename2 = "Rec20191108_014327_21.13.csv"

    full_path_wave_time = os.path.join(csv_root_path, filename1)
    full_path_wave_time2 = os.path.join(csv_root_path2, filename2)

    data_list, time_list = red_csv(full_path_wave_time2)
    print("len of data:", len(data_list))
    print("len of time:", len(time_list))
    print("time:", time_list)
    print("data:", data_list)

    Td = 19.98 # 20 secs
    # bpm = 19.98
    bpm = 21.13
    fig, axes = plt.subplots(4, 4, figsize=(10, 4))
    plot_time_signal(y_value=data_list, x_value=time_list, Td= time_list[-1], axes=axes[0, 0], legendText=str(len(time_list))+" samples for " + str(bpm) +" bpm")
    plot_fft(y_value=data_list, Td=time_list[-1], axes_list=[axes[0, 1], axes[0, 2], axes[0, 3]], half_plot=True, zoom=True)

    # upsample to 32000 samples with linear interpolation
    x32000, y32000 = LinearUpsampleInterp(x=time_list, y=data_list, Td=19.98, desiredN=32000)
    print("time 3200:", x32000)
    print("y3200:", y32000)
    print("time 3200 shape:", x32000.shape)
    # plot upsampled signal
    plot_time_signal(y_value=y32000, x_value=x32000, Td=Td, axes=axes[1, 0], legendText=str(len(x32000))+" samples for " + str(bpm) +" bpm")
    plot_fft(y_value=y32000, Td=x32000[-1], axes_list=[axes[1, 1], axes[1, 2], axes[1, 3]], half_plot=True,
             zoom=True)

    # then downsample to desired len
    x320_32000 = EquallyDownSample(targetInput=x32000, dersiredN=320)
    y320_32000 = EquallyDownSample(targetInput=y32000, dersiredN=320)
    plot_time_signal(y_value=y320_32000, x_value=x320_32000, Td=Td, axes=axes[2, 0], legendText=str(len(x320_32000))+" samples from 32000 samples for " + str(bpm) +" bpm")
    plot_fft(y_value=y320_32000, Td=x320_32000[-1], axes_list=[axes[2, 1], axes[2, 2], axes[2, 3]], half_plot=True,
             zoom=True)


    # upsample to 3200 samples with linear interpolation
    x3200, y3200 = LinearUpsampleInterp(x=time_list, y=data_list, Td=Td, desiredN=3200)
    x320_3200 = EquallyDownSample(targetInput=x3200, dersiredN=320)
    y320_3200 = EquallyDownSample(targetInput=y3200, dersiredN=320)
    plot_time_signal(y_value=y320_3200, x_value=x320_3200, Td=Td, axes=axes[3, 0], legendText=str(len(x320_3200))+" samples from 3200 samples for " + str(bpm) +" bpm")
    plot_fft(y_value=y320_3200, Td=x320_3200[-1], axes_list=[axes[3, 1], axes[3, 2], axes[3, 3]], half_plot=True,
             zoom=True)
    print("x320:", x320_32000)
    print("y320", y320_32000)
    print("x320 len:", len(x320_32000))
    print("y320 len:", len(y320_32000))

    print("32000 x end:", x32000[-1])
    print("320_32000 x end:", x320_32000 [-1])
    print("320_3200 x end:", x320_3200[-1])
    print("x end:", time_list[-1])

    print("32000 x start:", x32000[0])
    print("320_32000 x start:", x320_32000[0])
    print("320_3200 x start:", x320_3200[0])
    print("x start:", time_list[0])




    plt.subplots_adjust(hspace=0.47)
    plt.show()