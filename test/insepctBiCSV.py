import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import fft,ifft
import seaborn
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from pandas.plotting import autocorrelation_plot
# CSV path
csv_root_path = "I:\\dataset\\BreathingData\\test_csv\\"
filename1="Rec20191108_043542_14.15.csv"
# filename2= "Rec20191001_124136vi.csv"


full_path_wave_time = os.path.join(csv_root_path, filename1)

# full_path_frame_time = os.path.join(csv_root_path, filename2)

fig, axes = plt.subplots(2, 4, figsize=(10, 4))

# print("full_path:", full_path_wave_time)
data_time_wave_raw = pd.read_csv(full_path_wave_time)
print("data_time_wave_raw", data_time_wave_raw)
#
csv_values =  data_time_wave_raw.iloc[:, 1]
time_values = data_time_wave_raw.iloc[:, 0]




# data_frame_time = pd.read_csv(full_path_frame_time, usecols=[0, 1])
# print(data_frame_time)
# data_frame_time["time(ms)"] = data_frame_time["time(ms)"]/1000
# fps = int(data_frame_time["count"].values[-1]/data_frame_time["time(ms)"].values[-1])
# print(data_frame_time)
# print(fps)
# axes[1].plot(data_frame_time["time(ms)"], data_frame_time["count"], label = " fps: %s" %(fps))
# axes[1].set_xlabel("count")
# axes[1].set_ylabel("time(s)")
# axes[1].legend()
# data =  data_time_wave.loc[:,'resp_wave'].rolling(1).apply(lambda x: x.autocorr(), raw=False)
# data =  data_time_wave.loc[:, 'resp_wave']
# time = data_time_wave.iloc[:, 0]
# print("data:", data.values)
# print("time:", time.values)

# down sampling data
data_value = csv_values.values
time_value =  time_values.values
raw_mean =  np.mean(data_value)
raw_signal_no_mean =  data_value - raw_mean
def rescale_list(input_list, size):
    """Given a list and a size, return a rescaled/samples list. For example,
    if we want a list of size 5 and we have a list of size 25, return a new
    list of size five which is every 5th element of the origina list."""
    assert len(input_list) >= size

    # Get the number to skip between iterations.
    skip = len(input_list) // (size-1)
    print("skip:", skip)

    # Build our new output.
    output = [input_list[i] for i in range(0, len(input_list), skip)]

    # Cut off the last one if needed.
    return output[:size]
    # return output
num_down_samples =  320
down_sampled_value = rescale_list(raw_signal_no_mean, num_down_samples)
down_sampled_time = rescale_list(time_value, num_down_samples)

print("data_value", data_value, len(data_value))
print("down_sampled_value", down_sampled_value, len(down_sampled_value))
print("time_values", time_value, len(time_value))
print("down_sampled_time", down_sampled_time, len(down_sampled_time))


# axes[0, 0].stem(time_value, data_value, 'b', markerfmt='bo', label='raw 1000 samples')
# axes[0, 0].stem(down_sampled_time, down_sampled_value, 'r', markerfmt='ro', label='320 samples')

axes[0, 0].plot(time_value, raw_signal_no_mean,'b', marker='*', label="raw 1000 samples")
print("csv_values", data_value)
print("time_values", time_value)
print("time length:", len(time_value))
print("csv_value length:", len(csv_values))
axes[0, 0].set_xlabel("time(ms)")
axes[0, 0].set_ylabel("resp_wave")
axes[0, 0].set_xlim([0, 20])
# axes[0, 0].set_ylim([500, 600])
axes[0, 0].legend(loc="best")


axes[1, 0].plot(down_sampled_time, down_sampled_value, 'r', marker="*", label=str(num_down_samples)+ " samples")
# axes[1, 0].plot(down_sampled_time, down_sampled_value)
axes[1, 0].set_xlabel("time(ms)")
axes[1, 0].set_ylabel("resp_wave")
axes[1, 0].set_xlim([0, 20])
axes[1, 0].legend(loc="best")
# axes[1, 0].set_ylim([500,600])
# Draw Plot
# ax = plt.gca(xlim=(1, 500), ylim=(-1.0, 1.0))
# plt.rcParams.update({'figure.figsize':(9,5), 'figure.dpi':120})
# autocorrelation_plot(data.values.tolist())
# acf_500 = acf(data.values, nlags=499)


# autocorrelation calculation
# plot_acf(csv_values.values, lags=999, ax=axes[0, 1])

# fft
fs = 50
N= len(raw_signal_no_mean)
delta_f  = fs/N
f_axis =  np.arange(0, fs, delta_f)
# fft_out = fft(data_value, n=N)
# yf=abs(fft(data_value)) # 取绝对值
yf_norm=abs(fft(raw_signal_no_mean))/N #归一化处理
print(yf_norm.shape)
axes[0, 1].plot(f_axis, yf_norm)
axes[0, 1].set_xlabel("frequency(Hz[0->fs])")
axes[0, 1].set_ylabel("magnitude")
axes[0, 1].set_ylabel("fft")
axes[0, 1].set_xlim([0, 50])

f_axis_half = np.arange(0, fs//2, delta_f)
yf_half =  yf_norm[0: len(f_axis_half)]
print("y_half", yf_half.shape)
print("f_axis_half", f_axis_half.shape)
print(f_axis_half)
axes[0, 2].plot(f_axis_half, yf_half, marker = "+")
axes[0, 2].set_xlabel("frequency(Hz)[0->fs/2]")
axes[0, 2].set_ylabel("magnitude")
axes[0, 2].set_ylabel("fft")
axes[0, 2].set_xlim([0, fs//2])
# axes[1].legend()

axes[0, 3].plot(f_axis_half, yf_half, marker = "+")
axes[0, 3].set_xlabel("frequency(Hz)[0->2s]")
axes[0, 3].set_ylabel("magnitude")
axes[0, 3].set_ylabel("fft")
axes[0, 3].set_xlim([0, 2])
max_indx = np.argmax(yf_half)
bpm= f_axis_half[max_indx]*60
show_max='['+str(f_axis_half[max_indx])+ str((bpm))+' '+str(yf_half[max_indx])+']'
axes[0, 3].annotate(show_max, xy=(f_axis_half[max_indx], yf_half[max_indx]), xytext=(f_axis_half[max_indx]+0.3, yf_half[max_indx]+0.3),
             arrowprops=dict(facecolor='black', shrink=-0.05),
             )

plt.show()