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
from scipy.signal import butter, lfilter, freqz
# CSV path
csv_root_path = "I:\\dataset\\BreathingData\\test_csv\\"
csv_root_path2 = "I:\\dataset\\BreathingData\\train_csv\\"
filename1="Rec20191108_043542_14.15.csv"
filename2="Rec20191108_014327_21.13.csv"
# filename2= "Rec20191001_124136vi.csv"


full_path_wave_time = os.path.join(csv_root_path, filename1)
full_path_wave_time2 = os.path.join(csv_root_path2, filename2)
# full_path_frame_time = os.path.join(csv_root_path, filename2)

fig, axes = plt.subplots(4, 4, figsize=(10, 4))

# print("full_path:", full_path_wave_time)
data_time_wave_raw = pd.read_csv(full_path_wave_time)
data_time_wave_raw2 = pd.read_csv(full_path_wave_time2)
print("data_time_wave_raw", data_time_wave_raw)
#
csv_values =  data_time_wave_raw.iloc[:, 1]
time_values = data_time_wave_raw.iloc[:, 0]

csv_values2 =  data_time_wave_raw2.iloc[:, 1]
time_values2 = data_time_wave_raw2.iloc[:, 0]


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
data_value2 = csv_values2.values
time_value2 =  time_values2.values

raw_mean =  np.mean(data_value)
raw_signal_no_mean = data_value - raw_mean
raw_mean2 =  np.mean(data_value2)
raw_signal_no_mean2 = data_value2 - raw_mean2
def rescale_list(input_list, size):
    """Given a list and a size, return a rescaled/samples list. For example,
    if we want a list of size 5 and we have a list of size 25, return a new
    list of size five which is every 5th element of the origina list."""
    assert len(input_list) >= size

    # Get the number to skip between iterations.
    skip = len(input_list) // (size)
    print("skip:", skip)

    # Build our new output.
    output = [input_list[i] for i in range(0, len(input_list), skip)]

    # Cut off the last one if needed.
    return output[:size]
    # return output
num_down_samples =  40
# down_sampled_value = rescale_list(raw_signal_no_mean, num_down_samples)
down_sampled_value = rescale_list(raw_signal_no_mean2, num_down_samples)
down_sampled_time = rescale_list(time_value, num_down_samples)

print("data_value", data_value, len(data_value))
print("down_sampled_value", down_sampled_value, len(down_sampled_value))
print("time_values", time_value, len(time_value))
print("down_sampled_time", down_sampled_time, len(down_sampled_time))

# down-sample csv
axes[3, 0].plot(down_sampled_time, down_sampled_value,'b', marker='*', label="down_sampled" +str(len(down_sampled_value)) +" samples")
print("down_sampled_value", down_sampled_value)
print("down_sampled_time", down_sampled_time)
print("down_sampled_time length:", len(down_sampled_time))
print("down_sampled_value length:", len(down_sampled_value))
# axes[3, 0].set_xlabel("time(s) for 14.15 bpm")
axes[3, 0].set_xlabel("time(s) for 21.13 bpm")
axes[3, 0].set_ylabel("removed DC resp_wave")
axes[3, 0].set_xlim([0, 20])
# axes[0, 0].set_ylim([500, 600])
axes[3, 0].legend(loc="best")
# fft

Td = 20
fs = num_down_samples/Td
d_N= len(down_sampled_value)
d_delta_f  = fs/d_N
d_f_axis =  np.arange(0, fs, d_delta_f)
# fft_out = fft(data_value, n=N)
# yf=abs(fft(data_value)) # 取绝对值
d_yf_norm=abs(fft(down_sampled_value))/d_N #归一化处理
print(d_yf_norm.shape)
axes[3, 1].plot(d_f_axis, d_yf_norm)
axes[3, 1].set_xlabel("frequency(Hz[0->fs])")
axes[3, 1].set_ylabel("magnitude(abs(fft(y))/N)")

axes[3, 1].set_xlim([0, fs])

d_f_axis_half = np.arange(0, fs//2, d_delta_f)
d_yf_half =  d_yf_norm[0: len(d_f_axis_half)]
print("d_y_half", d_yf_half.shape)
print("d_f_axis_half", d_f_axis_half.shape)
print(d_f_axis_half)
axes[3, 2].plot(d_f_axis_half, d_yf_half, marker = "+")
axes[3, 2].set_xlabel("frequency(Hz)[0->fs/2]")
axes[3, 2].set_ylabel("magnitude(abs(fft(y))/N)")

axes[3, 2].set_xlim([0, fs//2])
# axes[1].legend()

axes[3, 3].plot(d_f_axis_half, d_yf_half, marker = "+")
axes[3, 3].set_xlabel("zoom frequency(Hz)[0->2s]")
axes[3, 3].set_ylabel("magnitude(abs(fft(y))/N)")

axes[3, 3].set_xlim([0, 2])
max_indx = np.argmax(d_yf_half)
bpm= d_f_axis_half[max_indx]*60
show_max='['+str(round(d_f_axis_half[max_indx], 2))+'(' +str(round(bpm, 2))+'bpm)'+' '+str(round(d_yf_half[max_indx], 2))+']'
axes[3, 3].annotate(show_max, xy=(d_f_axis_half[max_indx], d_yf_half[max_indx]), xytext=(d_f_axis_half[max_indx]+0.3, d_yf_half[max_indx]+0.3),
             arrowprops=dict(facecolor='black', shrink=-0.05),
             )




# axes[0, 0].stem(time_value, data_value, 'b', markerfmt='bo', label='raw 1000 samples')
# axes[0, 0].stem(down_sampled_time, down_sampled_value, 'r', markerfmt='ro', label='320 samples')

axes[0, 0].plot(time_value, raw_signal_no_mean,'b', marker='*', label="raw_" +str(len(raw_signal_no_mean)) +" samples")
print("csv_values", data_value)
print("time_values", time_value)
print("time length:", len(time_value))
print("csv_value length:", len(csv_values))
axes[0, 0].set_xlabel("time(s) for 14.15 bpm")
axes[0, 0].set_ylabel("removed DC resp_wave")
axes[0, 0].set_xlim([0, 20])
# axes[0, 0].set_ylim([500, 600])
axes[0, 0].legend(loc="best")

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
axes[0, 1].set_ylabel("magnitude(abs(fft(y))/N)")

axes[0, 1].set_xlim([0, 50])

f_axis_half = np.arange(0, fs//2, delta_f)
yf_half =  yf_norm[0: len(f_axis_half)]
print("y_half", yf_half.shape)
print("f_axis_half", f_axis_half.shape)
print(f_axis_half)
axes[0, 2].plot(f_axis_half, yf_half, marker = "+")
axes[0, 2].set_xlabel("frequency(Hz)[0->fs/2]")
axes[0, 2].set_ylabel("magnitude(abs(fft(y))/N)")

axes[0, 2].set_xlim([0, fs//2])
# axes[1].legend()

axes[0, 3].plot(f_axis_half, yf_half, marker = "+")
axes[0, 3].set_xlabel("zoom frequency(Hz)[0->2s]")
axes[0, 3].set_ylabel("magnitude(abs(fft(y))/N)")

axes[0, 3].set_xlim([0, 2])
max_indx = np.argmax(yf_half)
bpm= f_axis_half[max_indx]*60
show_max='['+str(round(f_axis_half[max_indx], 2))+'(' +str(round(bpm, 2))+'bpm)'+' '+str(round(yf_half[max_indx], 2))+']'
axes[0, 3].annotate(show_max, xy=(f_axis_half[max_indx], yf_half[max_indx]), xytext=(f_axis_half[max_indx]+0.3, yf_half[max_indx]+0.3),
             arrowprops=dict(facecolor='black', shrink=-0.05),
             )

# example 2
axes[1, 0].plot(time_value2, raw_signal_no_mean2,'b', marker='*', label="raw 1000 samples")
print("csv_values2", data_value2)
print("time_values2", time_value2)
print("time length2:", len(time_value2))
print("csv_value length2", len(csv_values2))
axes[1, 0].set_xlabel("time(s) for 21.13 bpm")
axes[1, 0].set_ylabel("removed DC resp_wave")
axes[1, 0].set_xlim([0, 20])
# axes[0, 0].set_ylim([500, 600])
axes[1, 0].legend(loc="best")

# fft
fs = 50
N2= len(raw_signal_no_mean2)
delta_f2  = fs/N2
f_axis2 =  np.arange(0, fs, delta_f2)
# fft_out = fft(data_value, n=N)
# yf=abs(fft(data_value)) # 取绝对值
yf_norm2=abs(fft(raw_signal_no_mean2))/N2 #归一化处理
print(yf_norm2.shape)
axes[1, 1].plot(f_axis2, yf_norm2)
axes[1, 1].set_xlabel("frequency(Hz[0->fs])")
axes[1, 1].set_ylabel("magnitude(abs(fft(y))/N)")

axes[1, 1].set_xlim([0, 50])

f_axis_half2 = np.arange(0, fs//2, delta_f2)
yf_half2 =  yf_norm2[0: len(f_axis_half2)]
print("y_half2", yf_half2.shape)
print("f_axis_half2", f_axis_half2.shape)
print(f_axis_half2)
axes[1, 2].plot(f_axis_half2, yf_half2, marker = "+")
axes[1, 2].set_xlabel("frequency(Hz)[0->fs/2]")
axes[1, 2].set_ylabel("magnitude(abs(fft(y))/N)")

axes[1, 2].set_xlim([0, fs//2])
# axes[1].legend()

axes[1, 3].plot(f_axis_half2, yf_half2, marker = "+")
axes[1, 3].set_xlabel("zoom frequency(Hz)[0->2s]")
axes[1, 3].set_ylabel("magnitude(abs(fft(y))/N)")

axes[1, 3].set_xlim([0, 2])
max_indx2 = np.argmax(yf_half2)
bpm2= f_axis_half2[max_indx2]*60
show_max2='['+str(round(f_axis_half2[max_indx2],2))+'(' +str(round(bpm2,2))+'bpm)'+' '+str(round(yf_half2[max_indx2],2))+']'
axes[1, 3].annotate(show_max2, xy=(f_axis_half2[max_indx2], yf_half2[max_indx2]), xytext=(f_axis_half2[max_indx2]+0.3, yf_half2[max_indx2]+0.3),
             arrowprops=dict(facecolor='black', shrink=-0.05),
             )



# filtering------------------------------->
from scipy.signal import butter, lfilter

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


# Sample rate and desired cutoff frequencies (in Hz).
fs= 50
# lowcut1=0.01
# highcut1 = 1.5
order= 4 # for FIR filter
desired_f1 = f_axis_half[max_indx]
print("desired frequency:", desired_f1)
print("delta frequency:", delta_f)
lowcut1=desired_f1-2*delta_f
highcut1 = desired_f1+2*delta_f


from scipy.signal import freqz

for order in [3, 4, 5, 6]:
    b, a = butter_bandpass(lowcut1, highcut1, fs, order=order)
    w1, h1 = freqz(b, a, worN=N)
    axes[2, 3].plot((fs * 0.5 / np.pi) * w1, abs(h1), marker= "+", label="order = %d" % order)
axes[2, 3].set_xlim([0, 2])
axes[2, 3].set_xlabel("zoom Bandpass[0.1->1.5Hz] frequency(Hz)[0->2s]")
axes[2, 3].set_ylabel("Gain")
axes[2, 3].plot([0, 0.5 * fs], [np.sqrt(0.5), np.sqrt(0.5)],
             '--', label='sqrt(0.5)')
# filtered_signal1 = butter_bandpass_filter(raw_signal_no_mean, )
max_indx3 = np.argmax(abs(h1))
ff_h = abs(h1)
ff_axis =  (fs * 0.5 / np.pi) * w1
print("target bp:", ff_axis[max_indx3],ff_h[max_indx3])
# low_index =  ff_axis.index(lowcut1)
show_max3='['+str(round(ff_axis[max_indx3],2))+' '+str(round(ff_h[max_indx3],2))+']'
# axes[2, 3].annotate(show_max3, xy=(ff_axis[max_indx3], ff_h[max_indx3]), xytext=(ff_axis[max_indx3]+0.1, ff_h[max_indx3]+0.1),
#              arrowprops=dict(facecolor='black', shrink=-0.05),
#              )
axes[2, 3].legend()

axes[2, 2].plot(f_axis_half, yf_half, marker = "+")
axes[2, 2].set_xlabel("zoom frequency(Hz)[0->2s]")
axes[2, 2].set_ylabel("magnitude(abs(fft(y))/N)")

axes[2, 2].set_xlim([0, 2])


y = butter_bandpass_filter(data_value, lowcut1, highcut1, fs, order=4)
axes[2, 0].plot(time_value2, y ,'b', marker='*', label="raw 1000 samples")
print("y_values2", y)
print("time_values2", time_value2)
print("time length2:", len(time_value2))
print("csv_value length2", len(y))
axes[2, 0].set_xlabel("time(s) for 21.13 bpm")
axes[2, 0].set_ylabel("removed DC resp_wave")
axes[2, 0].set_xlim([0, 20])
# axes[0, 0].set_ylim([500, 600])
axes[2, 0].legend(loc="best")

# fft
fs = 50
N2= len(y)
delta_f2  = fs/N2
f_axis2 =  np.arange(0, fs, delta_f2)
# fft_out = fft(data_value, n=N)
# yf=abs(fft(data_value)) # 取绝对值
yff_norm2=abs(fft(y))/N2 #归一化处理
print(yf_norm2.shape)
axes[2, 1].plot(f_axis2, yff_norm2, marker="+")
axes[2, 1].set_xlabel("frequency(Hz[0->fs])")
axes[2, 1].set_ylabel("magnitude(abs(fft(y))/N)")
axes[2, 1].set_xlim([0, 2])








print("data_value 14.15:", len(data_value))
print("data_sample_value 14.15", len(down_sampled_value))

plt.subplots_adjust(hspace=0.47)
plt.show()