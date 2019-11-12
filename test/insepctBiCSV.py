import os
import pandas as pd
import matplotlib.pyplot as plt

# CSV path
csv_root_path = "I:\\dataset\\BreathingData\\train_csv\\"
filename1="Rec20191106_104636_15.38.csv"
# filename2= "Rec20191001_124136vi.csv"


full_path_wave_time = os.path.join(csv_root_path, filename1)

# full_path_frame_time = os.path.join(csv_root_path, filename2)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# print("full_path:", full_path_wave_time)
#
data_time_wave = pd.read_csv(full_path_wave_time, usecols=[0, 1], index_col=0)

print(data_time_wave)
print(data_time_wave)

axes[0].plot(data_time_wave)
axes[0].set_xlabel("time(ms)")
axes[0].set_ylabel("resp_wave")


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
#
plt.show()