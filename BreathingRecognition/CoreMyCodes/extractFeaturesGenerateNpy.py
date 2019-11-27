"""
This script generates extracted features for each video, which other
models make use of.

You can change you sequence length and limit to a set number of classes
below.

class_limit is an integer that denotes the first N classes you want to
extract features from. This is useful is you don't want to wait to
extract all 101 classes. For instance, set class_limit = 8 to just
extract features for the first 8 (alphabetical) classes in the dataset.
Then set the same number when training models.
"""
import numpy as np
import os.path
from extractor import Extractor
from tqdm import tqdm
import csv
import glob
import pandas as pd

class DataSet():
    def __init__(self, seq_length=40, class_limit=None, image_shape=(224, 224, 3), sequence_path=None,
                 csv_path_root=None, dataset_path_root=None, csv_signal_full_path=None):
        """Constructor.
        seq_length = (int) the number of frames to consider
        class_limit = (int) number of classes to limit the data to.
            None = no limit.
        """
        self.csv_signal_full_path = csv_signal_full_path
        self.dataset_path_root =  dataset_path_root
        self.seq_length = seq_length
        self.class_limit = class_limit
        if sequence_path == None:
            self.sequence_path = os.path.join(self.dataset_path_root, 'sequences')
        else:
            self.sequence_path =  sequence_path

        self.max_frames = 300  # max number of frames a video can have for us to use it

        # Get the data  from the csv file. return a list of records
        self.data = self.get_data(csv_path_root)

        print(self.data)

        # Get the classes.  get the information from “self.data” get the classes from each record and append to class
        # variable and sorted classes. use self.class_limit to get the first "class_limit" number of the classes
        self.classes = self.get_classes()   # self.classes is a list contains first class_limit number of class names
        print(self.classes)
        # Now do some minor data cleaning.  # only use the data frames between  self.seq_length <=L<= self.max_frames
        # self.data = self.clean_data()

        self.image_shape = image_shape

    @staticmethod
    def get_data(csv_path_root):
        """Load our data from file."""
        if csv_path_root == None:
            with open('my_data_file.csv', 'r') as fin:
                reader = csv.reader(fin)
                data = list(reader)
        else:
            with open(os.path.join(csv_path_root, 'my_data_file.csv'), 'r') as fin:
                reader = csv.reader(fin)
                data = list(reader)

        return data

    def clean_data(self):
        """Limit samples to greater than the sequence length and fewer
        than N frames. Also limit it to classes we want to use."""
        data_clean = []
        for item in self.data:
            if int(item[3]) >= self.seq_length and int(item[3]) <= self.max_frames \
                    and item[1] in self.classes:
                data_clean.append(item)

        return data_clean

    def get_classes(self):
        """Extract the classes from our data. If we want to limit them,
        only return the classes we need."""
        classes = []
        for item in self.data:
            if item[1] not in classes:
                classes.append(item[1])

        # Sort them.
        classes = sorted(classes)

        # Return.
        if self.class_limit is not None:
            return classes[:self.class_limit]
        else:
            return classes




    def get_frames_for_sample(self, sample):
        """Given a sample row from the data file, get all the corresponding frame
        filenames."""
        path = os.path.join(self.dataset_path_root, sample[0], sample[1])
        filename = sample[2]
        images = sorted(glob.glob(os.path.join(path, filename + '*jpg')))
        return images

    @staticmethod
    def read_csv(csv_signal_full_path):
        """
        full path: full_path of csv file and
        return: raw_signal_no_mean (0 mean ) and time values
        """
        data_time_wave_raw = pd.read_csv(csv_signal_full_path)
        csv_values = data_time_wave_raw.iloc[:, 1]
        time_values = data_time_wave_raw.iloc[:, 0]
        data_value = csv_values.values
        raw_mean = np.mean(data_value)
        raw_signal_no_mean = data_value - raw_mean
        time_value = time_values.values
        return raw_signal_no_mean, time_value

    @staticmethod
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


if __name__ == '__main__':

    # dataset root
    data_root = "I:\\dataset\\BreathingData_16_29\\"
    # Set defaults.
    seq_length = 40
    class_limit = None  # Number of classes to extract. Can be 1-101 or None for all.

    # Get the dataset.
    data = DataSet(seq_length=seq_length, class_limit=class_limit, dataset_path_root=data_root) # get list of records from csv with the conditions: seq_length, class_limit, max_frames

    # get the model. # initialize the pre-trained model feature extractor load weights return the model object
    model = Extractor()
    print("data length:", len(data.data))
    # Loop through data.
    pbar = tqdm(total=len(data.data))  # 生成一个迭代器进度条  tqdm(iterator) create a progress bar for loop

    video_count = 0
    print("total num of videos:", len(data.data))

    # each npy file + 1 csv next value -------> one training sample
    training_pair_file =[] # total number of  training samples
    Td = 19.98 # for downsampling
    desiredN = 320

    # seuqnece data root
    sequence_data_saved_root = os.path.join(data_root, 'sequences')

    if os.path.exists(sequence_data_saved_root):
        pass
    else:
        os.makedirs(sequence_data_saved_root)

    for video in data.data:  # data.data is the cleaned data records



        video_count+=1
        print("video:", video)
        print("video count=:", video_count)

        train_csv_folderName =  video[0]+ "_csv"
        csv_filename =  video[2] + ".csv"
        full_path_wave_time = os.path.join(data_root, train_csv_folderName, csv_filename)
        print("full_path_wave_time", full_path_wave_time)

        # read csv breathing records
        raw_signal_no_mean, time_value = data.read_csv(full_path_wave_time)
        print("raw_signal_no_mean len:", raw_signal_no_mean.shape)
        print("time value len:", time_value.shape)
        # down sample the csv records

        time320, y320 = data.UP_Down_interp1d(x=time_value, y=raw_signal_no_mean, Td=19.98, desiredN=desiredN)
        print("time 320 len:", time320.shape)
        print("y320 len:", y320.shape)


    #
        # Get the frames for this video.
        frames = data.get_frames_for_sample(video)  # return all frames(images) in one class folder according to the
        # print("frames:", frames)
        print("len of frames:", len(frames))
        # print(frames)

        sequence = []  # for restore all the frame feature of one video
        # generate 40 frames and next value of csv records in time to be the one training sample:
        for image in frames:
            features = model.extract(image)
            # print("image:", image)
            # print("features:", features.shape)
            sequence.append(features)
        print("sequence len", len(sequence))

        # Get the path to the sequence for this video. video is format by
        #   ["train or test", "class", "video filename", "the total number of frames generated"]


        for i in range(data.seq_length, len(sequence)):
            one_npy_path = os.path.join(data_root, 'sequences', video[2] + '-' + str(seq_length) + \
                                '-features'+"_" + str(i-seq_length) + "-" + str(i))  # numpy will auto-append .npy
            print("npy path:", one_npy_path)


            one_training_features = sequence[i-seq_length:i]
            one_training_target =  y320[i]
            print("one_training_features", len(one_training_features))

            # training_pair_file contains rows of  each training pair as [train or test, target y]
            training_pair_file.append([video[0], str(one_training_target), one_npy_path, len(sequence)])

            # Check if we already have it.
            if os.path.isfile(one_npy_path + '.npy'):
                print("%s.npy already exsits" % one_npy_path)
                continue

            # Save the sequence.
            np.save(one_npy_path, one_training_features)

        pbar.update(1)
    if not os.path.exists('my_training_pairs.csv'):
        with open('my_training_pairs.csv', 'w', newline="") as fout:
            writer = csv.writer(fout)
            writer.writerows(training_pair_file)
    else:
        os.remove('my_training_pairs.csv')
        with open('my_training_pairs.csv', 'w', newline="") as fout:
            writer = csv.writer(fout)
            writer.writerows(training_pair_file)
    pbar.close()
