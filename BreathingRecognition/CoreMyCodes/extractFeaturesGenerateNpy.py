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

class DataSet():
    def __init__(self, seq_length=40, class_limit=None, image_shape=(224, 224, 3), sequence_path=None, csv_path_root=None, dataset_path_root=None):
        """Constructor.
        seq_length = (int) the number of frames to consider
        class_limit = (int) number of classes to limit the data to.
            None = no limit.
        """
        self.seq_length = seq_length
        self.class_limit = class_limit
        if sequence_path == None:
            self.sequence_path = os.path.join('data', 'sequences')
        else:
            self.sequence_path =  sequence_path
        self.dataset_path_root =  dataset_path_root
        self.max_frames = 300  # max number of frames a video can have for us to use it

        # Get the data  from the csv file. return a list of records
        self.data = self.get_data(csv_path_root)

        print(self.data)

        # Get the classes.  get the information from “self.data” get the classes from each record and append to class
        # variable and sorted classes. use self.class_limit to get the first "class_limit" number of the classes
        self.classes = self.get_classes()   # self.classes is a list contains first class_limit number of class names
        print(self.classes)
        # Now do some minor data cleaning.  # only use the data frames between  self.seq_length <=L<= self.max_frames
        self.data = self.clean_data()

        self.image_shape = image_shape

    @staticmethod
    def get_data(csv_path_root):
        """Load our data from file."""
        if csv_path_root == None:
            with open('my_data_file.csv', 'r') as fin:
                reader = csv.reader(fin)
                data = list(reader)
        else:
            with open('my_data_file.csv', 'r') as fin:
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

    def get_target_csv(self):
        pass


    def get_frames_for_sample(self, sample):
        """Given a sample row from the data file, get all the corresponding frame
        filenames."""
        path = os.path.join(self.dataset_path_root, sample[0], sample[1])
        filename = sample[2]
        images = sorted(glob.glob(os.path.join(path, filename + '*jpg')))
        return images

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
    for video in data.data:  # data.data is the cleaned data records
        print("video:", video)
        # Get the path to the sequence for this video. video is format by
        #   ["train or test", "class", "video filename", "the total number of frames generated"]
        path = os.path.join(data_root, 'sequences', video[2] + '-' + str(seq_length) + \
            '-features')  # numpy will auto-append .npy
        print("path:", path)

    #     # Check if we already have it.
    #     if os.path.isfile(path + '.npy'):
    #         pbar.update(1) # update the progress bar
    #         print("%s.npy already exsits" % path)
    #         continue
    #
        # Get the frames for this video.
        frames = data.get_frames_for_sample(video)  # return all frames(images) in one class folder according to the
        # print("frames:", frames)
        print("len of frames:", len(frames))


        # "video" content(from csv records)
    #
    #     # Now downsample to just the ones we need. # resample the frame list ==> down sample fps equivalently,
    #     # seq_length =  how many frames we want, skip = total frames/ seq_length
    #     frames = data.rescale_list(frames, seq_length)
    #
    #     # Now loop through and extract features to build the sequence.
    #     sequence = []
    #     for image in frames:
    #         features = model.extract(image)
    #         print("image:", image)
    #         print("features:", features)
    #         sequence.append(features)
    #
    #     # Save the sequence.
    #     np.save(path, sequence)
    #
    #     pbar.update(1)
    #
    # pbar.close()
