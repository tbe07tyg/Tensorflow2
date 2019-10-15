import os
import csv
from tensorflow.python.keras.utils.data_utils import Sequence
from tensorflow.python.keras.utils import to_categorical
import numpy as np
import random
import re
import glob
from numpy import math
import operator

class mySeqFeatureRegGenerator(Sequence):
    def __init__(self, batch_size, train_test, data_type, shuffle=True,
                 task_type="regression", seq_length=40, class_limit=None, image_shape=(224, 224, 3), sequence_path=None, csv_path_root=None):

        self.train_test= train_test
        self.batch_size = batch_size
        self.data_type = data_type
        self.task_type = task_type
        self.shuffle = shuffle
        # parameters
        self.seq_length = seq_length
        self.class_limit = class_limit
        if sequence_path == None:
            self.sequence_path = os.path.join('data', 'sequences')
        else:
            self.sequence_path = sequence_path
        self.max_frames = 300  # max number of frames a video can have for us to use it

        # Get the data  from the csv file. return a list of records
        self.data = self.get_data(csv_path_root)

        # Get the classes.  get the information from “self.data” get the classes from each record and append to class
        # variable and sorted classes. use self.class_limit to get the first "class_limit" number of the classes
        self.classes = self.get_classes()  # self.classes is a list contains first class_limit number of class names
        print(self.classes)
        # Now do some minor data cleaning.  # only use the data frames between  self.seq_length <=L<= self.max_frames
        self.data = self.clean_data()

        self.image_shape = image_shape

        train, test = self.split_train_test()
        self.used_data = train if self.train_test == 'train' else test
        print("Loading %d samples into memory for %sing." % (len(self.used_data), train_test))
        self.indexes = np.arange(len(self.used_data))


    def __len__(self):
        return math.ceil(len(self.used_data) / self.batch_size)

    def __getitem__(self, idx):
        # indexes = self.indexes[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_data  = self.frame_generator(idx)
        return batch_data

    # def on_epoch_end(self):
    #     # after each epoch we neeed to renew the global entire self.indexes , not local index in getitem func. and shuffle
    #     # the global self.indexes so that we can each epoch to shuffle our entire dataset.
    #     self.indexes = np.arange(len(self.used_data))
    #     if self.shuffle == True:
    #         np.random.shuffle(self.indexes)




    @staticmethod
    def get_data(csv_path_root):
        """Load our data from file."""
        if csv_path_root == None:
            with open(os.path.join("data", 'data_file.csv'), 'r') as fin:
                reader = csv.reader(fin)
                data = list(reader)
        else:
            with open(os.path.join(csv_path_root, 'data_file.csv'), 'r') as fin:
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

    def get_class_one_hot(self, class_str):
        """Given a class as a string, return its number in the classes
        list. This lets us encode and one-hot it for training."""
        # Encode it first.
        label_encoded = self.classes.index(class_str)

        # Now one-hot it.
        label_hot = to_categorical(label_encoded, len(self.classes))

        assert len(label_hot) == len(self.classes)

        return label_hot

    def split_train_test(self):
        """Split the data into train and test groups."""
        train = []
        test = []
        for item in self.data:
            if item[0] == 'train':
                train.append(item)
            else:
                test.append(item)
        return train, test

    def get_all_sequences_in_memory(self, train_test, data_type):
        """
        This is a mirror of our generator, but attempts to load everything into
        memory so we can train way faster.
        """
        # Get the right dataset.
        train, test = self.split_train_test()
        data = train if train_test == 'train' else test

        print("Loading %d samples into memory for %sing." % (len(data), train_test))

        X, y = [], []
        for row in data:

            if data_type == 'images':
                frames = self.get_frames_for_sample(row)
                frames = self.rescale_list(frames, self.seq_length)

                # Build the image sequence
                sequence = self.build_image_sequence(frames)

            else:
                sequence = self.get_extracted_sequence(data_type, row)

                if sequence is None:
                    print("Can't find sequence. Did you generate them?")
                    raise ValueError("Can't find sequence. Did you generate them?")

            X.append(sequence)
            print(X)



    def frame_generator(self, idx):
        """Return a generator that we can use to train on. There are
        a couple different things we can return:

        data_type: 'features', 'images'
        """

        # fetch the batch of data by indexes
        batch = self.used_data[idx * self.batch_size:(idx + 1) * self.batch_size]
        # print(len(batch))
        X, y = [], []
        count = 0
        for _ in batch:
            sample = random.choice(self.used_data)
            # print(count)
            if self.data_type is "images":
                # Get and resample frames.
                frames = self.get_frames_for_sample(sample)
                frames = self.rescale_list(frames, self.seq_length)

                # Build the image sequence
                sequence = self.build_image_sequence(frames)
            else:
                # Get the sequence from disk.
                sequence = self.get_extracted_sequence(self.data_type, sample)
                # print(sequence.shape)
                if sequence is None:
                    raise ValueError("Can't find sequence. Did you generate them?")

            X.append(sequence)
            # print("X length:", len(X))
            if self.task_type == "classification":
                y.append(self.get_class_one_hot(sample[1]))
                # print(y)
            elif self.task_type == "regression":
                # print("here")
                # print(sample[1])
                y.append(float(self.get_target_value_regression(sample[1])))
                # print(y)
            else:
                pass
            # print("X:", len(X))
            # count += 1
        # print(np.array(X).shape)
        return np.array(X), np.array(y)

    # def build_image_sequence(self, frames):
    #     """Given a set of frames (filenames), build our sequence."""
    #     return [process_image(x, self.image_shape) for x in frames]

    def get_extracted_sequence(self, data_type, sample):
        """Get the saved extracted features."""
        filename = sample[2]
        path = os.path.join(self.sequence_path, filename + '-' + str(self.seq_length) + \
            '-' + data_type + '.npy')
        if os.path.isfile(path):
            return np.load(path)
        else:
            return None

    def get_frames_by_filename(self, filename, data_type):
        """Given a filename for one of our samples, return the data
        the model needs to make predictions."""
        # First, find the sample row.
        sample = None
        for row in self.data:
            if row[2] == filename:
                sample = row
                break
        if sample is None:
            raise ValueError("Couldn't find sample: %s" % filename)

        if data_type == "images":
            # Get and resample frames.
            frames = self.get_frames_for_sample(sample)
            frames = self.rescale_list(frames, self.seq_length)
            # Build the image sequence
            sequence = self.build_image_sequence(frames)
        else:
            # Get the sequence from disk.
            sequence = self.get_extracted_sequence(data_type, sample)

            if sequence is None:
                raise ValueError("Can't find sequence. Did you generate them?")

        return sequence

    @staticmethod
    def get_frames_for_sample(sample):
        """Given a sample row from the data file, get all the corresponding frame
        filenames."""
        path = os.path.join('data', sample[0], sample[1])
        filename = sample[2]
        images = sorted(glob.glob(os.path.join(path, filename + '*jpg')))
        return images

    @staticmethod
    def get_filename_from_image(filename):
        parts = filename.split(os.path.sep)
        return parts[-1].replace('.jpg', '')

    @staticmethod
    def rescale_list(input_list, size):
        """Given a list and a size, return a rescaled/samples list. For example,
        if we want a list of size 5 and we have a list of size 25, return a new
        list of size five which is every 5th element of the origina list."""
        assert len(input_list) >= size

        # Get the number to skip between iterations.
        skip = len(input_list) // size

        # Build our new output.
        output = [input_list[i] for i in range(0, len(input_list), skip)]

        # Cut off the last one if needed.
        return output[:size]

    def print_class_from_prediction(self, predictions, nb_to_return=5):
        """Given a prediction, print the top classes."""
        # Get the prediction for each label.
        label_predictions = {}
        for i, label in enumerate(self.classes):
            label_predictions[label] = predictions[i]

        # Now sort them.
        sorted_lps = sorted(
            label_predictions.items(),
            key=operator.itemgetter(1),
            reverse=True
        )

        # And return the top N.
        for i, class_prediction in enumerate(sorted_lps):
            if i > nb_to_return - 1 or class_prediction[1] == 0.0:
                break
            print("%s: %.2f" % (class_prediction[0], class_prediction[1]))


    def get_target_value_regression(self, target_str):
        """
        Given a target breathing per minute string value which contains value. Thie extract number in the string only
        for regression training process
        :param target_str: bpm str
        :return:  number in target str
        """
        return re.findall(r"\d+\.?\d*", target_str)[0]

