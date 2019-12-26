from tensorflow.keras.utils import Sequence
import os
import cv2
import glob
import math
from random import shuffle
import numpy as np

class DataGen(Sequence):
    def __init__(self, batch_size, input_image_path, image_size, name,vein_mask_path=None, tongue_mask_path=None, shuffle=True, mode="tongue_vein"):

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.mode = mode
        self.name =  name
        print(self.name, ":\n")
        self.input_image_path = input_image_path # e.g. "../Data/ALL_data/inputs/train/*"
        self.vein_mask_path = vein_mask_path
        self.tongue_mask_path = tongue_mask_path
        self.image_size = image_size


        # list images
        self.input_image_addrs = glob.glob(self.input_image_path)  # return a list of strings which are the absolute path of each images in the given folder
        # sort file paths
        self.input_image_addrs = self.sort_paths(self.input_image_addrs)

        # pair the input and label addrs
        self.pair_input_mask_addrs()
        self.num_batch =0
        self.on_epoch_end()

    def get_name(self, path):
        # os.path.basename(),返回path最后的文件名。若path以/或\结尾，那么就会返回空值。
        # os.path.splitext(),分离文件名与扩展名；默认返回(fname,fextension)元组
        name, _ = os.path.splitext(os.path.basename(path))
        # print(name)
        return name


    def sort_paths(self, paths):  # get the filename without suffix
        if all(self.get_name(path).isdigit() for path in paths):
            sorted_paths = sorted(paths, key=lambda path: int(self.get_name(path)))
        else:
            sorted_paths = sorted(paths)
        return sorted_paths

    def pair_input_mask_addrs(self):
        if self.mode == "tongue_vein":
            vein_mask_addrs = glob.glob(self.vein_mask_path)
            tongue_mask_addrs = glob.glob(self.tongue_mask_path)
            vein_mask_addrs = self.sort_paths(vein_mask_addrs)
            tongue_mask_addrs = self.sort_paths(tongue_mask_addrs)
            print("vein_mask_addrs：", vein_mask_addrs)
            print("tongue_mask_addrs：", tongue_mask_addrs)

            self.data_pairs = list(zip(self.input_image_addrs,  vein_mask_addrs, tongue_mask_addrs))  # zip. 打包 inputs and labels filename path
            shuffle(self.data_pairs)  # shuffle 打乱顺序

        elif self.mode == "tongue":
            tongue_mask_addrs = glob.glob(self.tongue_mask_path)
            tongue_mask_addrs = self.sort_paths(tongue_mask_addrs)
            print("tongue_mask_addrs：", tongue_mask_addrs)
            self.data_pairs = list(zip(self.input_image_addrs, tongue_mask_addrs))  # zip. 打包 inputs and labels filename path
            shuffle(self.data_pairs)  # shuffle 打乱顺序
        else:
            vein_mask_addrs = glob.glob(self.vein_mask_path)
            vein_mask_addrs = self.sort_paths(vein_mask_addrs)
            print("vein_mask_addrs：",vein_mask_addrs)
            self.data_pairs = list(
                zip(self.input_image_addrs, vein_mask_addrs))  # zip. 打包 inputs and labels filename path
            shuffle(self.data_pairs)  # shuffle 打乱顺序


    def loadData(self, idx):
        indexes = self.indexes[idx * self.batch_size:(idx + 1) * self.batch_size]

        print(self.name)
        # Find the batch list of records
        batch_list = [self.data_pairs[k] for k in indexes]
        # print("batch_list:", batch_list)

        if self.mode == "tongue_vein":
            X, Y, Z = [], [], []
            for sample in batch_list:
                # print("one sample:", sample)

                one_input_addr = sample[0]
                one_input_image = cv2.imread(one_input_addr, 1)  # read BGR color image without alpha
                one_vein_mask= sample[1]
                one_tongue_mask= sample[2]
                one_vein_mask = cv2.imread(one_vein_mask, 0)
                one_tongue_mask = cv2.imread(one_tongue_mask, 0)

                # resize image_size
                one_input_image = cv2.resize(one_input_image, (self.image_size, self.image_size))
                one_vein_mask = cv2.resize(one_vein_mask, (self.image_size, self.image_size))
                one_tongue_mask = cv2.resize(one_tongue_mask, (self.image_size, self.image_size))

                X.append(one_input_image)
                Y.append(one_vein_mask)
                Z.append(one_tongue_mask)
            return np.array(X), np.array(Y), np.array(Z)

        else:
            X, Y= [], []
            for sample in batch_list:
                # print("one sample:", sample)
                one_input_addr = sample[0]
                one_input_image = cv2.imread(one_input_addr, 1)  # read BGR color image without alpha
                one_mask = sample[1]
                one_mask = cv2.imread(one_mask, 0)

                # resize image_size and mask
                one_input_image = cv2.resize(one_input_image, (self.image_size, self.image_size))
                one_mask = cv2.resize(one_mask, (self.image_size, self.image_size))


                X.append(one_input_image)
                Y.append(one_mask)

        return np.array(X), np.array(Y)









    def __len__(self):
        return math.ceil(len(self.input_image_addrs) / self.batch_size)

    def __getitem__(self, idx):
        # indexes = self.indexes[idx * self.batch_size:(idx + 1) * self.batch_size]
        # self.num_batch +=1
        batch_data  = self.loadData(idx)
        # print("total batch:", self.num_batch)
        return batch_data

    def on_epoch_end(self):
        # after each epoch we neeed to renew the global entire self.indexes , not local index in getitem func. and shuffle
        # the global self.indexes so that we can each epoch to shuffle our entire dataset.
        self.indexes = np.arange(len(self.data_pairs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)