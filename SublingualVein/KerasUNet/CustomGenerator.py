from tensorflow.keras.utils import Sequence
import os
import cv2
import glob
import math
from random import shuffle

class DataGen(Sequence):
    def __init__(self, batch_size, input_image_path, vein_mask_path=None, tongue_mask_path=None, shuffle=True, mode="tongue_vein"):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.mode = mode
        self.on_epoch_end()
        self.input_image_path = input_image_path # e.g. "../Data/ALL_data/inputs/train/*"
        self.vein_mask_path = vein_mask_path
        self.tongue_mask_path = tongue_mask_path

        # list images
        self.input_image_addrs = glob.glob(self.input_image_path)  # return a list of strings which are the absolute path of each images in the given folder
        # sort file paths
        self.input_image_addrs = self.sort_paths(self.input_image_addrs)

        # pair the input and label addrs
        self.pair_input_mask_addrs()

    def get_name(self, path):
        # os.path.basename(),返回path最后的文件名。若path以/或\结尾，那么就会返回空值。
        # os.path.splitext(),分离文件名与扩展名；默认返回(fname,fextension)元组
        name, _ = os.path.splitext(os.path.basename(path))
        print(name)
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

            self.data_pairs = list(zip(self.input_image_addrs, self.tongue_mask_addrs, self.vein_mask_addrs))  # zip. 打包 inputs and labels filename path
            shuffle(self.data_pairs)  # shuffle 打乱顺序

        elif self.mode == "tongue":
            tongue_mask_addrs = glob.glob(self.tongue_mask_path)
            tongue_mask_addrs = self.sort_paths(tongue_mask_addrs)
            print("tongue_mask_addrs：", tongue_mask_addrs)
            self.data_pairs = list(zip(self.input_image_addrs, self.tongue_mask_addrs))  # zip. 打包 inputs and labels filename path
            shuffle(self.data_pairs)  # shuffle 打乱顺序
        else:
            vein_mask_addrs = glob.glob(self.vein_mask_path)
            vein_mask_addrs = self.sort_paths(vein_mask_addrs)
            print("vein_mask_addrs：",vein_mask_addrs)
            self.data_pairs = list(
                zip(self.input_image_addrs, self.vein_mask_addrs))  # zip. 打包 inputs and labels filename path
            shuffle(self.data_pairs)  # shuffle 打乱顺序

    def read_rgb_image(self, addrs):

        pass

    def read_masks(self, addrs):

        pass


    def loadData(self, idx):
        indexes = self.indexes[idx * self.batch_size:(idx + 1) * self.batch_size]
        self.counter += 1
        print(self.counter)
        # Find the batch list of records
        batch_list = [self.data_pairs[k] for k in indexes]
        # print(len(batch))
        X, y = [], []
        count = 0
        for sample in batch_list:
            print(sample)
            # unzip the addrs paris








    def __len__(self):
        return math.ceil(len(self.input_image_addrs) / self.batch_size)

    def __getitem__(self, idx):
        # indexes = self.indexes[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_data  = self.loadData(idx)
        return batch_data
