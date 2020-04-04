#!/usr/bin/env python
# coding: utf-8

# ## This is script is used for program custom generators
# 

# In[1]:


# imports 
import tensorflow as tf
from tensorflow.keras.utils import Sequence
from glob import glob
import numpy as np
import math
get_ipython().system('pip install pydicom')
get_ipython().system('pip install scikit-image')
get_ipython().system('pip install sklearn')
import pydicom


# for preprocessing
from skimage.transform import resize
from sklearn import preprocessing


# In[2]:


# define generators from inheritting Sequence class
class DicomGenegeratorAutoTFio(Sequence):
    
    # default functions needs to be overwriiten
    def __init__(self, batch_size, train_or_test, dims=(512, 512),shuffle=True, train_images_root=None, val_images_root=None, n_channels=1, prepro=True):
        self.train_images_root = train_images_root
        self.val_images_root = val_images_root
        self.batch_size = batch_size
        self.train_or_test =  train_or_test
        self.dims = dims
        self.shuffle = shuffle
        self.n_channels = n_channels
        self.prepro = prepro
        
        # custom functions inside function---------->
        def get_data_paths(root_dir):
            all_paths = [] 
            for each in root_dir:
                print("root:", each)
                one_root_paths =  sorted(glob(each +'/*.DCM'))
                print("one_root_paths:", len(one_root_paths))
                print()
                all_paths = all_paths+ one_root_paths
            return all_paths

        if self.train_or_test == "train":
            self.dcm_paths = get_data_paths(self.train_images_root)
#             print(f'Found {len(self.dcm_paths)} training images')
        else: 
            self.dcm_paths =  get_data_paths(self.val_images_root)
#             print(f'Found {len(self.dcm_paths)} validation images')

        self.on_epoch_end()
        

    def __len__(self):
        num_batches = math.ceil(len(self.dcm_paths) / self.batch_size)
#         print(f'Found {len(self.dcm_paths)} {self.train_or_test} images ')
        return math.ceil(len(self.dcm_paths) / self.batch_size)

    def __getitem__(self, idx):
        # Generate indexes of the batch
        indexes = self.indexes[idx*self.batch_size: (idx+1)*self.batch_size]
        
        # Find related DCMs' paths
        batch_temp_dcm_paths = [self.dcm_paths[k] for k in indexes]
        
      
        return self._generate_X_X(batch_temp_dcm_paths)
    
    def preprocessing(self, image):
        """
        --- Rescale Image
        --- Rotate Image
        --- Resize Image
        --- Flip Image
        --- PCA etc.
        
        """
#         print("range before resize:[{},{}]".format(np.min(image), np.max(image)))
        # 1. resize
        image = resize(image, self.dims, preserve_range=True) # set preserve_range = True otherwise the range will be changed
        # 2. normalization scale image with zero mean and unit std. per image
#         print("range after resize:[{},{}]".format(np.min(image), np.max(image)))
#         print("range before scaling:[{},{}]".format(np.min(image), np.max(image)))
#         print("mean before scaling:{}]".format(np.mean(image)))
#         print("std before scaling:{}]".format(np.std(image)))
#         print("shape before scaling:{}".format(image.shape))
#         image = preprocessing.scale(image.reshape((self.dims[0]*self.dims[1], -1))).reshape(self.dims)
#         print("mean after scaling:[{}]".format(np.mean(image)))
#         print("std after scaling:{}]".format(np.std(image)))
#         print("range after scaling:[{},{}]".format(np.min(image), np.max(image)))
#         print("shape after scaling:{}".format(image.shape))
        return image
    
    def _generate_X_X(self, batch_temp_dcm_paths):
        """
        batch_temp_dcm_paths : the batch of paths for reading dcms
        return: return the numpy array of dcm raw pix data
        """
        # initialization
        X = np.empty((self.batch_size, *self.dims, self.n_channels))
#         y = np.empty((self.batch_size), dtype=int)
        # Generate data
        for i, path in enumerate(batch_temp_dcm_paths):
            # Store sample
#             print("Per DCM path:",path)
            temp = self._load_dcm(path)
            # check whether needs to preprocess
            if self.prepro ==  True:
#                 print("prerpocessing.....")
                temp = self.preprocessing(temp)
            else:
                pass
            
            X[i,] = np.expand_dims(temp, axis=-1)
        
        return X, X
    
    def _generate_Y(self, batch_temp_dcm_paths):
        pass
    
    def _load_dcm(self,dcm_path):
#         print(dcm_path)
        ds = pydicom.dcmread(dcm_path)
        pix = ds.pixel_array
#         print("Shape info: ---------->", pix.shape )
#         print("raw DCM content range [{}, {}]".format(np.min(pix), np.max(pix)))
        return pix
        
        
    def on_epoch_end(self):
        #'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.dcm_paths))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)


# ## main test

# In[3]:



if __name__ == '__main__':
    # root dirs for dcms
    train_images_root = sorted(glob('/media/ytx/Japan_Deep_Data/dataset/LeiSang/myTry/BleedingDataDCM/train/*'))
    val_images_root = sorted(glob('/media/ytx/Japan_Deep_Data/dataset/LeiSang/myTry/BleedingDataDCM/val/*'))
    
    #hyperparameters
    batch_size = 2
    
    # create generator instances
    train_dcm_gen =  DicomGenegeratorAutoTFio(batch_size=batch_size, 
                                          train_or_test="train", 
                                          dims =(512, 512),
                                          shuffle=False,
                                          train_images_root=train_images_root)
    val_dcm_gen =  DicomGenegeratorAutoTFio(batch_size=batch_size, 
                                          train_or_test="val", 
                                          dims =(512, 512),
                                          shuffle=False,
                                          val_images_root=val_images_root)
    
    print(f'Found {train_dcm_gen.__len__()} training batches')
    print(f'Found {val_dcm_gen.__len__()} validation batches')
    
    # check the data in data generator
    for idx, data in enumerate(train_dcm_gen):
        print(idx)
        print("train_sample shape:", data[0].shape) # remember data generator now return (X, X)
        print("train_target shape:", data[1].shape)
#         print("min: {} max:{}".format(np.min(data), np.max(data)))
        
    for idx, data in enumerate(val_dcm_gen):
        print(idx)
        print("train_sample shape:", data[0].shape) # remember data generator now return (X, X)
        print("train_target shape:", data[1].shape)
#         print("min: {} max:{}".format(np.min(data), np.max(data)))

    
    print("train dataset is ok")
    print("val dataset is ok")


# 
# 
# 

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




