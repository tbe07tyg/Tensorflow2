#!/usr/bin/env python
# coding: utf-8

# ## autoenocder with own dataset

# In[1]:


from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Reshape, Dropout, UpSampling2D
from tensorflow.keras.callbacks import LambdaCallback
import numpy as np
from tensorflow import keras
import tensorflow as tf
from glob import glob
import math
from tensorflow import io

# import pydicom


# In[2]:


#  set dataset path
train_images_root = sorted(glob('/media/ytx/Japan_Deep_Data/dataset/LeiSang/myTry/BleedingDataDCM/train/*'))
# train_masks = sorted(glob('I:/dataset/infaredSublingualVein/train/tongue_labels/*'))

val_images_root = sorted(glob('/media/ytx/Japan_Deep_Data/dataset/LeiSang/myTry/BleedingDataDCM/val/*'))
# val_masks = sorted(glob('I:/dataset/infaredSublingualVein/validation/tongue_labels/*'))
print("train_images_root: ", train_images_root)
print("val_images_root: ", val_images_root)

train_all_image_paths =  []
val_all_image_paths = []

print("train paths--------------------------------------------------------------->")
for each in train_images_root:
    print("root:", each)
    one_root_paths =  sorted(glob(each +'/*.DCM'))
    print("one_root_paths:", len(one_root_paths))
    print()
    train_all_image_paths = train_all_image_paths+ one_root_paths
    print("train_all_image_paths:", len(train_all_image_paths))

print("val paths--------------------------------------------------------------->")
for each in val_images_root:
    print("root:", each)
    one_root_paths =  sorted(glob(each +'/*.DCM'))
    print("one_root_paths:", len(one_root_paths))
    print()
    val_all_image_paths = val_all_image_paths+ one_root_paths
print(f'Found {len(train_all_image_paths)} training images')
print(f'Found {len(val_all_image_paths)} validation images')


# In[3]:


# hyperparameter
encoding_size = 32
batch_size =  2
image_size = 512
img_width = img_height = image_size 
Epoches = 30
total_num_batches_per_epoch = math.ceil(len(train_all_image_paths) / batch_size)

total_num_batches_per_val = math.ceil(len(val_all_image_paths) / batch_size)
print("batch size:", batch_size)
print("total_num_batches per epoch:", total_num_batches_per_epoch)
print("input image_size:", image_size)
print("total epoches:", Epoches)


# In[4]:


# load dcm 
# !pip install -q tensorflow-io-nightly

import tensorflow_io as tfio
import matplotlib.pyplot as plt
def load_dcm(dcm_path, dcm=False):
   
    if dcm:
        
        print("read dcm data")
        _bytes = tf.io.read_file(dcm_path)
        img = tfio.image.decode_dicom_image( _bytes)  # defualt on_error= "strict": throw an error if can not throw one except; scale =  perserve defulat means keeps the value they are
        print(img.shape)  # (1, 512, 512) after decode image shape is this, WHICH CAN BE SEEN FROM PLTDicom.ipynb
        
        img =  tf.squeeze(img) # squeez the dimension to (512, 512)
        img =  tf.expand_dims(img, axis =-1) # expand the last dimension to 1, as (512, 512, 1)
        img.set_shape([None, None, 1])  # need to set the shape because the shape will becomes unknown with preprocessing function load
    else:
        
       raise "please choose dcm as input formart"
    return img


# In[5]:


# preprocessing
def resize(image):
    print(image.shape)
    resized_image = tf.image.resize(image, size=[image_size, image_size], method='bilinear')
    return resized_image

def std_norm(image):
    image = tf.image.per_image_standardization(image)
    return image


def random_flip_auto(image):
    flip = tf.random.uniform(
        shape=[1, ], minval=0, maxval=2, dtype=tf.int32)[0]
    image = tf.case([
        (tf.greater(flip, 0), lambda: tf.image.flip_left_right(image))
    ], default=lambda: image)
    return image

@tf.function()
def train_preprocess_inputs_auto(image_path):
    print(image_path)
    with tf.device('/cpu:0'):
      
        # image = load_image(image_path) # infraed image input. there for 8 bit input
        image = tf.cast(load_dcm(image_path, dcm=True), tf.float32)  # infraed image input. there for 8 bit input
        
        print("load image shape:", image.shape)
#         mask = load_image(mask_path, mask=True)
#         mask = tf.cast(mask > 0, dtype=tf.float32)
        print(image)
#         image = resize(image)
        # image, mask = random_scale(image, mask) # random resize
        image = std_norm(image)  # norm before padding and crop_pad
        # image, mask = pad_inputs(image, mask)  # and pad to raw size
        # image, mask = random_crop(image, mask)  #
        image = random_flip_auto(image)
        print("prepro image shape:", image.shape)
        return image, image

#         image = resize(image)import tensorflow_io as tfio
@tf.function()
def val_preprocess_inputs_auto(image_path):
    print(image_path)
    with tf.device('/cpu:0'):
      
        # image = load_image(image_path) # infraed image input. there for 8 bit input
        image = tf.cast(load_dcm(image_path, dcm=True), tf.float32)  # infraed image input. there for 8 bit input
        
        print("load image shape:", image.shape)
#         mask = load_image(mask_path, mask=True)
#         mask = tf.cast(mask > 0, dtype=tf.float32)
        print(image)
#         image = resize(image)
        # image, mask = random_scale(image, mask) # random resize
        image = std_norm(image)  # norm before padding and crop_pad
        # image, mask = pad_inputs(image, mask)  # and pad to raw size
        # image, mask = random_crop(image, mask)  #
        image = random_flip_auto(image)
        print("prepro image shape:", image.shape)
        return image, image


# In[6]:


print("The entire training dataset is:", len(train_all_image_paths))
print("The entire validation dataset is:", len(val_all_image_paths))
train_dataset = tf.data.Dataset.from_tensor_slices(train_all_image_paths)
val_dataset = tf.data.Dataset.from_tensor_slices(val_all_image_paths)
# for idx, data in enumerate(train_dataset):
#     print(idx)
#     print(data)
    
# train_dataset = train_dataset.shuffle(1024)
train_dataset = train_dataset.map(map_func=train_preprocess_inputs_auto,
                                  num_parallel_calls=tf.data.experimental.AUTOTUNE)
train_dataset = train_dataset.batch(batch_size=batch_size, drop_remainder=False) # drop reminder... if true batch= 6 otherwise =7
train_dataset = train_dataset.repeat(Epoches)
train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

val_dataset = val_dataset.map(map_func=val_preprocess_inputs_auto,
                                  num_parallel_calls=tf.data.experimental.AUTOTUNE)
val_dataset = val_dataset.batch(batch_size=batch_size, drop_remainder=False) # drop reminder... if true batch= 6 otherwise =7
# val_dataset = val_dataset.repeat(Epoches)
val_dataset = val_dataset.prefetch(tf.data.experimental.AUTOTUNE)

# for idx, data in enumerate(train_dataset):
#     print(idx)
#     print(data[0].shape)
#     print("min: {} max:{}".format(np.min(data[0]), np.max(data[0])))


# for idx, data in enumerate(val_dataset):
#     print(idx)
#     print(data[0].shape)
#     print("min: {} max:{}".format(np.min(data[0]), np.max(data[0])))
# print("train dataset is ok")
# print("val dataset is ok")


# In[7]:


# # create model
def U_NetV2():
    inputs = keras.layers.Input((image_size, image_size, 1))
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(drop5))
#     merge6 = concatenate([drop4, up6], axis=3)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(up6)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

    up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv6))
#     merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(up7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

    up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv7))
#     merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(up8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

    up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv8))
#     merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(up9)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    outputs = Conv2D(1, 1, activation='linear')(conv9)
    model = keras.models.Model(inputs, outputs)
    return model
model =U_NetV2()
model.compile(loss='mse', optimizer='adam',
              metrics=['mse'])
model.summary()

# keras.utils.plot_model(model, show_shapes=True, dpi=200, expand_nested=True)


# In[8]:


# add the image call back
import os
from datetime import datetime
logdir =  os.path.join("logs","image" + datetime.now().strftime("%Y%m%d-%H%M%S"))
# Define the basic TensorBoard callback.
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)
file_writer_img = tf.summary.create_file_writer(logdir + '/img')

@tf.function()
def draw_input_output(epoch, logs):
    output = model.predict(val_dataset)
    print(output.shape)
    with file_writer_img.as_default():
        tf.summary.image("test_output", tf.reshape(output,[-1, image_size , image_size , 1]), step=epoch)
#         tf.summary.image("test_noisy_input", tf.reshape(x_test_noisy,[-1, image_size , image_size , 1]), step=epoch)
        tf.summary.image("test_input", tf.reshape(val_dataset,[-1, image_size , image_size , 1]), step=epoch)
log_mg = keras.callbacks.LambdaCallback(on_epoch_end=draw_input_output)


# In[ ]:


# Fit the model
# !export LD_LIBRARY_PATH=/usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH
# model.fit(train_dataset, epochs=Epoches, steps_per_epoch=total_num_batches_per_epoch, validation_steps=total_num_batches_per_val,validation_data=val_dataset,
#           callbacks=[tensorboard_callback, log_mg],verbose=1)
# model.fit(train_dataset, epochs=Epoches, steps_per_epoch=total_num_batches_per_epoch, validation_steps=total_num_batches_per_val,validation_data=val_dataset,
#           callbacks=[tensorboard_callback],verbose=1)

model.fit(train_dataset, epochs=Epoches, steps_per_epoch=6, validation_steps=6,validation_data=val_dataset,
          callbacks=[tensorboard_callback],verbose=1)


# In[ ]:


# open tensorboard


# In[ ]:





# In[ ]:





# In[ ]:




