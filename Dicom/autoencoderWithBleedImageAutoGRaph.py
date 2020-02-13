#!/usr/bin/env python
# coding: utf-8

# # auto graph program  with bleeding train for auto encoder

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
ckp_log_root = "ckpts"
EPOCHS =30
total_num_batches_per_epoch = math.ceil(len(train_all_image_paths) / batch_size)
print("batch size:", batch_size)
print("total_num_batches per epoch:", total_num_batches_per_epoch)
print("input image_size:", image_size)


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
# train_dataset = train_dataset.repeat(EPOCHS)
train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

val_dataset = val_dataset.map(map_func=val_preprocess_inputs_auto,
                                  num_parallel_calls=tf.data.experimental.AUTOTUNE)
val_dataset = val_dataset.batch(batch_size=batch_size, drop_remainder=False) # drop reminder... if true batch= 6 otherwise =7
# val_dataset = val_dataset.repeat(EPOCHS)
val_dataset = val_dataset.prefetch(tf.data.experimental.AUTOTUNE)


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


# In[9]:


# evaluation -------------------------------------------->
train_avg_loss = tf.keras.metrics.Mean(name='train_avg_loss')
train_avg_metric = tf.keras.metrics.Mean(name='train_avg_metric')
test_avg_loss = tf.keras.metrics.Mean(name='test_avg_metric')
test_avg_metric = tf.keras.metrics.Mean(name='test_avg_metric')


# define loss
@tf.function
def mse_loss(y_true, y_pred, smooth=1):
    loss =tf.keras.losses.MSE(y_true,y_pred)
    return loss

@tf.function
def mae_loss(y_true, y_pred, smooth=1):
    loss =tf.keras.losses.MAE(y_true,y_pred)
    return loss


# ## #
# # above is global zone 

# In[10]:


def write_tb_logs_image(writer, name_list, value_list, step,max_outs):
    with writer.as_default():
        # optimizer.iterations is actually the entire counter from step 1 to step total batch
        for i in range(len(name_list)):
            # print(value_list[i].shape)
            # batch_images = np.expand_dims(value_list[i], -1)
            # print(batch_images.shape)
            tf.summary.image(name_list[i], value_list[i], step=step, max_outputs=max_outs)
            # value_list[i].reset_states()  # Clear accumulated values with .reset_states()
        writer.flush()

def write_tb_logs_scaler(writer, name_list, value_list, step):
    with writer.as_default():
        # optimizer.iterations is actually the entire counter from step 1 to step total batch
        for i in range(len(name_list)):
            tf.summary.scalar(name_list[i], value_list[i], step=step)
            # print(value_list[i].result())
            # value_list[i].reset_states()  # Clear accumulated values with .reset_states()
            # print(value_list[i].result())
        writer.flush()

@tf.function
def train_step(input_feature, labels, model, optimizer):
    with tf.GradientTape() as tape:
        predictions = model(input_feature)
        train_loss = mse_loss(labels, predictions)
        metric = mae_loss(labels, predictions)
    gradients = tape.gradient(train_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_avg_loss(train_loss)
    train_avg_metric(metric)

@tf.function
def test_step(input_feature, labels):

    predictions = model(input_feature)
    print("prediction shape:", predictions.shape)

    t_loss = mse_loss(labels, predictions)
    metric = mae_loss(labels, predictions)
    test_avg_loss(t_loss)
    test_avg_metric(metric)

    return predictions


# In[11]:


def  train_and_checkpoint(train_dataset, model, EPOCHS, opt,
                         train_summary_writer, test_summary_writer, graph_writer=None,
                         ckpt=None, ckp_freq=0, manager=None):
    temp_dice = 0 # mae the less the better

    ckpt.restore(manager.latest_checkpoint)
    #
    if manager.latest_checkpoint:
        print("Restored from {}".format(manager.latest_checkpoint))
    else:
        print("Initializing from scratch.")


    for epoch in range(EPOCHS):
        if epoch == 0:
            tf.summary.trace_on(graph=True, profiler=False)
        # lr_epoch= epoch
        epoch+=1
        test_avg_metric_list = []
        batch_count = 0
        # each train epoch
        for x, y in train_dataset.take(10):
            print("ckpt.step:", int(ckpt.step))
            print("x.shape:", x.shape)
            print("y.shape:", y.shape)
            write_tb_logs_image(train_summary_writer, ["input_image"], [x], opt.iterations, batch_size)
            write_tb_logs_image(train_summary_writer, ["input_target"], [y], opt.iterations, batch_size)
            batch_count+=1
            # print(x.shape)
            # print(y.shape)

            # train step ---- for batch training
            train_step(x, y, model, opt)

            batch_template = 'Step: {} Epoch {}- Batch[{}/{}], Train Avg Loss: {}, Train Avg dice: {}'
        # #
            print(batch_template.format(int(ckpt.step),
                                        epoch,
                                        batch_count,
                                        total_num_batches_per_epoch,
                                        train_avg_loss.result(),
                                        train_avg_metric.result()))


            print("lr:", opt._decayed_lr(tf.float32).numpy())
            write_tb_logs_scaler(train_summary_writer, ["lr"],
                                 [opt._decayed_lr(tf.float32)], int(ckpt.step))
            ckpt.step.assign_add(1)

        # val dataset per epoch end
        for x_val, y_val in val_dataset.take(4):
            # print("x_val.shape:", x_val.shape)
            # print("y_val.shape:", y_val.shape)
            predictions = test_step(x_val, y_val)
            write_tb_logs_image(test_summary_writer, ["val_input_image"], [x_val], opt.iterations, batch_size)
            write_tb_logs_image(test_summary_writer, ["val_input_target"], [y_val], opt.iterations, batch_size)
            write_tb_logs_image(test_summary_writer, ["predictions"], [predictions], opt.iterations, batch_size)

        epoch_template = 'Val ------> Epoch {}, Loss: {}, Dice: {}, Val Loss: {}, Val Dice: {}'
        print("*"*130)
        print(epoch_template.format(epoch,
                              train_avg_loss.result(),
                              train_avg_metric.result(),
                              test_avg_loss.result(),
                              test_avg_metric.result()))
        print("*"*130)
        # write train logs # with the same name for train and test write will write multiple curves into one plot
        write_tb_logs_scaler(train_summary_writer, ["epoch_avg_loss", "epoch_avg_Dice"],  # validation and train name need to be the same otherwise wont plot in one figure
                             [train_avg_loss.result(), train_avg_metric.result()], epoch)

        write_tb_logs_scaler(test_summary_writer, ["epoch_avg_loss", "epoch_avg_Dice"],
                             [test_avg_loss.result(), test_avg_metric.result()], epoch)

        if test_avg_metric.result() > temp_dice:
            # 保存模型
            # print("saving model...")
            # model.save('my_model.h5')
            # print("model saved")
            save_path = manager.save()
            print("Saved checkpoint for step {}: {}".format(int(ckpt.step), save_path))
            temp_dice = test_avg_metric.result()
            print("temp_avg_val_dice:", temp_dice.numpy())
        else:
            pass

        if epoch == 1:

            # at epoch end write graph of the all the computation model
            with graph_writer.as_default():
                tf.summary.trace_export(
                    name="my_func_trace",
                    step=0,
                    profiler_outdir=os.path.join('logs', 'graph'))


# # MAIN

# In[16]:


import os

if __name__ == '__main__':
    
    
    strategy = tf.distribute.MirroredStrategy()
    # with strategy.scope():

    # use designed model
    model = U_NetV2()
    # plot model graph
    # tf.keras.utils.plot_model(model, show_shapes=True, dpi=200, expand_nested=True)
    tb_log_root = "logs"
    ckpt = tf.train.Checkpoint(step=tf.Variable(1), net=model)
    manager = tf.train.CheckpointManager(ckpt, ckp_log_root, max_to_keep=3)

    if not os.path.exists(tb_log_root):
        print("build tensorboard log folder")
        os.makedirs(tb_log_root)
    train_summary_writer = tf.summary.create_file_writer(os.path.join(tb_log_root, 'train')) # tensorboard --logdir /tmp/summaries
    test_summary_writer = tf.summary.create_file_writer(os.path.join(tb_log_root, 'test'))
    graph_writer = tf.summary.create_file_writer(os.path.join(tb_log_root, 'graph'))

    initial_learning_rate = 1e-4
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=400,  # learning rate decay after every 100 steps
        decay_rate=0.96, #
        staircase=True)
    opt = tf.keras.optimizers.Adam(lr_schedule)
    train_and_checkpoint(train_dataset, model, EPOCHS, opt=opt, train_summary_writer=train_summary_writer,
                         test_summary_writer=test_summary_writer, graph_writer=graph_writer, ckpt=ckpt, manager=manager)


# In[ ]:




