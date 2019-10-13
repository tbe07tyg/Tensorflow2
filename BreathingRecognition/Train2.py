"""
Train our RNN on extracted features or images in Tensorflow 2.0.
"""
import os
from tensorflow.keras.callbacks import TensorBoard, LearningRateScheduler, ModelCheckpoint
from UiltiFuncs import schedule, get_compilied_model
from ModelZoo import LstmReg
import tensorflow as tf

def train(data_type, seq_length, model_tpye, saved_model=None,
          class_limit=None, image_shape=None,
          load_to_memory=False, batch_size=32, nb_epoch=100,NUM_GPUS=1):


    # Helper: Save the model.
    modelSavedPath = os.path.join('data', 'checkpoints', model_tpye + '-' + data_type + \
            '.{epoch:03d}-{val_loss:.3f}.hdf5')
    if not os.path.exists(modelSavedPath):
        os.makedirs(modelSavedPath)
    checkpoint_callback = ModelCheckpoint(
        filepath=os.path.join('data', 'checkpoints', model_tpye + '-' + data_type + \
            '.{epoch:03d}-{val_loss:.3f}.hdf5'),
        verbose=1,
        save_best_only=True)

    # Learning Rate Schedule callback
    lr_schedule_callback = LearningRateScheduler(schedule)

    # custom data generator




    # select model
    if model_tpye == "lstm_reg":
        # features_length = 2048
        # inputs = tf.keras.Input(shape=(seq_length, features_length))
        model = LstmReg()

    # define optimizers
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    mse_loss_fn = tf.keras.losses.MeanSquaredError()
    metric = tf.keras.metrics.mean_absolute_error()

    # fit the custom generator



    if NUM_GPUS == 1:
        model = get_compilied_model(model, loss=mse_loss_fn, opt=optimizer, metric=metric)
        model.fig_generator
    else:
        mirrored_strategy = tf.distribute.MirroredStrategy()
        with mirrored_strategy.scope():
            model= get_compilied_model(model, loss=mse_loss_fn, opt=optimizer, metric=metric)



