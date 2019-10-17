"""
Train our RNN on extracted features or images in Tensorflow 2.0.
"""
import os
from tensorflow import keras
from tensorflow.keras.callbacks import TensorBoard, LearningRateScheduler, ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam

from UiltiFuncs import schedule, get_compilied_model, Clean_CheckpointsCaches
from ModelZoo import LstmReg, Lstm
from CustomGenerator import mySeqFeatureRegGenerator
import tensorflow as tf
import matplotlib.pyplot as plt

def train(data_type, seq_length, model_tpye,  log_path, saved_model=None,
          class_limit=None, image_shape=None,
          load_to_memory=False, batch_size=32, nb_epoch=100, NUM_GPUS=1, lr_plan =False, lr=1e-4):


    # Helper: Save the model.
    modelSavedPath = './checkpoints'
    if not os.path.exists(modelSavedPath):
        os.makedirs(modelSavedPath)
    checkpoint_callback = ModelCheckpoint(
        filepath=os.path.join(modelSavedPath, model_tpye +  "-" + str(lr)+'-' + data_type + \
            '.{epoch:03d}-{val_accuracy:.3f}.hdf5'),
        verbose=1,
        monitor="val_accuracy",
        save_best_only=True)

    # clean saved check points caches
    ck_cleaner = Clean_CheckpointsCaches(model_type=model_tpye, folder_path=modelSavedPath,feature_type=data_type)

    # EarlyStop_callback
    ES_callback = EarlyStopping(patience=10)
    # Tensorboard_callback
    tb_callback = TensorBoard(log_dir=log_path, update_freq='batch',
                              histogram_freq=1, profile_batch=3)

    # writer = tf.summary.create_file_writer(log_path)
    # tf.summary.trace_on(graph=True, profiler=True)


    # # custom data generator
    # train_Generator = mySeqFeatureRegGenerator(batch_size=batch_size, train_test="train", data_type=data_type,
    #                                            task_type="classification", seq_length=seq_length, class_limit=class_limit,
    #                                            csv_path_root="I:\\DeepLearning\\BreathingRecognition\\five-video-classification-methods-master\\data",
    #                                            sequence_path="I:\\DeepLearning\\BreathingRecognition\\five-video-classification-methods-master\\data\\sequences"
    #                                            )
    #
    # test_Generator = mySeqFeatureRegGenerator(batch_size=batch_size, train_test="test", data_type=data_type,
    #                                            task_type="classification", seq_length=seq_length,
    #                                           csv_path_root="I:\\DeepLearning\\BreathingRecognition\\five-video-classification-methods-master\\data",
    #                                           sequence_path="I:\\DeepLearning\\BreathingRecognition\\five-video-classification-methods-master\\data\\sequences"
    #                                           )

    # FOR DESKTOP
    train_Generator = mySeqFeatureRegGenerator(batch_size=batch_size, train_test="train", data_type=data_type,
                                               task_type="classification", seq_length=seq_length,
                                               class_limit=class_limit,
                                               csv_path_root="C:\\deeplearningProjects\\BreathingRecognition\\five-video-classification-methods-master\\data",
                                               sequence_path="C:\\deeplearningProjects\\BreathingRecognition\\five-video-classification-methods-master\\data\\sequences"
                                               )

    test_Generator = mySeqFeatureRegGenerator(batch_size=batch_size, train_test="test", data_type=data_type,
                                              task_type="classification", seq_length=seq_length,
                                              csv_path_root="C:\\deeplearningProjects\\BreathingRecognition\\five-video-classification-methods-master\\data",
                                              sequence_path="C:\\deeplearningProjects\\BreathingRecognition\\five-video-classification-methods-master\\data\\sequences"
                                              )

    # select model
    model = None
    if model_tpye == "lstm_reg":
        # features_length = 2048
        # inputs = tf.keras.Input(shape=(seq_length, features_length))
        model = LstmReg(image_shape).model()
        loss = "mse"
        metrics = ["mse", "mae"]
    elif model_tpye == "lstm":
        print("model_type", model_tpye)
        num_classes = len(train_Generator.classes)
        model = Lstm(num_classes, image_shape).model()
        loss = "categorical_crossentropy"
        metrics = ["accuracy"]
        if num_classes >= 10:
            metrics.append('top_k_categorical_accuracy')
    else:
        pass

    tf.summary.trace_export(
        name="my_trace",
        step=0,
        profiler_outdir=log_path)
    # define optimizers
    if lr_plan ==True:
        nb_epoch=100
        optimizer = Adam(lr=1e-8)  # for learning rate schedular
        lr_schedule = tf.keras.callbacks.LearningRateScheduler(
            lambda epoch: 1e-8 * 10 ** (epoch / 20))
    else:
        optimizer = Adam(lr=lr, decay=0.3e-5)
        lr_schedule = tf.keras.callbacks.LearningRateScheduler(
            lambda epoch: 1e-8 * 10 ** (epoch / nb_epoch))
    print("loss:", loss)
    print("metrics:", metrics)
    print("optimizer:", optimizer)

    if NUM_GPUS == 1:
        model = get_compilied_model(model, loss=loss, opt=optimizer, metrics=metrics)
        # print summary
        print(model.summary())
        # fit the custom generator
        history =model.fit_generator(generator=train_Generator,
                            validation_data=test_Generator,
                            use_multiprocessing=True,
                            callbacks=[lr_schedule, checkpoint_callback, ES_callback, tb_callback, ck_cleaner],
                            workers=4,
                            epochs=nb_epoch,
                            shuffle=True)

        # # resume training from the checkpoint
        # model_info = model.fit(train_dataset,
        #                        epochs=NUM_EPOCHS_2, callbacks=[checkpoint_callback],
        #                        validation_data=test_dataset,
        #                        validation_freq=1,
        #                        initial_epoch=INIT_EPOCH_2)

    else:
        mirrored_strategy = tf.distribute.MirroredStrategy()
        with mirrored_strategy.scope():
            model= get_compilied_model(model, loss=loss, opt=optimizer, metrics=metrics)
            # print summary
            print(model.summary())
            # fit the custom generator
            history = model.fit_generator(generator=train_Generator,
                                validation_data=test_Generator,
                                use_multiprocessing=True,
                                callbacks=[lr_schedule, checkpoint_callback, ES_callback, tb_callback, ck_cleaner],
                                workers=4,
                                epochs=nb_epoch,
                                shuffle=True)

            # # resume training from the checkpoint
            # model_info = model.fit(train_dataset,
            #                        epochs=NUM_EPOCHS_2, callbacks=[checkpoint_callback],
            #                        validation_data=test_dataset,
            #                        validation_freq=1,
            #                        initial_epoch=INIT_EPOCH_2)

    if lr_plan ==True:
        plt.semilogx(history.history["lr"], history.history["loss"])
        plt.axis([1e-8, 1e-4, 0, 30])
        plt.savefig('lr.png') #
def main():
    """These are the main training settings. Set each before running
    this file."""
    # model can be one of lstm, lrcn, mlp, conv_3d, c3d
    model = 'lstm'
    saved_model = None  # None or weights file
    class_limit = None  # int, can be 1-101 or None
    seq_length = 40
    load_to_memory = False  # pre-load the sequences into memory
    batch_size = 32
    nb_epoch = 1000
    feature_length =2048
    lr = 1e-4
    tb_log_path = os.path.join(os.getcwd(), 'logs', model)

    # Chose images or features and image shape based on network.
    if model in ['conv_3d', 'c3d', 'lrcn']:
        data_type = 'images'
        image_shape = (80, 80, 3)
    elif model in ['lstm', 'mlp', 'lstm_reg']:
        data_type = 'features'
        image_shape = (seq_length, feature_length)
    else:
        raise ValueError("Invalid model. See train.py for options.")

    train(data_type, seq_length, model, saved_model=saved_model,
      class_limit=class_limit, image_shape=image_shape,
      load_to_memory=load_to_memory, batch_size=batch_size, nb_epoch=nb_epoch, lr=lr, log_path=tb_log_path)

if __name__ == '__main__':
    main()