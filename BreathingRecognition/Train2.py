"""
Train our RNN on extracted features or images in Tensorflow 2.0.
"""
import os
from tensorflow import keras
from tensorflow.keras.callbacks import TensorBoard, LearningRateScheduler, ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam

from UiltiFuncs import schedule, get_compilied_model
from ModelZoo import LstmReg, Lstm
from CustomGenerator import mySeqFeatureRegGenerator
import tensorflow as tf

def train(data_type, seq_length, model_tpye, saved_model=None,
          class_limit=None, image_shape=None,
          load_to_memory=False, batch_size=32, nb_epoch=100, NUM_GPUS=1):


    # Helper: Save the model.
    modelSavedPath = './checkpoints'
    if not os.path.exists(modelSavedPath):
        os.makedirs(modelSavedPath)
    checkpoint_callback = ModelCheckpoint(
        filepath=os.path.join(modelSavedPath, model_tpye + '-' + data_type + \
            '.{epoch:03d}-{val_loss:.3f}.hdf5'),
        verbose=1,
        save_best_only=True)

    # Learning Rate Schedule callback
    lr_schedule_callback = LearningRateScheduler(schedule)

    # EarlyStop_callback
    ES_callback = EarlyStopping(patience=10)

    # Tensorboard_callback
    tb_callback = TensorBoard(log_dir=os.path.join(os.getcwd(), 'logs', model_tpye))

    # custom data generator
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
        model = LstmReg(image_shape)
        loss = "mse"
        metrics = ["mse", "mae"]
    elif model_tpye == "lstm":
        print("model_type", model_tpye)
        num_classes = len(train_Generator.classes)
        model = Lstm(num_classes)
        loss = "categorical_crossentropy"
        metrics = ["accuracy"]
        if num_classes >= 10:
            metrics.append('top_k_categorical_accuracy')
    else:
        pass


    # define optimizers
    optimizer = Adam(lr=1e-5, decay=1e-6)

    print("loss:", loss)
    print("metrics:", metrics)
    print("optimizer:", optimizer)

    if NUM_GPUS == 1:
        model = get_compilied_model(model, loss=loss, opt=optimizer, metrics=metrics)
        # fit the custom generator
        model.fit_generator(generator=train_Generator,
                            validation_data=test_Generator,
                            use_multiprocessing=True,
                            callbacks=[lr_schedule_callback, checkpoint_callback, ES_callback, tb_callback],
                            workers=4,
                            epochs=nb_epoch,
                            shuffle=True)

        # # resume training from the checkpoint
        # model_info = model.fit(train_dataset,
        #                        epochs=NUM_EPOCHS_2, callbacks=[checkpoint_callback],
        #                        validation_data=test_dataset,
        #                        validation_freq=1,
        #                        initial_epoch=INIT_EPOCH_2)
        # print summary
        print(model.summary())
    else:
        mirrored_strategy = tf.distribute.MirroredStrategy()
        with mirrored_strategy.scope():
            model= get_compilied_model(model, loss=loss, opt=optimizer, metrics=metrics)

            # fit the custom generator
            model.fit_generator(generator=train_Generator,
                                validation_data=test_Generator,
                                use_multiprocessing=True,
                                callbacks=[lr_schedule_callback, checkpoint_callback, ES_callback, tb_callback],
                                workers=4,
                                epochs=nb_epoch,
                                shuffle=True)
            # print summary
            print(model.summary())
            # # resume training from the checkpoint
            # model_info = model.fit(train_dataset,
            #                        epochs=NUM_EPOCHS_2, callbacks=[checkpoint_callback],
            #                        validation_data=test_dataset,
            #                        validation_freq=1,
            #                        initial_epoch=INIT_EPOCH_2)


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
          load_to_memory=load_to_memory, batch_size=batch_size, nb_epoch=nb_epoch)

if __name__ == '__main__':
    main()