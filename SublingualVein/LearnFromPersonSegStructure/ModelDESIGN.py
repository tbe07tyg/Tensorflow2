from tensorflow import keras
from SublingualVein.KerasUNet.hyperparameters import image_size
import tensorflow as tf

def con_bn_act(x, filters, kernel_size=(3, 3), padding="same", strides=1):
    initializer = tf.random_normal_initializer(0., 0.02)
    c = keras.layers.Conv2D(filters, kernel_size, strides=strides, padding=padding,
                             kernel_initializer=initializer, use_bias=False)(x)
    c = tf.keras.layers.BatchNormalization()(c)
    c = tf.keras.layers.ReLU()(c)
    return c

def down_block(x, filters, kernel_size=(3, 3), padding="same", strides=1):
    c = con_bn_act(x, filters, kernel_size, padding=padding, strides=strides)
    c = con_bn_act(c, filters, kernel_size, padding=padding, strides=strides)
    p = keras.layers.MaxPool2D((2, 2), (2, 2))(c)
    return c, p

def up_block(x, skip, filters, kernel_size=(3, 3), padding="same", strides=1):
    us = keras.layers.UpSampling2D((2, 2))(x)
    concat = keras.layers.Concatenate()([us, skip])
    c = con_bn_act(concat, filters, kernel_size, padding=padding, strides=strides)
    c = con_bn_act(c, filters, kernel_size, padding=padding, strides=strides)
    return c

def bottleneck(x, filters, kernel_size=(3, 3), padding="same", strides=1):
    c = con_bn_act(x, filters, kernel_size, padding=padding, strides=strides)
    c = con_bn_act(c, filters, kernel_size, padding=padding, strides=strides)
    return c


def UNet(inChannels=3):
    f = [16, 32, 64, 128, 256]
    inputs = keras.layers.Input((image_size, image_size, inChannels))

    p0 = inputs
    c1, p1 = down_block(p0, f[0])  # 128 -> 64
    c2, p2 = down_block(p1, f[1])  # 64 -> 32
    c3, p3 = down_block(p2, f[2])  # 32 -> 16
    c4, p4 = down_block(p3, f[3])  # 16->8

    bn = bottleneck(p4, f[4])

    u1 = up_block(bn, c4, f[3])  # 8 -> 16
    u2 = up_block(u1, c3, f[2])  # 16 -> 32
    u3 = up_block(u2, c2, f[1])  # 32 -> 64
    u4 = up_block(u3, c1, f[0])  # 64 -> 128

    outputs = keras.layers.Conv2D(1, (1, 1), padding="same", activation="sigmoid")(u4)
    model = keras.models.Model(inputs, outputs)
    return model