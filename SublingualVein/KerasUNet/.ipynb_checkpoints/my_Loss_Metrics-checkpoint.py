from tensorflow.keras import backend as K
import tensorflow as tf



@tf.function
def dice_coef(y_true, y_pred, smooth=1):
  intersection = K.sum(y_true * y_pred, axis=[1,2,3])
  union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
  dice = K.mean((2. * intersection + smooth)/(union + smooth), axis=0)
  return dice

@tf.function
def iou_coef(y_true, y_pred, smooth=1):
  intersection = K.sum(K.abs(y_true * y_pred), axis=[1,2,3])
  union = K.sum(y_true,[1,2,3])+K.sum(y_pred,[1,2,3])-intersection
  iou = K.mean((intersection + smooth) / (union + smooth), axis=0)
  return iou

@tf.function()
def my_loss(y_true, y_pred, LOSS_MODE):
    # mask = tf.equal(y_true, 255)
    # mask = tf.logical_not(mask)
    # print(y_pred)
    # print(y_true)
    # print(tf.losses.binary_crossentropy(y_true, y_pred).shape)
    # print(tf.abs(y_true - y_pred).shape)
    if LOSS_MODE == "L1+BCE":
        total_loss = tf.expand_dims(tf.losses.binary_crossentropy(y_true, y_pred), -1) + tf.abs(y_true - y_pred) # tf.losses.binary_crossentropy(y_true, y_pred) result shape (3, 1024, 1280)
    elif LOSS_MODE == "BCE":
        total_loss = tf.expand_dims(tf.losses.binary_crossentropy(y_true, y_pred), -1)
    elif LOSS_MODE == "L1":
        total_loss = tf.abs(y_true - y_pred)
    else:
        total_loss = tf.losses.Huber(y_true, y_pred)
    # y_pred = tf.boolean_mask(y_pred, mask)
    return total_loss

@tf.function()
def my_loss_BCE(y_true, y_pred):
    total_loss = tf.expand_dims(tf.losses.binary_crossentropy(y_true, y_pred), -1)

    return total_loss