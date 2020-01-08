import matplotlib.pyplot as plt
import tensorflow as tf

def display(display_list):
  plt.figure(figsize=(15, 15))

  title = ['Input Image', 'True Mask', 'Predicted Mask']

  for i in range(len(display_list)):
    plt.subplot(1, len(display_list), i+1)
    plt.title(title[i])
    # if display_list[i].shape[-1] ==1:
    #   display_list[i].reshape([display_list[i].shape[0], display_list[i].shape[1]])
    # else:
    #   display_list[i].reshape([display_list[i].shape[0], display_list[i].shape[1], 3])
    plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
    plt.axis('off')
  plt.show()

def normalize(input_image, input_mask):
  input_image = tf.cast(input_image, tf.float32)/255.0
  # input_mask -= 1
  input_mask =  tf.cast(input_mask, tf.float32)/255.0
  return input_image, input_mask

def create_mask(pred_mask):
  pred_mask = tf.argmax(pred_mask, axis=-1)
  print(pred_mask.shape)
  pred_mask = pred_mask[..., tf.newaxis]
  print(pred_mask.shape)
  return pred_mask[0]

