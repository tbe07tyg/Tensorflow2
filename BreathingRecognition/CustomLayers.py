from tensorflow.keras import layers
import tensorflow as tf

class SummaryImage(layers.Layer):

  def __init__(self, image, max_outputs):
    super(SummaryImage, self).__init__()
    self.image = image
    self.max_outputs = max_outputs

  def call(self, inputs):
    tf.summary.image(self.image, max_outputs=self.max_outputs)
    return inputs