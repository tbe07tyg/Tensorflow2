import tensorflow as tf
# import
#
base_model = tf.keras.applications.MobileNetV2(input_shape=[224, 224, 3], include_top=False)
tf.keras.utils.plot_model(base_model, to_file="base_MobileNetV2.png", show_shapes=True, dpi=64)


