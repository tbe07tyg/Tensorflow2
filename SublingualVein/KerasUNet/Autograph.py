import tensorflow as tf
from glob import glob
from SublingualVein.LearnFromPersonSegStructure.ModelDESIGN import UNet
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
import os
from SublingualVein.KerasUNet.hyperparameters import image_size
import math
import numpy as np
print('TensorFlow', tf.__version__)
os.environ['TF_XLA_FLAGS'] = '--tf_xla_cpu_global_jit'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

os.environ["PATH"] += os.pathsep + 'I:/DeepLearning/TensorflowV2/graphviz-2.38/release/bin/'

# H, W = 784, 784
batch_size = 2
EPOCHS = 1000
log_freq = 1
ckp_log_root = "/logs"


# train_images = sorted(glob('resized_images/*'))
# train_masks = sorted(glob('resized_masks/*'))
#
# val_images = sorted(glob('validation_data/images/*'))
# val_masks = sorted(glob('validation_data/masks/*'))

train_images = sorted(glob('I:/dataset/infaredSublingualVein/train/raw_image/*'))
train_masks = sorted(glob('I:/dataset/infaredSublingualVein/train/tongue_labels/*'))

val_images = sorted(glob('I:/dataset/infaredSublingualVein/validation/raw_image/*'))
val_masks = sorted(glob('I:/dataset/infaredSublingualVein/validation/tongue_labels/*'))

print(f'Found {len(train_images)} training images')
print(f'Found {len(train_masks)} training masks')

print(f'Found {len(val_images)} validation images')
print(f'Found {len(val_masks)} validation masks')

total_num_batches_per_epoch = math.ceil(len(train_images) / batch_size)
print("batch size:", batch_size)
print("total_num_batches per epoch:", total_num_batches_per_epoch)
for i in range(len(train_masks)):
    print(train_images)
    print("train_image:", train_images[i].split('/')[-1].split('\\')[-1].split('.')[0])
    print("train_mask:",train_masks[i].split('/')[-1].split('\\')[-1].split('.')[0])
    assert train_images[i].split('/')[-1].split('\\')[-1].split('.')[0] \
           == train_masks[i].split('/')[-1].split('\\')[-1].split('.')[0]

for i in range(len(val_masks)):
    assert val_images[i].split('/')[-1].split('\\')[-1].split('.')[0] \
           == val_masks[i].split('/')[-1].split('\\')[-1].split('.')[0]


def random_scale(image, mask, min_scale=0.65, max_scale=2.5):
    random_scale = tf.random.uniform(shape=[1],
                                     minval=min_scale,
                                     maxval=max_scale)
    dims = tf.cast(tf.shape(image), dtype=tf.float32)
    new_dims = tf.cast(random_scale * dims[:2], dtype=tf.int32)
    scaled_image = tf.image.resize(image, size=new_dims, method='bilinear')
    scaled_mask = tf.image.resize(mask, size=new_dims, method='nearest')
    return scaled_image, scaled_mask


def pad_inputs(image,
               mask,
               crop_height=image_size,
               crop_width=image_size,
               ignore_value=255,
               pad_value=0):
    dims = tf.cast(tf.shape(image), dtype=tf.float32)
    h_pad = tf.maximum(1 + crop_height - dims[0], 0)
    w_pad = tf.maximum(1 + crop_width - dims[1], 0)
    padded_image = tf.pad(image, paddings=[[0, h_pad], [0, w_pad], [
                          0, 0]], constant_values=pad_value)
    padded_mask = tf.pad(mask, paddings=[[0, h_pad], [0, w_pad], [
                         0, 0]], mode='CONSTANT', constant_values=ignore_value)
    return padded_image, padded_mask


def random_crop(image, mask, crop_height=image_size, crop_width=image_size):
    image_dims = tf.shape(image)
    offset_h = tf.random.uniform(
        shape=(1,), maxval=image_dims[0] - crop_height, dtype=tf.int32)[0]
    offset_w = tf.random.uniform(
        shape=(1,), maxval=image_dims[1] - crop_height, dtype=tf.int32)[0]

    image = tf.image.crop_to_bounding_box(image,
                                          offset_height=offset_h,
                                          offset_width=offset_w,
                                          target_height=crop_height,
                                          target_width=crop_height)
    mask = tf.image.crop_to_bounding_box(mask,
                                         offset_height=offset_h,
                                         offset_width=offset_w,
                                         target_height=crop_height,
                                         target_width=crop_height)
    return image, mask


def random_flip(image, mask):
    flip = tf.random.uniform(
        shape=[1, ], minval=0, maxval=2, dtype=tf.int32)[0]
    image = tf.case([
        (tf.greater(flip, 0), lambda: tf.image.flip_left_right(image))
    ], default=lambda: image)
    mask = tf.case([
        (tf.greater(flip, 0), lambda: tf.image.flip_left_right(mask))
    ], default=lambda: mask)
    return image, mask

def std_norm(image):
    image = tf.image.per_image_standardization(image)
    return image



def load_image(image_path, mask=False):
    img = tf.io.read_file(image_path)
    if mask:
        img = tf.image.decode_image(img, channels=1)
        img.set_shape([None, None, 1])
    # elif mask == "auto":
    #     print("auto true")
    #     img = tf.image.decode_image(img, channels=None)
    else:
        img = tf.image.decode_image(img, channels=3)
        img.set_shape([None, None, 3])
    return img

def load_bmp(image_path):
    img = tf.io.read_file(image_path)
    img = tf.io.decode_bmp(img)
    img.set_shape([None, None, 1])
    return img


@tf.function()
def train_preprocess_inputs(image_path, mask_path):
    with tf.device('/cpu:0'):
        # image = load_image(image_path) # infraed image input. there for 8 bit input
        image = tf.cast(load_bmp(image_path), tf.float32)  # infraed image input. there for 8 bit input
        print("load image shape:", image.shape)
        mask = load_image(mask_path, mask=True)
        mask = tf.cast(mask > 0, dtype=tf.float32)
        print(image)
        # image, mask = random_scale(image, mask) # random resize
        image = std_norm(image)  # norm before padding and crop_pad
        # image, mask = pad_inputs(image, mask)  # and pad to raw size
        # image, mask = random_crop(image, mask)  #
        image, mask = random_flip(image, mask)
        print("prepro image shape:", image.shape)
        print("prepro mask shape:", mask.shape)

        # image = image[:, :, ::-1] - tf.constant([103.939, 116.779, 123.68]) #  # 將 BGR 圖片轉為 RGB 圖片
        # image = image[:, :, ::-1]   # # 將 BGR 圖片轉為 RGB 圖片
        # image = image[:, :, ::-1] - tf.constant([0.0, 0.0, 0.0])  # # 將 BGR 圖片轉為 RGB 圖片
        return image, mask

@tf.function()
def test_preprocess_inputs(image_path, mask_path):
    with tf.device('/cpu:0'):
        # image = load_image(image_path) # infraed image input. there for 8 bit input
        image = tf.cast(load_bmp(image_path), tf.float32)  # infraed image input. there for 8 bit input
        print("load image shape:", image.shape)
        mask = load_image(mask_path, mask=True)
        mask = tf.cast(mask > 0, dtype=tf.float32)
        print(image)
        # image, mask = random_scale(image, mask) # random resize
        image = std_norm(image)  # norm before padding and crop_pad
        # image, mask = pad_inputs(image, mask)  # and pad to raw size
        # image, mask = random_crop(image, mask)  #
        image, mask = random_flip(image, mask)
        print("prepro image shape:", image.shape)
        print("prepro mask shape:", mask.shape)

        # image = image[:, :, ::-1] - tf.constant([103.939, 116.779, 123.68]) #  # 將 BGR 圖片轉為 RGB 圖片
        # image = image[:, :, ::-1]   # # 將 BGR 圖片轉為 RGB 圖片
        # image = image[:, :, ::-1] - tf.constant([0.0, 0.0, 0.0])  # # 將 BGR 圖片轉為 RGB 圖片
        return image, mask


train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_masks))
train_dataset = train_dataset.shuffle(1024)
train_dataset = train_dataset.map(map_func=train_preprocess_inputs,
                                  num_parallel_calls=tf.data.experimental.AUTOTUNE)
train_dataset = train_dataset.batch(batch_size=batch_size, drop_remainder=True) # drop reminder... if true batch= 6 otherwise =7
# train_dataset = train_dataset.repeat(1000)
train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

val_dataset = tf.data.Dataset.from_tensor_slices((val_images, val_masks))
val_dataset = val_dataset.shuffle(512)
val_dataset = val_dataset.map(map_func=test_preprocess_inputs,
                              num_parallel_calls=tf.data.experimental.AUTOTUNE)
val_dataset = val_dataset.batch(batch_size=batch_size, drop_remainder=True)
# val_dataset = val_dataset.repeat(1000)
val_dataset = val_dataset.prefetch(tf.data.experimental.AUTOTUNE)

print("train_dataset:", train_dataset)

# evaluation -------------------------------------------->
train_avg_loss = tf.keras.metrics.Mean(name='train_avg_loss')
train_avg_metric = tf.keras.metrics.Mean(name='train_avg_metric')
test_avg_loss = tf.keras.metrics.Mean(name='test_avg_metric')
test_avg_metric = tf.keras.metrics.Mean(name='test_avg_metric')
############################### above is global zone $$$$$$$$$$$$$$$$$$$$$$$$$$$
# for x, y in train_dataset.take(1):
#     print("image range: [%s, %s]"%(np.min(x), np.max(x)))
#     print("mask range: [%s, %s]" % (np.min(y), np.max(y)))
#     print("image shape:", x.shape)
#     print("mask shape:", y.shape)

# @tf.function()
# def learning_rate_fn(epoch):
#     if epoch < 5:
#         return 1e-5
#     elif epoch < 10:
#         return 2e-5
#     elif epoch <= 45:
#         return 1e-5
#     elif epoch > 45:
#         return 5e-6

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
            tf.summary.scalar(name_list[i], value_list[i].result(), step=step)
            # print(value_list[i].result())
            value_list[i].reset_states()  # Clear accumulated values with .reset_states()
            # print(value_list[i].result())
        writer.flush()

@tf.function
def train_step(input_feature, labels, model, optimizer):
    with tf.GradientTape() as tape:
        predictions = model(input_feature)
        train_loss = my_loss(labels, predictions,LOSS_MODE="BCE")
        dice = dice_coef(labels, predictions)
    gradients = tape.gradient(train_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_avg_loss(train_loss)
    train_avg_metric(dice)

@tf.function
def test_step(input_feature, labels):

    predictions = model(input_feature)
    print("prediction shape:", predictions.shape)

    t_loss = my_loss(labels, predictions,LOSS_MODE="BCE")
    t_dice = dice_coef(labels, predictions)
    test_avg_loss(t_loss)
    test_avg_metric(t_dice)

    return predictions



def train_and_checkpoint(train_dataset, model, EPOCHS, opt,
                         train_summary_writer, test_summary_writer, graph_writer=None,
                         ckpt=None, ckp_freq=0, manager=None):
    temp_mae = 100 # mae the less the better

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
        for x, y in train_dataset:
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

            ckpt.step.assign_add(1)

        # val dataset per epoch end
        for x_val, y_val in val_dataset:
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
                             [train_avg_loss, train_avg_metric], epoch)

        write_tb_logs_scaler(test_summary_writer, ["epoch_avg_loss", "epoch_avg_Dice"],
                             [test_avg_loss, test_avg_metric], epoch)
        if epoch == 1:

            # at epoch end write graph of the all the computation model
            with graph_writer.as_default():
                tf.summary.trace_export(
                    name="my_func_trace",
                    step=0,
                    profiler_outdir=os.path.join('logs', 'graph'))







if __name__ == '__main__':

    strategy = tf.distribute.MirroredStrategy()
    # with strategy.scope():
    model = UNet(inChannels=1)
    # plot model graph
    tf.keras.utils.plot_model(model, show_shapes=True, dpi=200, expand_nested=True)
    tb_log_root = "logs"
    ckpt = tf.train.Checkpoint(step=tf.Variable(1), net=model)
    manager = tf.train.CheckpointManager(ckpt, ckp_log_root, max_to_keep=3)

    if not os.path.exists(tb_log_root):
        print("build tensorboard log folder")
        os.makedirs(tb_log_root)
    train_summary_writer = tf.summary.create_file_writer(os.path.join(tb_log_root, 'train')) # tensorboard --logdir /tmp/summaries
    test_summary_writer = tf.summary.create_file_writer(os.path.join(tb_log_root, 'test'))
    graph_writer =  tf.summary.create_file_writer(os.path.join(tb_log_root, 'graph'))

    initial_learning_rate = 0.001
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=100000,
        decay_rate=0.96,
        staircase=True)
    opt = tf.keras.optimizers.Adam(initial_learning_rate)
    train_and_checkpoint(train_dataset, model, EPOCHS, opt=opt, train_summary_writer=train_summary_writer,
                         test_summary_writer=test_summary_writer, graph_writer=graph_writer, ckpt=ckpt, manager=manager)