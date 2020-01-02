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


# H, W = 784, 784
batch_size = 3
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
def preprocess_inputs(image_path, mask_path):
    with tf.device('/cpu:0'):
        # image = load_image(image_path) # infraed image input. there for 8 bit input
        image = load_bmp(image_path)  # infraed image input. there for 8 bit input
        print("load image shape:", image.shape)
        mask = load_image(mask_path, mask=True)
        mask = tf.cast(mask > 0, dtype=tf.float32)

        image, mask = random_scale(image, mask) # random resize
        image = std_norm(image)  # norm before padding and crop_pad
        image, mask = pad_inputs(image, mask)  # and pad to raw size
        image, mask = random_crop(image, mask)  #
        image, mask = random_flip(image, mask)
        print("prepro image shape:", image.shape)
        print("prepro mask shape:", mask.shape)

        # image = image[:, :, ::-1] - tf.constant([103.939, 116.779, 123.68]) #  # 將 BGR 圖片轉為 RGB 圖片
        # image = image[:, :, ::-1]   # # 將 BGR 圖片轉為 RGB 圖片
        # image = image[:, :, ::-1] - tf.constant([0.0, 0.0, 0.0])  # # 將 BGR 圖片轉為 RGB 圖片
        return image, mask


train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_masks))
train_dataset = train_dataset.shuffle(1024)
train_dataset = train_dataset.map(map_func=preprocess_inputs,
                                  num_parallel_calls=tf.data.experimental.AUTOTUNE)
train_dataset = train_dataset.batch(batch_size=batch_size, drop_remainder=True)
# train_dataset = train_dataset.repeat(1000)
train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

val_dataset = tf.data.Dataset.from_tensor_slices((val_images, val_masks))
val_dataset = val_dataset.shuffle(512)
val_dataset = val_dataset.map(map_func=preprocess_inputs,
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

@tf.function()
def learning_rate_fn(epoch):
    if epoch < 5:
        return 1e-5
    elif epoch < 10:
        return 2e-5
    elif epoch <= 45:
        return 1e-5
    elif epoch > 45:
        return 5e-6

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


@tf.function
def train_step(input_feature, labels, model, optimizer):
    with tf.GradientTape() as tape:
        predictions = model(input_feature)
        loss = tf.keras.losses.binary_crossentropy(labels, predictions)
        dice = dice_coef(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_avg_loss(loss)
    train_avg_metric(dice)

def train_and_checkpoint(train_dataset, model, EPOCHS, opt, ckpt=None, ckp_freq=0, manager=None):
    temp_mae = 100 # mae the less the better

    ckpt.restore(manager.latest_checkpoint)
    #
    if manager.latest_checkpoint:
        print("Restored from {}".format(manager.latest_checkpoint))
    else:
        print("Initializing from scratch.")


    for epoch in range(EPOCHS):
        # lr_epoch= epoch

        test_avg_metric_list = []
        batch_count = 0
        for x, y in train_dataset:
            print(x.shape)
            print(y.shape)
            batch_count+=1
            # print(x.shape)
            # print(y.shape)

            train_step(x, y, model, opt)
        #     # load input batch features
        #     train_batch_x, train_batch_y = get_extracted_batch_sequence(batch_records=each_batch)
        #     # print("train_batch_x.shape:", train_batch_x.shape)
        #     # print("train_batch_y.shape:", train_batch_y.shape)
        #     # print(type(train_batch_x))
        #     # print(type(train_batch_y))
        #     # write_tb_logs_image(train_summary_writer, ["input_features"], [train_batch_x], optimizer.iterations, batch_size)
        #
        #
        #     train_step(train_batch_x, train_batch_y, model, optimizer)
        # #
            batch_template = 'Step: {} Epoch {}- Batch[{}/{}], Train Avg Loss: {}, Train Avg dice: {}'
        # #
            print(batch_template.format(int(ckpt.step),
                                        epoch,
                                        batch_count,
                                        7,
                                        train_avg_loss.result(),
                                        train_avg_metric.result()))
        #
        #
        #     if batch==0:
        #         print("write model graph")
        #         tf.summary.trace_on(graph=True, profiler=True)
        #         write_tb_model_graph(train_summary_writer, "trainGraph", 0, tb_log_root)
        # for (test_batch, each_batch) in enumerate(test_dataset):  # validation after one epoch training
        #     # load input batch features
        #     test_batch_x, test_batch_y = get_extracted_batch_sequence(batch_records=each_batch)
        #
        #     test_step(test_batch_x, test_batch_y)
        #     batch_template = 'Epoch {} - Batch[{}/{}], test Avg Loss: {}, test Avg MAE: {}'
        #     test_avg_metric_list.append(test_avg_metric.result())
        #     print(batch_template.format(int(ckpt.step),
        #                                 test_batch + 1,
        #                                 test_total_Batches,
        #                                 test_avg_loss.result(),
        #                                 test_avg_metric.result()))
        # test_avg_metric_e=  sum(test_avg_metric_list)/len(test_avg_metric_list)
        # template = 'Validation Epoch {}, Train Avg Loss: {}, Train Avg MAE: {}, Test Avg Loss: {}, Test Avg MAE: {}'
        # print(template.format(int(ckpt.step),
        #                       train_avg_loss.result(),
        #                       train_avg_metric.result() ,
        #                       test_avg_loss.result(),
        #                       test_avg_metric_e))
        # #
        #
        # #
        # if int(ckpt.step) % ckpt_freq == 0 and test_avg_metric_e <temp_mae:
        #     print("save model...")
        #     temp_mae = test_avg_metric.result()
        #     save_path = manager.save()
        #     print("Saved checkpoint for epoch {}: {}".format(int(ckpt.step), save_path))
        #     print("Where test mae {:1.2f} ".format(test_avg_metric_e))
        #
        #
        # if tf.equal(optimizer.iterations % log_freq, 0):
        #     print("writing logs to tensorboard")
        #     # write train logs # with the same name for train and test write will write multiple curves into one plot
        #     write_tb_logs_scaler(train_summary_writer, ["avg_loss", "avg_MAE"],
        #                          [train_avg_loss, train_avg_metric], optimizer.iterations // log_freq)
        #
        #     write_tb_logs_scaler(test_summary_writer, ["avg_loss", "avg_MAE"],
        #                          [test_avg_loss, test_avg_metric], optimizer.iterations // log_freq)

            ckpt.step.assign_add(1)



if __name__ == '__main__':

    strategy = tf.distribute.MirroredStrategy()
    # with strategy.scope():
    model = UNet(inChannels=1)

    ckpt = tf.train.Checkpoint(step=tf.Variable(1), net=model)
    manager = tf.train.CheckpointManager(ckpt, ckp_log_root, max_to_keep=3)

    # lr_epoch = tf.Variable(1)
    lr_schedule = tf.keras.optimizers.schedules.LearningRateSchedule(learning_rate_fn)
    opt = tf.keras.optimizers.Adam(lr_schedule)
    train_and_checkpoint(train_dataset, model, EPOCHS, opt=opt, ckpt=ckpt, manager=manager)