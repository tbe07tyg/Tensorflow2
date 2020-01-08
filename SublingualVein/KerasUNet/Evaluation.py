import tensorflow as tf
from SublingualVein.LearnFromPersonSegStructure.ModelDESIGN import U_NetV2
import math, os
from glob import glob
from SublingualVein.KerasUNet.preprocessing import resize, std_norm, random_flip
from SublingualVein.KerasUNet.Ultis import display

print('TensorFlow', tf.__version__)
os.environ['TF_XLA_FLAGS'] = '--tf_xla_cpu_global_jit'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

os.environ["PATH"] += os.pathsep + 'I:/DeepLearning/TensorflowV2/graphviz-2.38/release/bin/'

# H, W = 784, 784
batch_size = 2
EPOCHS = 2000
log_freq = 1
ckp_log_root = "ckpts"


# train_images = sorted(glob('resized_images/*'))
# train_masks = sorted(glob('resized_masks/*'))
#
# val_images = sorted(glob('validation_data/images/*'))
# val_masks = sorted(glob('validation_data/masks/*'))

train_images = sorted(glob('I:/dataset/infaredSublingualVein/train/raw_image/*'))
train_masks = sorted(glob('I:/dataset/infaredSublingualVein/train/tongue_labels/*'))

val_images = sorted(glob('I:/dataset/infaredSublingualVein/validation/raw_image/*'))
val_masks = sorted(glob('I:/dataset/infaredSublingualVein/validation/tongue_labels/*'))



print(f'Found {len(val_images)} validation images')
print(f'Found {len(val_masks)} validation masks')

total_num_batches_per_epoch = math.ceil(len(val_images) / batch_size)
print("batch size:", batch_size)
print("total_num_batches per epoch:", total_num_batches_per_epoch)

for i in range(len(val_masks)):
    assert val_images[i].split('/')[-1].split('\\')[-1].split('.')[0] \
           == val_masks[i].split('/')[-1].split('\\')[-1].split('.')[0]


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
def test_preprocess_inputs(image_path, mask_path):
    with tf.device('/cpu:0'):
        # image = load_image(image_path) # infraed image input. there for 8 bit input
        image = tf.cast(load_bmp(image_path), tf.float32)  # infraed image input. there for 8 bit input
        print("load image shape:", image.shape)
        mask = load_image(mask_path, mask=True)
        mask = tf.cast(mask > 0, dtype=tf.float32)
        image, mask = resize(image, mask)
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

val_dataset = tf.data.Dataset.from_tensor_slices((val_images, val_masks))
val_dataset = val_dataset.shuffle(512)
val_dataset = val_dataset.map(map_func=test_preprocess_inputs,
                              num_parallel_calls=tf.data.experimental.AUTOTUNE)
val_dataset = val_dataset.batch(batch_size=batch_size, drop_remainder=True)
# val_dataset = val_dataset.repeat(1000)
val_dataset = val_dataset.prefetch(tf.data.experimental.AUTOTUNE)

def create_mask(pred_mask):
  pred_mask = tf.argmax(pred_mask, axis=-1)
  pred_mask = pred_mask[..., tf.newaxis]
  return pred_mask[0]

if __name__ == '__main__':

    # use designed model
    model = U_NetV2(inChannels=1)
    #
    ckpt = tf.train.Checkpoint(step=tf.Variable(1), net=model)
    manager = tf.train.CheckpointManager(ckpt, ckp_log_root, max_to_keep=3)

    ckpt.restore(manager.latest_checkpoint)

    for x_val, y_val in val_dataset:
        # print("x_val.shape:", x_val.shape)
        # print("y_val.shape:", y_val.shape)
        predictions = model.predict(x_val)

        display([x_val[0], y_val[0], create_mask(predictions)])




