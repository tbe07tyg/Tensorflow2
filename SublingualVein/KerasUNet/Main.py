from SublingualVein.KerasUNet.CustomGenerator import DataGen
from SublingualVein.KerasUNet.ModelDESIGN import UNet
from SublingualVein.KerasUNet.hyperparameters import batch_size, train_input_image_path, train_vein_mask_path, \
    train_tongue_mask_path, val_input_image_path, val_tongue_mask_path, val_vein_mask_path, epoches, image_size
import tensorflow as tf
from SublingualVein.KerasUNet.Ultis import display, create_mask

# from IPython.display import clear_output
import copy


# Model Compile
model = UNet()
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=[tf.keras.metrics.MeanIoU(num_classes=2)])
model.summary()



# DataGenerator

Train_Gen =  DataGen(batch_size,  train_input_image_path, image_size=image_size, name="train", vein_mask_path=train_vein_mask_path,
                     tongue_mask_path=train_tongue_mask_path, shuffle=True, mode="tongue")
Val_Gen =  DataGen(batch_size, val_input_image_path, image_size=image_size, name="val", vein_mask_path=val_vein_mask_path,
                   tongue_mask_path=val_tongue_mask_path, shuffle=True, mode="tongue")


# create display callback
check_Gen = copy.copy(Train_Gen)
# check the input and label
for image, mask in check_Gen:
    for i in range(image.shape[0]):
        sample_image = image[i].reshape(image_size, image_size,3)
        sample_mask = mask[i].reshape(image_size, image_size)
        print("sample image shape:", sample_image.shape)
        print("sample_mask shape:", sample_mask.shape)

        display([sample_image, sample_mask])
    break


def show_predictions(dataset=None, num=1):
  if dataset:
    for image, mask in dataset.take(num):
      pred_mask = model.predict(image)
      display([image[0], mask[0], create_mask(pred_mask)])
  else:
    display([sample_image, sample_mask,
             create_mask(model.predict(sample_image[tf.newaxis, ...]))])

class DisplayCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs=None):
    # clear_output(wait=True)
    show_predictions()
    print ('\nSample Prediction after epoch {}\n'.format(epoch+1))

# Fit data and train the model
train_steps =  Train_Gen.__len__()
val_steps =  Val_Gen.__len__()
# print("batch size:", batch_size)
# print("train input len：", len(Train_Gen.data_pairs))
# print("num of train batches:", train_steps)
# print("val input len：", len(Val_Gen.data_pairs))
# print("num of val batches:", val_steps)

model.fit_generator(Train_Gen, validation_data=Val_Gen, steps_per_epoch=train_steps, validation_steps=val_steps,
                    epochs=epoches,callbacks=[DisplayCallback()])