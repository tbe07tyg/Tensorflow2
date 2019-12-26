from SublingualVein.KerasUNet.CustomGenerator import DataGen
from SublingualVein.KerasUNet.ModelDESIGN import UNet
from SublingualVein.KerasUNet.hyperparameters import batch_size, train_input_image_path, train_vein_mask_path, \
    train_tongue_mask_path, val_input_image_path, val_tongue_mask_path, val_vein_mask_path, epoches, image_size
from SublingualVein.KerasUNet.Ultis import display




# Model Compile
model = UNet()
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["acc"])
model.summary()


# DataGenerator

Train_Gen =  DataGen(batch_size,  train_input_image_path, image_size=image_size, name="train", vein_mask_path=train_vein_mask_path,
                     tongue_mask_path=train_tongue_mask_path, shuffle=True, mode="tongue")
Val_Gen =  DataGen(batch_size, val_input_image_path, image_size=image_size, name="val", vein_mask_path=val_vein_mask_path,
                   tongue_mask_path=val_tongue_mask_path, shuffle=True, mode="tongue")

# # check the input and label
# for image, mask in Train_Gen:
#     for i in range(image.shape[0]):
#         print(image[i].shape)
#         print(mask[i].shape)
#         print(image[i].reshape((image_size, image_size,3)).shape)
#         display([image[i].reshape(image_size, image_size,3), mask[i].reshape(image_size, image_size)])

# Fit data and train the model
train_steps =  Train_Gen.__len__()
val_steps =  Val_Gen.__len__()
# print("batch size:", batch_size)
# print("train input len：", len(Train_Gen.data_pairs))
# print("num of train batches:", train_steps)
# print("val input len：", len(Val_Gen.data_pairs))
# print("num of val batches:", val_steps)

model.fit_generator(Train_Gen, validation_data=Val_Gen, steps_per_epoch=train_steps, validation_steps=val_steps,
                    epochs=epoches)