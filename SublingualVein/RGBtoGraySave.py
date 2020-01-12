from PIL import Image
from glob import glob
import matplotlib.pyplot as plt
import tensorflow as tf
import os
from PIL import Image
import matplotlib
import imageio

additional_train_images = sorted(glob('I:\\dataset\\infaredSublingualVein\\fromStudent\\raw\\train/*'))

target_path = "I:\\dataset\\infaredSublingualVein\\fromStudent\\raw\\myGrayTrain/"

# img = Image.open('image.png').convert('LA')
# img.save('greyscale.png')


# print(dir(scipy.misc))
for i in range(len(additional_train_images)):
    filename =  additional_train_images[i].split('/')[-1].split('\\')[-1].split('.')[0]
    print("filename:", filename)
    print(additional_train_images[i])
    # img = Image.open(additional_train_images[i]).convert('LA')
    img = tf.io.read_file(additional_train_images[i])
    img = tf.io.decode_bmp(img)
    img =  tf.image.rgb_to_grayscale(img).numpy()
    print(img.shape)
    print(type(img))

    plt.imshow(img.reshape([img.shape[0], img.shape[1]]), cmap='gray')
    save_path = os.path.join(target_path, filename +".png")
    print(target_path)
    print(type(img))
    # im = Image.fromarray(img)
    # im.save(target_path)


    imageio.imwrite(save_path, img)
    plt.show()
