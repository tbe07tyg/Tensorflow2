from tensorflow import keras
import numpy as np


class ImageHistory(keras.callbacks.Callback):
    def on_batch_end(self, batch, logs={}):
        if batch % self.draw_interval == 0:
            images = []
            labels = []
            for item in self.data:
                image_data = item[0]
                label_data = item[1]
                y_pred = self.model.predict(image_data)
                images.append(y_pred)
                labels.append(label_data)
            image_data = np.concatenate(images, axis=2)
            label_data = np.concatenate(labels, axis=2)
            data = np.concatenate((image_data, label_data), axis=1)
            self.last_step += 1
            self.saveToTensorBoard(data, 'batch',
                                   self.last_step * self.draw_interval)
        return