import random
from tensorflow import keras
import wandb
from wandb.keras import WandbCallback
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Flatten, Dense, Concatenate, Dot, Lambda, Input
from tensorflow.keras.datasets import mnist
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255 #  nomrlize input pixel value
print("x_train shape:", x_train.shape)
print("y_train shape:", y_train.shape)
print("y", y_train)


# this funciton is tryting to make training samples
def make_pairs(x, y):
    print("max(y)", max(y))
    num_classes = max(y) + 1

    #     for i in range(num_classes):
    #         print(np.where(y==i))  # find the indice where the condition fullfil

    digit_indices = [np.where(y == i)[0] for i in range(num_classes)]
    # markdown the indices for each digital class, formulate class list with samples as  [class1, class2, class3, ..]
    # each class in the list is the sample indices in the raw training dataset
    #     print("digit_indices:", digit_indices)
    pairs = []
    labels = []

    for idx1 in range(len(x)):  # len(x) number of samples
        # add a matching example  ----------->
        x1 = x[idx1]
        label1 = y[idx1]  # got the specified label
        #  generate random choice index
        idx2 = random.choice(digit_indices[label1])  # for the specified label， random choose one element
        x2 = x[idx2]  # find corresponding input x with the chosen training sample index

        pairs += [[x1,
                   x2]]  # pair the input samples x ,added to the training pair list, at the same time attach the label (the x1 and x2 could be the same x or different),  regards it as positive(P)
        labels += [1]

        # add a not matching example -------------->
        label2 = random.randint(0, num_classes - 1)  # RANDOMly generate digital label from [0, 9]
        while label2 == label1:  # if label 2 == label 1
            label2 = random.randint(0, num_classes - 1)  # regenerate the digital label

        idx2 = random.choice(digit_indices[
                                 label2])  # once come to herem the label 2 is not equal to label 1, choose one element from not match label
        x2 = x[idx2]  # choose the x # find the corresponding training sample x

        pairs += [[x1, x2]]  # pair the x1, x2 which have different labels(not match)
        labels += [0]  # regard not match case with label 1

    return np.array(pairs), np.array(
        labels)  # return the match and not match training sample pairs and corresonding label (match = 1; not match = 0)


pairs_train, labels_train = make_pairs(x_train, y_train)
pairs_test, labels_test = make_pairs(x_test, y_test)

# take a peek at the data for matching case
fig, axes = plt.subplots(2, 2)
match_index =  random.choice(np.where(labels_train==1)[0]) # label ==1 means match random select one
print(np.where(labels_train==1))
print("match_indx:", match_index)
no_match_indx =  random.choice(np.where(labels_train==0)[0]) # label ==0 means no match  random select one
print("no match_indx:", no_match_indx)
# random choce one match sample and one unmatchs ample
axes[0, 0].imshow(pairs_train[match_index,0])
axes[0, 0].set_title("match x1")
axes[0, 1].imshow(pairs_train[match_index,1])
axes[0, 1].set_title("match x2")

axes[1, 0].imshow(pairs_train[no_match_indx,0])
axes[1, 0].set_title("no match x1")
axes[1, 1].imshow(pairs_train[no_match_indx,1])
axes[1, 1].set_title("no match x2")
left = 0.125  # the left side of the subplots of the figure
right = 0.9   # the right side of the subplots of the figure
bottom = 0.1  # the bottom of the subplots of the figure
top = 0.9     # the top of the subplots of the figure
wspace = 0.2  # the amount of width reserved for space between subplots,
              # expressed as a fraction of the average axis width
hspace = 0.5
plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)

seq1 = Sequential()
seq1.add(Flatten(input_shape=(28,28)))
seq1.add(Dense(128, activation='relu'))

seq2 = Sequential()
seq2.add(Flatten(input_shape=(28,28)))
seq2.add(Dense(128, activation='relu'))

merge_layer = Concatenate()([seq1.output, seq2.output])
dense_layer = Dense(1, activation="sigmoid")(merge_layer)
model = Model(inputs=[seq1.input, seq2.input], outputs=dense_layer)

model.compile(loss = "binary_crossentropy", optimizer="adam", metrics=["accuracy"])
model.summary()

wandb.init(project="siamesetry")
model.fit([pairs_train[:,0], pairs_train[:,1]], labels_train[:], batch_size=16, epochs= 10, callbacks=[WandbCallback()])