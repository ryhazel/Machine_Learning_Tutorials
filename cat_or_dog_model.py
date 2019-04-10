import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random
import pickle

DATADIR = "/Users/rhazel/Documents/kagglecatsanddogs_3367a/PetImages"
CATEGORIES = ["Dog", "Cat"]

for category in CATEGORIES:   # iterates through each of the 2 categories
    path = os.path.join(DATADIR, category)  # path to cats or dogs dir
    for img in os.listdir(path):  # in each category iterate through each img
        img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)  # convert img to array and grayscale
        '''plt.imshow(img_array, cmap="gray")
        plt.show()
        print(img_array.shape)'''

IMG_SIZE = 50


training_data = []


def create_training_data():
    for category in CATEGORIES:  # iterates through each of the 2 categories
        path = os.path.join(DATADIR, category)  # path to cats or dogs dir
        class_num = CATEGORIES.index(category)  # the class number (0 or 1 for dog or cat) is the index of CATEGORIES
        for img in os.listdir(path):  # in each category iterate through each img
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)  # convert img to array and grayscale
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # make all of the images the same size
                training_data.append([new_array, class_num])  # create a list called training_data that has new array and the class number
            except Exception as e:
                pass


create_training_data()
random.shuffle(training_data)

'''for sample in training_data[:10]:
    print(sample[1])'''

X = []
Y = []

for features, label in training_data:
    X.append(features)
    Y.append(label)

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
Y = np.array(Y)

pickle_out = open("X.pickle", "wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("Y.pickle", "wb")
pickle.dump(Y, pickle_out)
pickle_out.close()

pickle_in = open("X.pickle", "rb")
X = pickle.load(pickle_in)










