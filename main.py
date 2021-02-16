"""
A very basic and simple task using T.Keras for building a multi-class CNN.
This code will use the famous CIFAR-10 dataset which consists of 10 different image types.

Data:
CIFAR-10 is a dataset of 50,000 32x32 color training images, labeled over 10 categories, and 10,000 test images.
https://www.cs.toronto.edu/~kriz/cifar.html
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.metrics import classification_report, confusion_matrix


class CifarCNN(object):

    def __init__(self):
        self._x_train = None
        self._y_train = None
        self._x_test = None
        self._y_test = None

        self._y_categorical_train = None
        self._y_categorical_test = None

        self._model = None
        self._loss = None
        self._prediction = None

    def get_dataset(self):
        return (self._x_train, self._y_train), (self._x_test, self._y_test)

    def get_model(self):
        return self._model

    def load_and_normalize_data(self):
        print("\t *** Loading CIFAR10 dataset... ***")

        (self._x_train, self._y_train), (self._x_test, self._y_test) = cifar10.load_data()
        print("Train dataset shape: ", self._x_train.shape)
        print("Individual data shape: ", self._x_train[0].shape)
        print()

        max_value = self._x_train.max()
        self._x_train = self._x_train / max_value
        self._x_test = self._x_test / max_value

        return None

    def convert_labels(self):
        """
        We need to convert the labels into binary class matrix with the same shape for every data.
        Total number of classes = 10.

        :return: None
        """

        print("\t *** Converting the labels to binary matrix... ***")

        print("Labels shape BEFORE conversion: ", self._x_train.shape)
        print("Individual label shape and type BEFORE conversion: ", self._x_train[0].shape)
        print()

        #############################################


        #############################################

        print("Labels shape AFTER conversion: ", self._x_train.shape)
        print("Individual label shape and type AFTER conversion: ", self._x_train[0].shape)
        print()

        return None

    def build_model(self):
        """
        Here, the CNN architecture is built.
        Comments will help you to build the model.
        Please make sure you use the correct data shape, kernel size, and filters.
        For the activation function, relu is suggested.

        :return: None
        """
        if self._model is not None:
            self._model = None

        self._model = Sequential()

        """ Adding the layers """

        #############################################
        #   Convolutional layer

        #   Pooling layer

        #   Convolutional layer

        #   Pooling layer

        #   We need to flatten the images from 32 x 32 to 1024 before the final layer

        #   Adding a hidden layer with 256 neurons

        #   Adding the final layer i.e. the classifier layer
        #   Note that only need 10 neurons (i.e. 10 possible classes) and use a correct activtion function in order to
        #   assign decimal probabilities to each class.

        #   Now we need to compile the model
        #   Use appropriate loss and optimizer. Only use the 'accuracy' for the metrics.

        #############################################

        print()
        print("\t *** Summary of the model: ***")
        self._model.summary()

        return None

    def train_model(self):
        """
        Here, train your model.

        :return: None
        """
        #############################################
        #   add an early stopping parameter to stop the training when the model performance stops improving based on the
        #   loss of the validation set.

        #   Train

        #############################################

        self._loss = pd.DataFrame(self._model.history.history)

        #   Let's take a look at the objective function...
        print()
        print("\t *** Summary of the loss: ***")
        self._loss.head()

        return None

    def plot_accuracy(self):
        self._loss[['accuracy', 'val_accuracy']].plot()
        plt.show()

        return None

    def plot_loss(self):
        self._loss[['loss', 'val_loss']].plot()
        plt.show()

        return None

    def evaluate_training(self):
        print(self._model.metrics_names)
        print(self._model.evaluate(self._x_test, self._y_categorical_test, verbose=0))

        return None

    def evaluate_prediction(self):
        self._prediction = self._model.predict_classes(self._x_test)
        print("\t *** Classification report: ***")
        print(classification_report(self._y_test, self._prediction))

        print("\t *** Confusion matrix: ***")
        confusion_matrix(self._y_test, self._prediction)

        return None

    def plot_confusion_matrix(self):
        plt.figure(figsize=(10, 6))
        sns.heatmap(confusion_matrix(self._y_test, self._prediction), annot=True)
        plt.show()

        return None

    def predict_image(self):
        im = self._x_test[16]

        plt.imshow(im)
        pred_im = self._model.predict_classes(im.reshape(1, 32, 32, 3))

        #   5 is a dog: https://www.cs.toronto.edu/~kriz/cifar.html
        print("\t *** Prediction label = ", pred_im[0])
        plt.show()

        return None


if __name__ == '__main__':
    #   Please run line by line.

    cfr = CifarCNN()
    cfr.load_and_normalize_data()
    cfr.convert_labels()
    cfr.build_model()
    cfr.train_model()
    cfr.plot_accuracy()
    cfr.plot_loss()
    cfr.evaluate_training()
    cfr.evaluate_prediction()
    cfr.plot_confusion_matrix()
    cfr.predict_image()
