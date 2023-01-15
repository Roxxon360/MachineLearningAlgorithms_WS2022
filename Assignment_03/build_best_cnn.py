import numpy as np
import pandas as pd
import tensorflow as tf
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import BatchNormalization, RandomFlip, RandomRotation
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator

def build_best_model(X_train, X_test, y_train, y_test):

    #data_augmentation = tf.keras.Sequential(RandomRotation(factor=(-0.1, 0.1)))

    model = Sequential()

    #model.add(data_augmentation)
    model.add(Conv2D(64, kernel_size=3, activation = 'relu', input_shape = (28, 28, 1)))
    model.add(Conv2D(64, kernel_size=3, activation = 'relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(BatchNormalization())
    model.add(Conv2D(128, kernel_size=3, activation = 'relu'))
    model.add(MaxPooling2D(pool_size=2))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10))
    model.add(Activation('softmax'))

    opt = tf.keras.optimizers.Adadelta(learning_rate=0.6)

    model.compile(loss = 'categorical_crossentropy', optimizer=opt, metrics =['accuracy'])

    model.fit(X_train, y_train, validation_split = 0.1, epochs = 10, batch_size = 32)

    model.evaluate(X_test, y_test)

if __name__ == "__main__":

    data = pd.read_csv('alldigits.csv').astype('uint8')

    X = data.iloc[:,:-1].values.reshape(len(data), 28, 28, 1)
    X = np.array(X)
    X = X/255.0

    y = data.iloc[:,-1].values
    labels = to_categorical(y, num_classes = 10)
    labels = np.array(labels)

    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.1, random_state=42)

    build_best_model(X_train, X_test, y_train, y_test)