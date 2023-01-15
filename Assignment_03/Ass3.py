import keras_tuner
import numpy as np
import tensorflow as tf
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow import keras
import keras_tuner as kt
from keras import datasets

# Data Import

file_name = "alldigits.csv"
df = pd.read_csv(file_name)
data = df.to_numpy()

X = data[:, :-1]
y = data[:, -1]

X_temp = []
for picture in X:
    X_temp.append(picture.reshape(28, 28).transpose())

X = np.array(X_temp) / 255

# Test/Train/Val Split
test_size = 0.3
val_size = 0.2
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=20)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_size / (1 - test_size),
                                                  random_state=15)

# ----------------------------------Hyperparameter-Optimization--------------------------------

hp = kt.HyperParameters()


def build_model(hp):
    model = keras.Sequential()
    model.add(keras.layers.Conv2D(hp.Int(32, 96, 32), (3, 3), activation="relu", input_shape=(28, 28, 1)))
    model.add(keras.layers.MaxPooling2D((2, 2)))
    model.add(keras.layers.Conv2D(hp.Int(32,64,32), (3, 3), activation='relu'))
    model.add(keras.layers.MaxPooling2D((2, 2)))
    model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
    if hp.Boolean("b1"):
        model.add(keras.layers.Conv2D(64, (3, 3)))
        model.add(keras.layers.MaxPooling2D((2, 2)))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(20, activation='relu'))
    model.add(keras.layers.Dense(10))
    # model.add(keras.layers.Softmax())
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    return model


tuner = keras_tuner.RandomSearch(hypermodel=build_model, objective="val_loss", max_trials=5)
tuner.search(X_train, y_train, epochs=5, validation_data=(X_val, y_val))


# ----------------------------Final Model-------------------------------------------------------

finalModel = None

model = keras.Sequential()
model.add(keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)))
model.add(keras.layers.MaxPooling2D((2, 2)))
model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(keras.layers.MaxPooling2D((2, 2)))
model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(20, activation='relu'))
model.add(keras.layers.Dense(10))
# model.add(keras.layers.Softmax())
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10,
                       validation_data=(X_test, y_test))
finalModel = model