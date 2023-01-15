import numpy as np
import pandas as pd
import tensorflow as tf
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from sklearn.model_selection import train_test_split
import keras_tuner

def build_model(hp):

    model = tf.keras.Sequential()

    previous_layer_filters = hp.Int('filter0', min_value=64, max_value=128, step=32)

    model.add(Conv2D(previous_layer_filters, 
                    kernel_size=5, 
                    activation = 'relu', 
                    input_shape = (28, 28, 1)))
    model.add(MaxPooling2D(pool_size=3))

    layers = hp.Int('n_layers', min_value=2, max_value=3, step=1)

    for i in range(layers):
            previous_layer_filters = hp.Int(f'filter{i+1}', min_value = previous_layer_filters, max_value=256, step=32)
            model.add(Conv2D(previous_layer_filters, 
                            kernel_size=hp.Int(f'kernel{i+1}', min_value=3, max_value=5, step=2), 
                            activation = 'relu'))
            
    model.add(MaxPooling2D(pool_size=2))

    model.add(Flatten())
    model.add(Dense(hp.Int('dense', min_value=128, max_value=256, step=64)))
    model.add(Activation('relu'))
    model.add(Dropout(hp.Float('dropout', min_value = 0.25, max_value = 0.5, step = 0.25)))
    model.add(Dense(10))
    model.add(Activation('softmax'))

    hp_lr_adadelta = hp.Choice('learning_rate_adadelta', values=[1.5, 1.0, 0.5])

    optimizer = tf.keras.optimizers.Adadelta(learning_rate=hp_lr_adadelta)

    model.compile(loss = 'categorical_crossentropy', optimizer=optimizer, metrics =['accuracy'])

    return model

if __name__ == "__main__":
    data = pd.read_csv('alldigits.csv').astype('uint8')

    X = data.iloc[:,:-1].values.reshape(len(data), 28, 28, 1)
    X = np.array(X)
    X = X/255.0

    y = data.iloc[:,-1].values
    labels = to_categorical(y, num_classes = 10)
    labels = np.array(labels)

    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.1, random_state=42)

    tuner_rs = keras_tuner.RandomSearch(build_model, objective='val_accuracy', max_trials=50, overwrite = False, project_name='RandomSearch2', directory = 'ass3')
    tuner_rs.search(X_train, y_train, epochs=10, validation_data = (X_test, y_test))

    '''
    'values': {'filter0': 96, 'kernel0': 5, 'n_layers': 2, 'filter1': 96, 'kernel1': 5, 'dropout_1': False, 'dense': 192, 'dropout_2': True, 'learning_rate_adadelta': 1.0, 
    'learning_rate_adam': 0.0001, 'optimizer': 'adadelta', 'filter2': 160, 'kernel2': 3, 'dropout2': 0.5, 'dropout1': 0.25}

    score: 0.9850000143051147
    '''

    tuner_bo = keras_tuner.BayesianOptimization(build_model, objective='val_accuracy', max_trials=50, overwrite = False, project_name='BayesianOptimization2', directory = 'ass3')
    tuner_bo.search(X_train, y_train, epochs=10, validation_data = (X_test, y_test))

    '''
    'values': {'filter0': 64, 'kernel0': 5, 'n_layers': 1, 'filter1': 64, 'kernel1': 3, 'dropout_1': False, 'dense': 256, 'dropout_2': False, 
    'learning_rate_adadelta': 1.0, 'learning_rate_adam': 0.0001, 'optimizer': 'adadelta', 'filter2': 256, 'kernel2': 5, 'dropout1': 0.5, 'dropout2': 0.25

    score: 0.9850000143051147
    '''

    tuner_hb = keras_tuner.Hyperband(build_model, objective='val_accuracy', max_epochs=15, overwrite = False, project_name='Hyperband2', directory = 'ass3')
    tuner_hb.search(X_train, y_train, epochs=10, validation_data = (X_test, y_test))

    '''
    'values': {'filter0': 64, 'kernel0': 3, 'n_layers': 2, 'filter1': 256, 'kernel1': 5, 'dropout_1': True, 'dense': 128, 'dropout_2': False, 
    'learning_rate_adadelta': 0.1, 'learning_rate_adam': 0.001, 'optimizer': 'adam', 'tuner/epochs': 12, 'tuner/initial_epoch': 4, 
    'tuner/bracket': 2, 'tuner/round': 2, 'filter2': 256, 'kernel2': 3, 'dropout1': 0.25

    score: 0.9783333539962769
    '''

