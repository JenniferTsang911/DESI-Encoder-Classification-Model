from tensorflow.keras import layers
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization, Activation, LeakyReLU

import random
import numpy as np
import tensorflow as tf

random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

class Autoencoder(tf.keras.Model):
    def __init__(self, input_dim, latent_dim=250):
        super(Autoencoder, self).__init__()
        self.encoder = tf.keras.Sequential([
            layers.Dense(1500, activation='relu'),
            layers.BatchNormalization(),
            layers.Dense(1000, activation='relu'),
            layers.BatchNormalization(),
            layers.Dense(500, activation='relu'),
            layers.BatchNormalization(),
            layers.Dense(latent_dim, activation='relu'),
            layers.BatchNormalization(),
        ])
        self.decoder = tf.keras.Sequential([
            layers.Dense(500, activation='relu'),
            layers.BatchNormalization(),
            layers.Dense(1000, activation='relu'),
            layers.BatchNormalization(),
            layers.Dense(1500, activation='relu'),
            layers.BatchNormalization(),
            layers.Dense(input_dim, activation='sigmoid'),
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def encode(self, data):
        return self.encoder.predict(data)

def classification(input_dim, l2_lambda=0.001, dropout_rate=0.5):
    model = Sequential()
    model.add(Dense(128, input_dim=input_dim, activation='relu', kernel_regularizer=l2(l2_lambda)))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))
    # model.add(Dense(64, activation='relu', kernel_regularizer=l2(l2_lambda)))
    # model.add(BatchNormalization())
    # model.add(Dropout(dropout_rate))
    # model.add(Dense(32, activation='relu', kernel_regularizer=l2(l2_lambda)))
    # model.add(BatchNormalization())
    # model.add(Dropout(dropout_rate))
    model.add(Dense(6, activation='softmax'))  # 6 classes
    return model
# def classification(input_dim, l2_lambda=0.001, dropout_rate=0.5):
#     model = Sequential()
#     model.add(Conv1D(filters=32, kernel_size=3, padding='same', input_shape=(input_dim, 1)))
#     model.add(BatchNormalization())
#     model.add(Activation('relu'))
#     model.add(MaxPooling1D(pool_size=2))
#
#     model.add(Flatten())
#
#     model.add(Dense(64, kernel_regularizer=l2(l2_lambda)))
#     model.add(BatchNormalization())
#     model.add(Activation('relu'))
#     model.add(Dropout(dropout_rate))
#
#     model.add(Dense(6, activation='softmax'))  # 6 classes
#
#     return model

# def classification(input_dim, num_classes=6, l2_lambda=0.001, dropout_rate=0.5):
#     model = Sequential([
#         Conv1D(filters=32, kernel_size=7, strides=1, padding='same', input_shape=(input_dim, 1)),
#         LeakyReLU(alpha=0.1),
#         BatchNormalization(),
#         Dropout(dropout_rate),  # Add dropout layer
#         MaxPooling1D(pool_size=2, strides=2),
#
#         Conv1D(filters=96, kernel_size=5, padding='same', groups=2, kernel_regularizer=l2(l2_lambda)),  # Add L2 regularization
#         LeakyReLU(alpha=0.1),
#         BatchNormalization(),
#         Dropout(dropout_rate),  # Add dropout layer
#         MaxPooling1D(pool_size=2, strides=2),
#
#         Conv1D(filters=144, kernel_size=3, padding='same', kernel_regularizer=l2(0.01)),  # Add L2 regularization
#         LeakyReLU(alpha=0.1),
#
#         Conv1D(filters=144, kernel_size=3, padding='same', groups=2, kernel_regularizer=l2(l2_lambda)),  # Add L2 regularization
#         LeakyReLU(alpha=0.1),
#
#         Conv1D(filters=96, kernel_size=3, padding='same', groups=2, kernel_regularizer=l2(l2_lambda)),  # Add L2 regularization
#         LeakyReLU(alpha=0.1),
#         BatchNormalization(),
#         Dropout(dropout_rate),  # Add dropout layer
#         MaxPooling1D(pool_size=2, strides=2),
#
#         Flatten(),
#
#         Dense(50, activation='relu', kernel_regularizer=l2(l2_lambda)),  # Add L2 regularization and reduce neurons
#         Dense(num_classes, activation='softmax')
#     ])
#
#     return model
