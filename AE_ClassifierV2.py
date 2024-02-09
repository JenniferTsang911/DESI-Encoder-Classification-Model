from AE_ClassifierV2 import *
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import f1_score
from tensorflow.keras.callbacks import ProgbarLogger, TensorBoard, LearningRateScheduler
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import EarlyStopping
import seaborn as sns
import matplotlib.pyplot as plt

import random
import numpy as np
import tensorflow as tf

random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

import os
import datetime

def scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * tf.math.exp(-0.1)
def train_autoencoder(model, train_data, val_data, epochs, batch_size, learning_rate=0.001):
    progress_bar = ProgbarLogger(count_mode='steps')
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mean_squared_error')

    # Set up TensorBoard
    log_dir = os.path.join(
        "logs",
        "fit",
        datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
    )
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

    # Set up EarlyStopping
    early_stopping_callback = EarlyStopping(monitor='val_loss', patience=int(0.1*epochs), restore_best_weights=True)

    model.fit(train_data, train_data,
              validation_data=(val_data, val_data),
              epochs=epochs, batch_size=batch_size,
              callbacks=[progress_bar, tensorboard_callback, early_stopping_callback])  # Add the EarlyStopping callback here

    model.save('autoencoder_model_unbalanced_with_tissues', save_format='tf')
    return model.encoder.predict(train_data), model.encoder.predict(val_data)

def train_classifier(
        encoded_train_data, encoded_val_data, train_labels, val_labels, epochs, batch_size, learning_rate=0.001, l2_lambda = 0.01, dropout_rate=0.5):
    classifier = classification(encoded_train_data.shape[1], l2_lambda, dropout_rate)
    optimizer= Adam(learning_rate=learning_rate)
    classifier.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Set up TensorBoard
    log_dir = os.path.join(
        "logs",
        "fit",
        datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
    )
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

    # Set up EarlyStopping
    early_stopping_callback = EarlyStopping(monitor='val_loss', patience=int(0.1*epochs), restore_best_weights=True)

    classifier.fit(encoded_train_data, train_labels,
                   validation_data=(encoded_val_data, val_labels),
                   epochs=epochs, batch_size=batch_size,
                   callbacks=[tensorboard_callback, early_stopping_callback]
                   )  # Add the callbacks here

    return classifier
def evaluate_model(model, test_data, test_labels, class_names):
    # Predict the classes
    predictions = model.predict(test_data)
    predicted_classes = np.argmax(predictions, axis=1)

    # Convert the labels to class names
    test_class_names = [class_names[label] for label in test_labels]
    predicted_class_names = [class_names[label] for label in predicted_classes]

    # Calculate the F1 score
    f1 = f1_score(test_class_names, predicted_class_names, average='macro')

    # Print the F1 score
    print("F1 Score: ", f1)

    # Print the classification report
    print("Classification Report:\n", classification_report(test_class_names, predicted_class_names))

    # Print the confusion matrix
    print("Confusion Matrix:\n", confusion_matrix(test_class_names, predicted_class_names))

    # Generate the confusion matrix
    cm = confusion_matrix(test_class_names, predicted_class_names)

    # Normalize the confusion matrix
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Plot the normalized confusion matrix
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt=".2f", xticklabels=class_names, yticklabels=class_names, cmap='Blues')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

