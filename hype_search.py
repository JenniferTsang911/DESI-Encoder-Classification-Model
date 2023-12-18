from data_preprocess import *
import random
import numpy as np
import tensorflow as tf

random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

import numpy as np
from AE_ClassifierV2 import train_classifier
from sklearn.metrics import f1_score
import itertools

def grid_search(encoded_train_data, train_labels, encoded_val_data, val_labels, encoded_test_data, test_labels, param_grid):
    best_score = -np.inf
    best_params = None

    # Generate all combinations of hyperparameters
    all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]

    for i, params in enumerate(all_params):
        print(f"Iteration {i+1}/{len(all_params)}: {params}")

        # Train the model with the current hyperparameters
        model = train_classifier(
            encoded_train_data, encoded_val_data, train_labels, val_labels,
            epochs=500,
            batch_size=params['batch_size'],
            learning_rate=params['learning_rate'],
            l2_lambda=params['l2_lambda'],
            dropout_rate=params['dropout_rate']
        )

        # Predict the classes of the validation data
        y_pred = model.predict(encoded_test_data)
        y_pred = np.argmax(y_pred, axis=1)

        # Calculate the F1 score
        score = f1_score(test_labels, y_pred, average='macro')

        print(score)

        # Save the progress to a .txt file
        with open('iterations_ran.txt', 'a') as f:
            f.write(f"Iteration {i+1}/{len(all_params)}: F1 score = {score}, params = {params}\n")

        # If the score is better than the current best, update the best score and best parameters
        if score > best_score:
            best_score = score
            best_params = params

    return best_params, best_score
#Define the parameter grid
param_grid = {
    'learning_rate': [0.0001, 0.001, 0.01, 0.1],
    'dropout_rate': [0.1, 0.2, 0.3, 0.4, 0.5],
    'l2_lambda': [0.0001, 0.001, 0.01, 0.1, 1],
    'batch_size': [64, 128, 256, 512, 1024]
}

train_data, train_labels, val_data, val_labels, test_data, test_labels, class_names = process_data(
    r"C:\Users\jenni\Documents\GitHub\DESI-project\all_aligned_no_background_others_preprocessed.csv", balanced=True, should_shuffle=True)

autoencoder = tf.keras.models.load_model('autoencoder_model_unbalanced')
encoded_train_data = autoencoder.encoder.predict(train_data)
encoded_val_data = autoencoder.encoder.predict(val_data)
encoded_test_data = autoencoder.encoder.predict(test_data)

# Call the function
best_params, best_score = grid_search(encoded_train_data, train_labels, encoded_val_data, val_labels, encoded_test_data, test_labels, param_grid)

print(f'Best F1 score: {best_score}, best params: {best_params}')