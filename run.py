from data_preprocess import *
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from AE_ClassifierV2 import *
import random

random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

base_path = ''

train_data, train_labels, val_data, val_labels, test_data, test_labels, class_names = process_data(base_path, balanced=True, should_shuffle=True)


learning_rate = 0.001
input_dim = train_data.shape[1]  # P
latent_dim = 250  # or any other number you want
epochs_ae = 1000
batch_size_AE = 128

epochs = 500
batch_size = 128
learning_rateCL = 0.001
l2_lambda = 0.001
dropout_rate = 0.2

# autoencoder = Autoencoder(input_dim, latent_dim)
# encoded_train_data, encoded_val_data = train_autoencoder(
#      autoencoder, train_data, val_data, epochs_ae, batch_size_AE, learning_rate)

autoencoder = tf.keras.models.load_model('autoencoder_model')
encoded_train_data = autoencoder.encoder.predict(train_data)
encoded_val_data = autoencoder.encoder.predict(val_data)

classifier = train_classifier(
    encoded_train_data, encoded_val_data, train_labels, val_labels,
    epochs,
    batch_size,
    learning_rate,
    l2_lambda,
    dropout_rate
    )

encoded_test_data = autoencoder.encoder.predict(test_data)

evaluate_model(classifier, encoded_test_data, test_labels, class_names)
