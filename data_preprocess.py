import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.utils import class_weight
from sklearn.utils import shuffle
from imblearn.over_sampling import RandomOverSampler
from collections import Counter

import random
import numpy as np
import tensorflow as tf

random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

def process_data(input_file, balanced=False, should_shuffle=False):
    # Load the CSV file
    df = pd.read_csv(input_file)
    df = df.loc[df['Class'] != 'tissue']

    # Encode the labels
    le = LabelEncoder()
    df['Class'] = le.fit_transform(df['Class'])

    # Save the class names
    class_names = le.classes_

    # Get a list of unique slide names
    slide_names = df['Slide'].unique()

    # Split the slide names into training and temporary sets
    train_slide_names, temp_slide_names = train_test_split(slide_names, test_size=0.3, random_state=42)

    # Split the temporary set into testing and validation sets
    test_slide_names, val_slide_names = train_test_split(temp_slide_names, test_size=0.2, random_state=42)

    # Create the training, validation, and test DataFrames
    train_df = df[df['Slide'].isin(train_slide_names)]
    val_df = df[df['Slide'].isin(val_slide_names)]
    test_df = df[df['Slide'].isin(test_slide_names)]

    print("Training slides:", train_df['Slide'].unique())
    print("Validation slides:", val_df['Slide'].unique())
    print("Test slides:", test_df['Slide'].unique())

    # Reset the index of the DataFrames
    train_df.reset_index(drop=True, inplace=True)
    val_df.reset_index(drop=True, inplace=True)
    test_df.reset_index(drop=True, inplace=True)

    if balanced == True:
        # For train_df
        unique_classes_train = train_df.iloc[:, 1].unique()
        class_dataframes_train = {class_name: train_df[train_df.iloc[:, 1] == class_name] for class_name in unique_classes_train}
        min_samples_train = min(len(class_df) for class_df in class_dataframes_train.values())
        for class_name, class_df in class_dataframes_train.items():
            if len(class_df) > min_samples_train:
                class_dataframes_train[class_name] = class_df.sample(min_samples_train)
        train_df = pd.concat(class_dataframes_train.values())
    #
    #     # For val_df
    #     unique_classes_val = val_df.iloc[:, 1].unique()
    #     class_dataframes_val = {class_name: val_df[val_df.iloc[:, 1] == class_name] for class_name in unique_classes_val}
    #     min_samples_val = min(len(class_df) for class_df in class_dataframes_val.values())
    #     for class_name, class_df in class_dataframes_val.items():
    #         if len(class_df) > min_samples_val:
    #             class_dataframes_val[class_name] = class_df.sample(min_samples_val)
    #     val_df = pd.concat(class_dataframes_val.values())

    # Separate the features and labels in the training, validation, and test dataframes
    train_features = train_df[df.columns[4:]]
    train_labels = train_df['Class']
    val_features = val_df[df.columns[4:]]
    val_labels = val_df['Class']
    test_features = test_df[df.columns[4:]]
    test_labels = test_df['Class']

    # if balanced == True:
    #     # Count the number of samples in each class
    #     class_counts = Counter(train_labels.values)
    #
    #     # Find the maximum number of samples
    #     max_samples = max(class_counts.values())
    #
    #     # Create a dictionary that maps each class label to the desired number of samples
    #     # For minority classes, set the number of samples to double the current number
    #     sampling_strategy = {class_label: max_samples if count < max_samples else count * 2 for class_label, count in class_counts.items()}
    #
    #     ros = RandomOverSampler(sampling_strategy=sampling_strategy, random_state=42)
    #     train_features, train_labels = ros.fit_resample(train_features, train_labels)

    if should_shuffle == True:
        # Shuffle the training data
        train_features, train_labels = shuffle(train_features, train_labels)


    # Convert the features to TensorFlow tensors
    train_data = tf.convert_to_tensor(train_features.values, dtype=tf.float32)
    val_data = tf.convert_to_tensor(val_features.values, dtype=tf.float32)
    test_data = tf.convert_to_tensor(test_features.values, dtype=tf.float32)

    train_labels = tf.convert_to_tensor(train_labels.values, dtype=tf.int32)
    val_labels = tf.convert_to_tensor(val_labels.values, dtype=tf.int32)
    test_labels = tf.convert_to_tensor(test_labels.values, dtype=tf.int32)

    # class_weights = class_weight.compute_class_weight(
    # class_weight="balanced",
    # classes=np.unique(train_labels.numpy()),
    # y=train_labels.numpy()
    # )
    # class_weights = dict(enumerate(class_weights))  # Convert class_weights to a dictionary

    class_name_mapping = {
        'adenocarcinoma': 'AdC',
        'benign mucosa': 'BM',
        'inflammatory cells': 'IC',
        'serosa': 'Ser',
        'smooth muscle': 'SM',
        'submucosa': 'Sub'
    }

    class_names = list(class_name_mapping.values())

    return train_data, train_labels, val_data, val_labels, test_data, test_labels, class_names

# def process_data(input_file, balanced=False, should_shuffle=False):
#     # Load the CSV file
#     df = pd.read_csv(input_file)
#     df = df.loc[df['Class'] != 'tissue']
#
#     # Encode the labels
#     le = LabelEncoder()
#     df['Class'] = le.fit_transform(df['Class'])
#
#     # Save the class names
#     class_names = le.classes_
#
#     # Get a list of unique slide names
#     slide_names = df['Slide'].unique()
#
#     # Split the slide names into training and test sets
#     train_slide_names, test_slide_names = train_test_split(slide_names, test_size=0.2, random_state=42)
#
#     # Create the training and test DataFrames
#     train_df = df[df['Slide'].isin(train_slide_names)]
#     test_df = df[df['Slide'].isin(test_slide_names)]
#
#     print("Training slides:", train_df['Slide'].unique())
#     print("Test slides:", test_df['Slide'].unique())
#
#     # Reset the index of the DataFrames
#     train_df.reset_index(drop=True, inplace=True)
#     test_df.reset_index(drop=True, inplace=True)
#
#     if balanced == True:
#         # Balance the training set
#         unique_classes_train = train_df.iloc[:, 1].unique()
#         class_dataframes_train = {class_name: train_df[train_df.iloc[:, 1] == class_name] for class_name in unique_classes_train}
#         min_samples_train = min(len(class_df) for class_df in class_dataframes_train.values())
#         for class_name, class_df in class_dataframes_train.items():
#             if len(class_df) > min_samples_train:
#                 class_dataframes_train[class_name] = class_df.sample(min_samples_train)
#         train_df = pd.concat(class_dataframes_train.values())
#
#     # Separate the features and labels in the training and test dataframes
#     train_features = train_df[df.columns[4:]]
#     train_labels = train_df['Class']
#     test_features = test_df[df.columns[4:]]
#     test_labels = test_df['Class']
#
#     if should_shuffle == True:
#         # Shuffle the training data
#         train_features, train_labels = shuffle(train_features, train_labels)
#
#     # Convert the features to TensorFlow tensors
#     train_data = tf.convert_to_tensor(train_features.values, dtype=tf.float32)
#     test_data = tf.convert_to_tensor(test_features.values, dtype=tf.float32)
#
#     train_labels = tf.convert_to_tensor(train_labels.values, dtype=tf.int32)
#     test_labels = tf.convert_to_tensor(test_labels.values, dtype=tf.int32)
#
#     return train_data, train_labels, test_data, test_labels, class_names