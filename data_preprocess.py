import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import torch
from sklearn.model_selection import train_test_split

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def process_data(input_file, balanced=False):
    # Load the CSV file
    df = pd.read_csv(input_file)
    df = df.loc[df['Class'] != 'tissue']

    # Encode the labels
    le = LabelEncoder()
    df['Class'] = le.fit_transform(df['Class'])

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

        # For val_df
        unique_classes_val = val_df.iloc[:, 1].unique()
        class_dataframes_val = {class_name: val_df[val_df.iloc[:, 1] == class_name] for class_name in unique_classes_val}
        min_samples_val = min(len(class_df) for class_df in class_dataframes_val.values())
        for class_name, class_df in class_dataframes_val.items():
            if len(class_df) > min_samples_val:
                class_dataframes_val[class_name] = class_df.sample(min_samples_val)
        val_df = pd.concat(class_dataframes_val.values())

    # Separate the features and labels in the training, validation, and test dataframes
    train_features = train_df[df.columns[4:]]
    train_labels = train_df['Class']
    val_features = val_df[df.columns[4:]]
    val_labels = val_df['Class']
    test_features = test_df[df.columns[4:]]
    test_labels = test_df['Class']

    # Convert the features to PyTorch tensors
    train_data = torch.tensor(train_features.values, dtype=torch.float32).to(device)
    val_data = torch.tensor(val_features.values, dtype=torch.float32).to(device)
    test_data = torch.tensor(test_features.values, dtype=torch.float32).to(device)

    # Convert the labels to PyTorch tensors
    train_labels = torch.tensor(train_labels.values, dtype=torch.long).to(device)
    val_labels = torch.tensor(val_labels.values, dtype=torch.long).to(device)
    test_labels = torch.tensor(test_labels.values, dtype=torch.long).to(device)

    return train_data, train_labels, val_data, val_labels, test_data, test_labels