import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import torch

# Load the CSV dataset into a Pandas DataFrame
df = pd.read_csv(r"C:\Users\tsang\OneDrive - Queen's University\DESI project\DESI TXT colon\Annotated Dataset\2021 03 30 colon 0720931-3 Analyte 5_dataset.csv", header=0)

# Ignore the first column of the DataFrame
df = df.iloc[:, 1:]

# Find unique classes in the second column
unique_classes = df.iloc[:, 0].unique()

# Create a dictionary to store the data frames for each class
class_dataframes = {class_name: df[df.iloc[:, 0] == class_name] for class_name in unique_classes}

# Find the minimum number of samples among the classes
min_samples = min(len(class_df) for class_df in class_dataframes.values())

# Randomly select and remove rows from classes with more samples
for class_name, class_df in class_dataframes.items():
    if len(class_df) > min_samples:
        class_dataframes[class_name] = class_df.sample(min_samples)

# Concatenate the dataframes for each class back into a single dataframe
balanced_df = pd.concat(class_dataframes.values())

# Save the balanced dataset to a new CSV file
balanced_df.to_csv(r"2021 03 30 colon 0720931-3 Analyte 5_dataset_balanced.csv", index=False)

# # Load the dataset
# df = pd.read_csv(r"C:\Users\tsang\Documents\GitHub\dc-DeepMSI\balanced_dataset.csv", header=0)

# # Assuming the column name for labels is 'tissue_type'
# labels = df['Class'].values

# # Convert labels to numerical values if they are categorical
# le = LabelEncoder()
# labels = le.fit_transform(labels)

# # Convert labels to tensor
# labels = torch.from_numpy(labels)
