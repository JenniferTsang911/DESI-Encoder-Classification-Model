import pandas as pd
import random

# Load the CSV dataset into a Pandas DataFrame without column names
df = pd.read_csv('your_dataset.csv', header=None)

# Find unique classes in the first column
unique_classes = df[0].unique()

# Create a dictionary to store the data frames for each class
class_dataframes = {}

# Split the data into separate DataFrames based on the unique classes
for class_name in unique_classes:
    class_dataframes[class_name] = df[df[0] == class_name]

# Find the minimum number of samples among the classes
min_samples = min(len(class_dataframes[class_name]) for class_name in unique_classes)

# Randomly select and remove rows from classes with more samples
for class_name in unique_classes:
    if len(class_dataframes[class_name]) > min_samples:
        # Randomly shuffle the rows in the DataFrame
        class_dataframes[class_name] = class_dataframes[class_name].sample(frac=1, random_state=1)

        # Keep only the first 'min_samples' rows
        class_dataframes[class_name] = class_dataframes[class_name].head(min_samples)

# Concatenate the dataframes for each class back into a single dataframe
balanced_df = pd.concat(class_dataframes.values())

# Save the balanced dataset to a new CSV file
balanced_df.to_csv('balanced_dataset.csv', header=False, index=False)
