import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from AE_Classifier_Model import *
from data_preprocess import *
import pandas as pd

from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, ConfusionMatrixDisplay
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import matplotlib.pyplot as plt
import random

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from itertools import product
import json
from sklearn.metrics import f1_score
from sklearn.decomposition import PCA

def perform_pca(train_data, val_data, n_components):
    pca = PCA(n_components=n_components)

    # Fit the PCA on the training data and apply the transformation to the training data
    train_data_pca = pca.fit_transform(train_data.cpu().numpy())

    # Apply the PCA transformation to the validation data
    val_data_pca = pca.transform(val_data.cpu().numpy())

    # Convert the transformed data back to PyTorch tensors
    train_data_pca = torch.tensor(train_data_pca, dtype=torch.float32).to(device)
    val_data_pca = torch.tensor(val_data_pca, dtype=torch.float32).to(device)

    return train_data_pca, val_data_pca

def train_autoencoder(train_data, train_labels, val_data, val_labels, num_epochs_AE=1000, AE_LR=0.001, l2_AE=0.0001, encoding_dim=50):
    # Initialize the autoencoder model
    input_dim = train_data.shape[1]
    autoencoder = Autoencoder(input_dim, encoding_dim).to(device)

    # Define the loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(autoencoder.parameters(), lr=AE_LR, weight_decay=l2_AE)

    # Create DataLoader objects for the training and validation data
    train_loader = DataLoader(MassSpecDataset(train_data, train_labels), batch_size=128, shuffle=True)
    val_loader = DataLoader(MassSpecDataset(val_data, val_labels), batch_size=128)

    # Initialize lists to store loss values
    train_losses = []
    val_losses = []

    # Initialize best validation loss and patience counter
    best_val_loss = float('inf')
    patience_counter = 0

    patience = int(num_epochs_AE*0.1)

    # Start the training loop
    for epoch in range(num_epochs_AE):
        # Train on the training data
        for batch_data, _ in train_loader:
            optimizer.zero_grad()
            encoded, decoded = autoencoder(batch_data)
            loss = criterion(decoded, batch_data)

            # # Add L1 regularization
            # l1_reg = torch.tensor(0., requires_grad=True).to(device)
            # for name, param in autoencoder.named_parameters():
            #     if 'weight' in name:
            #         l1_reg = l1_reg + torch.norm(param, 1)
            # loss = loss + l1_AE * l1_reg

            loss.backward(retain_graph=True)
            optimizer.step()

        # Append the training loss for this epoch
        train_losses.append(loss.item())

        # Evaluate on the validation data
        val_loss = 0
        with torch.no_grad():
            for val_data, _ in val_loader:
                encoded, decoded = autoencoder(val_data)
                val_loss += criterion(decoded, val_data).item()
        val_loss /= len(val_loader)

        # Append the validation loss for this epoch
        val_losses.append(val_loss)

        # Check for early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping")
                break

        #print(f"Epoch {epoch+1}/{num_epochs_AE} - Loss: {loss.item()} - Val Loss: {val_loss}")

    train_data = autoencoder.encoder(train_data)
    val_data = autoencoder.encoder(val_data)

    #torch.save(autoencoder.state_dict(), 'autoencoder.pth')

    return autoencoder, train_data, val_data, train_losses, val_losses

def train_classifier(train_data, train_labels, val_data, val_labels, num_epochs_CL=200, CL_LR=0.001, dropoutCL=0.5, momentumCL=0.1, l2_CL=0.01):

    # autoencoder = Autoencoder(train_data.shape[1]).to(device)
    # state_dict = torch.load('autoencoder.pth')
    # autoencoder.load_state_dict(state_dict)
    #
    # train_data = autoencoder.encoder(train_data)
    # val_data = autoencoder.encoder(val_data)
    #
    print(train_data.shape)

    train_loader = DataLoader(MassSpecDataset(train_data, train_labels), batch_size=128, shuffle=True)
    val_loader = DataLoader(MassSpecDataset(val_data, val_labels), batch_size=128)

    print('Training classifier...')
    model = Classifier(train_data.shape[1], dropoutCL, momentumCL).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=CL_LR, weight_decay=l2_CL)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

    loss_values_classifier = []
    val_loss_values_classifier = []
    accuracy_values_classifier = []
    best_val_loss_CL = float('inf')
    epochs_no_improve_CL = 0
    patience = int(num_epochs_CL*0.1)

    for epoch in range(num_epochs_CL):
        for batch_data, batch_labels in train_loader:
            batch_labels = batch_labels.clone().detach()
            optimizer.zero_grad()
            outputs = model(batch_data)
            loss = criterion(outputs, batch_labels)
            loss.backward(retain_graph=True)
            optimizer.step()

        scheduler.step()

        correct_predictions = 0
        total_predictions = 0
        val_loss_CL = 0
        with torch.no_grad():
            for val_data, val_labels in val_loader:
                val_outputs = model(val_data)
                val_loss_CL += criterion(val_outputs, val_labels).item()

                # Calculate accuracy
                _, predicted = torch.max(val_outputs.data, 1)
                correct_predictions += (predicted == val_labels).sum().item()
                total_predictions += val_labels.size(0)

        val_loss_CL /= len(val_loader)
        val_loss_values_classifier.append(val_loss_CL)

        accuracy = correct_predictions / total_predictions
        accuracy_values_classifier.append(accuracy)

        if val_loss_CL < best_val_loss_CL:
            best_val_loss_CL = val_loss_CL
            epochs_no_improve_CL = 0
        else:
            epochs_no_improve_CL += 1

        if epochs_no_improve_CL == patience:
            print('Early stopping!')
            break

        loss_values_classifier.append(loss.item())
        print(f"Classifier Epoch {epoch+1}/{num_epochs_CL} - Loss: {loss.item()} - Val Loss: {val_loss_CL} - Accuracy: {accuracy}")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return model, loss_values_classifier, val_loss_values_classifier, accuracy_values_classifier

def train_CNN(train_data, train_labels, val_data, val_labels, num_epochs=200, learning_rate=0.0001, batch_size=64, dropout_rate=0.5, momentum=0.1, l2_cl=0.01):
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Use CUDA if available
    #
    # autoencoder = Autoencoder(train_data.shape[1]).to(device)
    # state_dict = torch.load('autoencoder.pth')
    # autoencoder.load_state_dict(state_dict)
    #
    # train_data = autoencoder.encoder(train_data)
    # val_data = autoencoder.encoder(val_data)

    num_channels = 1
    model = CNNClassifier(num_channels, dropout_rate, momentum).to(device)

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=l2_cl)  # L2 regularization

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

    # Create DataLoader objects for the training and validation data
    train_loader = DataLoader(MassSpecDataset(train_data, train_labels), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(MassSpecDataset(val_data, val_labels), batch_size=batch_size)

    best_val_loss = float('inf')
    epochs_no_improve_CL = 0

    loss_list = []
    val_loss_list = []

    # Start the training loop
    for epoch in range(num_epochs):
        # Train on the training data
        for batch_data, batch_labels in train_loader:
            batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)

            # Reshape the data to match the input shape expected by the CNN
            batch_data = batch_data.view(batch_data.size(0), 1, -1)

            optimizer.zero_grad()
            outputs = model(batch_data)
            loss = criterion(outputs, batch_labels)
            loss.backward(retain_graph=True)
            optimizer.step()

        # Evaluate on the validation data
        val_loss = 0
        patience = int(num_epochs*0.1)

        with torch.no_grad():
            for val_data, val_labels in val_loader:
                val_data, val_labels = val_data.to(device), val_labels.to(device)
                val_data = val_data.view(val_data.size(0), 1, -1)
                outputs = model(val_data)
                val_loss += criterion(outputs, val_labels).item()
        val_loss /= len(val_loader)

        loss_list.append(loss.item())
        val_loss_list.append(val_loss)

        # Learning rate scheduling
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve_CL = 0
        else:
            epochs_no_improve_CL += 1

        if epochs_no_improve_CL == patience:
            print('Early stopping!')
            break

        #print(f"Epoch {epoch+1}/{num_epochs} - Loss: {loss.item()} - Val Loss: {val_loss}")

    return model, loss_list, val_loss_list

def evaluate_model(autoencoder, model, test_loader):
    model.eval()  # Set the model to evaluation mode
    all_labels = []
    all_predictions = []

    with torch.no_grad():  # No need to calculate gradients
        for data, labels in test_loader:
            encoded_data, _ = autoencoder(data)  # Pass the test data through the autoencoder
            #encoded_data = encoded_data.view(encoded_data.size(0), 1, -1)
            outputs = model(encoded_data)  # Use the encoded test data
            _, predicted = torch.max(outputs.data, 1)
            all_labels.extend(labels.cpu())  # Move labels to CPU
            all_predictions.extend(predicted.cpu())  # Move predicted to CPU

    # Convert to numpy arrays for use with sklearn metrics
    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)

    # Compute confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)

    # Display confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)

    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')  # Save the plot to a file

    print("Precision:", precision_score(all_labels, all_predictions, average='weighted'))
    print("Recall:", recall_score(all_labels, all_predictions, average='weighted'))
    print("F1 Score:", f1_score(all_labels, all_predictions, average='weighted'))

def main():
    num_epochs_AE = 1000
    AE_LR = 0.008
    l2_AE = 0.0006

    num_epochs_CL = 1000
    batch_size = 64
    CL_LR = 0.001
    dropoutCL = 0.5
    momentumCL = 0.99
    l2_CL = 0.0001

    train_data, train_labels, val_data, val_labels, test_data, test_labels = process_data(r"C:\Users\jenni\Documents\GitHub\DESI-project\all_aligned_no_background_others_preprocessed.csv", balanced=True)

    #train_data, val_data = perform_pca(train_data, val_data, 1000)

    autoencoder, train_data, val_data, loss_values_autoencoder, val_loss_values_autoencoder = train_autoencoder(train_data, train_labels, val_data, val_labels, num_epochs_AE, AE_LR, l2_AE)

    #model, loss_values_classifier, val_loss_values_classifier = train_CNN(train_data, train_labels, val_data, val_labels, num_epochs_CL, CL_LR, batch_size, dropoutCL, momentumCL, l2_CL)
    model, loss_values_classifier, val_loss_values_classifier, accuracy_values_classifier = train_classifier(train_data, train_labels, val_data, val_labels, num_epochs_CL, CL_LR, dropoutCL, momentumCL, l2_CL)

    # pca = PCA(n_components=1000)
    # test_data_pca = pca.fit_transform(test_data.cpu().numpy())
    # test_data_pca = pca.transform(test_data.cpu().numpy())

    # test_data_pca = torch.tensor(test_data_pca, dtype=torch.float32).to(device)
    # test_loader = DataLoader(MassSpecDataset(test_data_pca, test_labels), batch_size=128)

    # all_predicted = []
    #
    # # Use the trained model to predict the test labels
    # with torch.no_grad():
    #     for batch_data, _ in test_loader:
    #         outputs = model(batch_data)
    #         _, predicted = torch.max(outputs.data, 1)
    #         all_predicted.append(predicted.cpu().numpy())
    #
    # # Convert the list of predicted labels to a numpy array
    # all_predicted = np.concatenate(all_predicted)
    #
    # # Calculate the F1 score
    # f1 = f1_score(test_labels.cpu(), all_predicted, average='macro')
    # print(f"F1 Score: {f1}")
    #
    # # Calculate the confusion matrix
    # cm = confusion_matrix(test_labels.cpu(), all_predicted)
    #
    # # Display the confusion matrix
    # disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    # disp.plot(cmap=plt.cm.Blues)
    # plt.show()

    #Create a DataLoader for the test data
    test_loader = DataLoader(MassSpecDataset(test_data, test_labels), batch_size=64)

    evaluate_model(autoencoder,model, test_loader)

    #Create a figure and a set of subplots
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))  # Adjust the figsize as needed

    # Plot the training and validation loss curves for the autoencoder
    axs[0].plot(loss_values_autoencoder, label='Training Loss - Autoencoder')
    axs[0].plot(val_loss_values_autoencoder, label='Validation Loss - Autoencoder')
    axs[0].set_title('Loss curves for the Autoencoder')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Loss')
    axs[0].legend()

    # Plot the training and validation loss curves for the classifier
    axs[1].plot(loss_values_classifier, label='Training Loss - Classifier')
    axs[1].plot(val_loss_values_classifier, label='Validation Loss - Classifier')
    axs[1].set_title('Loss curves for the Classifier')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Loss')
    axs[1].legend()

    # Display the figure
    plt.tight_layout()
    plt.show()

    plt.figure()
    plt.plot(accuracy_values_classifier, label='accuracy')

    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Curve')

    plt.legend()

    plt.show()

main()
'''
def process_data(input_file):
    # Load the CSV file
    df = pd.read_csv(input_file)

    # Encode the labels
    le = LabelEncoder()
    df['Class'] = le.fit_transform(df['Class'])

    # Get the encoded value for 'tissues'
    tissues_class_value = le.transform(['tissues'])[0]

    # Rest of your code...

    return train_data, train_labels, val_data, val_labels, test_data, test_labels, tissues_class_value

def train_model(train_data, train_labels, val_data, val_labels, test_data, test_labels, tissues_class_value, num_epochs_AE=1000, num_epochs_CL=200, AE_LR=0.001, CL_LR=0.001, dropoutCL=0.5, momentumCL=0.1, momentumAE=0.1):
    # Rest of your code...

    # Train the autoencoder
    for epoch in range(num_epochs_AE):
        # Your training code...

    # Remove the 'tissues' class from the training data and the labels
    tissues_class_index = (train_labels != tissues_class_value)
    train_data = train_data[tissues_class_index]
    train_labels = train_labels[tissues_class_index]

    # Rest of your code...

    return model, loss_values_classifier, loss_values_autoencoder, val_loss_values_classifier, val_loss_values_autoencoder, test_data'''