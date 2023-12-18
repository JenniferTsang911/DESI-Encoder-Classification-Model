import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
import torch.nn.functional as F

#try to only feed the ROI selections into the autoencoder w/ modification
#just extract data that is other than background or tissues. LImiting it to only ROI (Helps reduce the size)
#use a dr approach to reduce dimensionality to a resonable point (we need a higher one for classification)
#Change things to a dimension reduct and then classification. What happens if we use classical methods (PCA + LDA) as comapred to using deep learning approach
# Compare the results between the two. What is the difference between the two? PCA LDA vs machine learning DR and classification
# Add tissue data as well into the DR (will this imrpove the classification accuracy). This means we used unlabelled data to improve accuracy
# for the AE, we have another dense layer (recieves teh DR reduced data) and classify the data. All labeled data.
#ask GPT to create an AE with dataload with validation. Then use the AE to classify the data. Then compare the results
#read csv, and only keep the data of the 6 labels, then recreate the csv file and open on slicer
#you can fake the m/z it does not matter for the slicer

# Define the dataset
class MassSpecDataset(Dataset):
    def __init__(self, data, labels=None):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.labels is not None:
            return self.data[idx], self.labels[idx]
        else:
            return self.data[idx]


# class Autoencoder(nn.Module):
#     def __init__(self, input_dim, latent_dim=100):
#         super(Autoencoder, self).__init__()
#
#         # Encoder
#         self.encoder = nn.Sequential(
#             nn.Linear(input_dim, 500),
#             nn.ReLU(),
#             nn.Linear(500, latent_dim),
#             nn.ReLU()
#         )
#
#         # Decoder
#         self.decoder = nn.Sequential(
#             nn.Linear(latent_dim, 500),
#             nn.ReLU(),
#             nn.Linear(500, input_dim),
#             nn.Sigmoid()  # Assuming input values are normalized between 0 and 1
#         )
#
#         self.apply(self.init_weights)
#
#     def forward(self, x):
#         encoded = self.encoder(x)
#         decoded = self.decoder(encoded)
#         return encoded, decoded
#
#     def init_weights(self, m):
#         if type(m) == nn.Linear:
#             # Uniform Initialization
#             #torch.nn.init.uniform_(m.weight)
#
#             # Normal Initialization
#             torch.nn.init.normal_(m.weight)
#
#             # Xavier Initialization
#             #torch.nn.init.xavier_uniform_(m.weight)
#
#             # He Initialization
#             #torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
#
#             # Orthogonal Initialization
#             # torch.nn.init.orthogonal_(m.weight)
#
#             m.bias.data.fill_(0.01)
class Autoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim=250):
        super(Autoencoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 1000),
            nn.ReLU(),
            nn.Linear(1000, 500),
            nn.ReLU(),
            nn.Linear(500, encoding_dim),
            nn.ReLU()
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 500),
            nn.ReLU(),
            nn.Linear(500, 1000),
            nn.ReLU(),
            nn.Linear(1000, input_dim),
            nn.Sigmoid()  # Use Sigmoid activation for reconstruction in [0, 1]
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded
class Classifier(nn.Module):
    def __init__(self, input_dim=100, dropout_rate=0.3, momentum=0.1):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.bn1 = nn.BatchNorm1d(128, momentum=momentum)

        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.bn2 = nn.BatchNorm1d(64, momentum=momentum)

        self.fc3 = nn.Linear(64, 32)
        self.dropout3 = nn.Dropout(dropout_rate)
        self.bn3 = nn.BatchNorm1d(32, momentum=momentum)

        self.fc4 = nn.Linear(32, 16)
        self.dropout4 = nn.Dropout(dropout_rate)
        self.bn4 = nn.BatchNorm1d(16, momentum=momentum)

        self.fc5 = nn.Linear(16, 6)

        self.apply(self.init_weights)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout2(x)

        x = self.fc3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.dropout3(x)

        x = self.fc4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.dropout4(x)

        x = self.fc5(x)
        return F.log_softmax(x, dim=1)

    def init_weights(self, m):
        if type(m) == nn.Linear:
            if m.out_features == 6:  # Output layer
                torch.nn.init.xavier_uniform_(m.weight)
            else:  # Layers with ReLU activation
                torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            m.bias.data.fill_(0.01)

class CNNClassifier(nn.Module):
    def __init__(self, num_channels=1, dropout_rate=0.3, momentum=0.1, output_size=64):
        super(CNNClassifier, self).__init__()
        self.conv1 = nn.Conv1d(num_channels, 256, kernel_size=3, stride=1, padding=1)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.bn1 = nn.BatchNorm1d(256, momentum=momentum)

        self.conv2 = nn.Conv1d(256, 128, kernel_size=3, stride=1, padding=1)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.bn2 = nn.BatchNorm1d(128, momentum=momentum)

        self.conv3 = nn.Conv1d(128, 64, kernel_size=3, stride=1, padding=1)
        self.dropout3 = nn.Dropout(dropout_rate)
        self.bn3 = nn.BatchNorm1d(64, momentum=momentum)

        self.conv4 = nn.Conv1d(64, 32, kernel_size=3, stride=1, padding=1)
        self.dropout4 = nn.Dropout(dropout_rate)
        self.bn4 = nn.BatchNorm1d(32, momentum=momentum)

        # Calculate the output size of the convolutional layers
        self.conv_output_size = output_size * 50  # Adjust this value based on your convolutional layers

        self.fc = nn.Linear(self.conv_output_size, 6)  # Adjusted input size

        # Apply weight initialization
        self.apply(self.init_weights)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.bn1(x)
        x = self.dropout1(x)

        x = F.relu(self.conv2(x))
        x = self.bn2(x)
        x = self.dropout2(x)

        x = F.relu(self.conv3(x))
        x = self.bn3(x)
        x = self.dropout3(x)

        # New convolutional layer
        x = F.relu(self.conv4(x))
        x = self.bn4(x)
        x = self.dropout4(x)

        # Flatten the tensor along the channel dimension before passing to the fully connected layer
        x = x.view(x.size(0), -1)

        x = self.fc(x)
        return F.log_softmax(x, dim=1)

    def init_weights(self, m):
        if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
