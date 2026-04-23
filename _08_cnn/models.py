import torch
import torch.nn.functional as F
import torch.nn as nn

# TODO NOTE: nn.Conv2d supports complex types! Try complex input with CNN?

class Hoffman_CNN(nn.Module):
    def __init__(self, in_channels, height, width):
        super().__init__()
        # Get input dimensions
        self.in_channels = in_channels
        self.height = height
        self.width = width

        # Define the convolutional layers
        # NOTE padding='same' preserves the 
        self.conv1 = nn.Conv2d(in_channels, 7, kernel_size=3, stride=1, padding='same')
        self.conv2 = nn.Conv2d(7, 14, kernel_size=3, stride=1, padding='same')
        self.conv3 = nn.Conv2d(14, 28, kernel_size=3, stride=1, padding='same')
        self.conv4 = nn.Conv2d(28, 56, kernel_size=3, stride=1, padding='same')
        self.conv5 = nn.Conv2d(56, 112, kernel_size=3, stride=1, padding='same')

        # Get size of features to regress in fully connected layer
        # NOTE THIS WORKS, LAZY LINEAR DOES NOT
        with torch.no_grad():
            dummy_input = torch.zeros(1, in_channels, height, width)
            dummy_output = self.forward_features(dummy_input)
            print(f'dummy out shape: {dummy_output.shape}')
            print(f'dummy_out.view(1, -1).shape {dummy_output.view(1, -1).shape}')
            in_features_size = dummy_output.view(1, -1).shape[1]
            print(f'flat size shape: {in_features_size}')

        # Define final fully connected layer to regress features to output vectors
        self.fc = nn.Linear(in_features_size, 2 * height * width)

    def forward_features(self, xb):

        # Five convolutional layers
        xb = F.relu(self.conv1(xb))
        xb = F.max_pool2d(xb, kernel_size=2, stride=2)

        xb = F.relu(self.conv2(xb))
        xb = F.max_pool2d(xb, kernel_size=2, stride=2)

        xb = F.relu(self.conv3(xb))
        xb = F.max_pool2d(xb, kernel_size=2, stride=2)

        xb = F.relu(self.conv4(xb))
        xb = F.max_pool2d(xb, kernel_size=2, stride=2)

        xb = F.relu(self.conv5(xb))
        xb = F.max_pool2d(xb, kernel_size=2, stride=2)

        # 20% random dropout, only applied during training
        xb = F.dropout(xb, p=0.2, training=self.training)  
        
        return xb


    def forward(self, xb):

        xb = self.forward_features(xb)

        # Flatten features to 1D vector
        xb = torch.flatten(xb, start_dim=1)

        # Regress features to 1D vector of ui and vi outputs
        xb = self.fc(xb)

        # Return the reshaped batch of ui and vi outputs
        return xb.view(-1, 2, self.height, self.width)