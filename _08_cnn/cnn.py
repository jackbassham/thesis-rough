import helpers
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
from . import utils

# TODO Refactor THIS ONE
import torch
import torch.nn as nn
import torch.nn.functional as F

# TODO NOTE: nn.Conv2d supports complex types! Try complex input with CNN?

MODEL_STR = 'cnn'

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


    def forward(self, xb):
        print(f'inputs: {xb.shape}')

        xb = F.relu(self.conv1(xb))
        xb = F.max_pool2d(xb, kernel_size=2, stride=2)
        print(f'layer 1: {xb.shape}')

        xb = F.relu(self.conv2(xb))
        xb = F.max_pool2d(xb, kernel_size=2, stride=2)
        print(f'layer 2: {xb.shape}')
        

        xb = F.relu(self.conv3(xb))
        xb = F.max_pool2d(xb, kernel_size=2, stride=2)
        print(f'layer 3: {xb.shape}')


        xb = F.relu(self.conv4(xb))
        xb = F.max_pool2d(xb, kernel_size=2, stride=2)
        print(f'layer 4: {xb.shape}')


        xb = F.relu(self.conv5(xb))
        xb = F.max_pool2d(xb, kernel_size=2, stride=2)
        print(f'layer 5: {xb.shape}')


        # 20% random dropout
        xb = F.dropout(xb, p=0.2)
        print(f'dropout layer: {xb.shape}')


        # Flatten to 1D vector
        xb = torch.flatten(xb, start_dim=1)
        print(f'flatten: {xb.shape}')


        # # Fully Connected Layer: Regress to 1D vector of ui and vi outputs
        # xb = F.linear(xb.shape, 2 * self.height * self.width)
        # print(f'fully connected: {xb.shape}')

        # Fully Connected Layer: Regress to 1D vector of ui and vi outputs
        self.fc = nn.LazyLinear(2 * self.height * self.width)
        xb = self.fc(xb)
        print(f'fully connected: {xb.shape}')


        # Return the batch of ui and vi outputs
        xb = xb.view(-1, 2, self.height, self.width)
        print(f'outputs: {xb.shape}')

        return xb


def main(cfg):

    # Set random seed for reproducibility
    helpers.set_seed()

    # Get device
    device = utils.check_for_accelerator()

    # Load tensor datasets from numpy arrays
    train_ds, val_ds, test_ds = utils.get_datasets(cfg, device)

    # Get batched data loaders from tensor datasets
    train_dl, val_dl, test_dl = utils.get_batched_data(train_ds, val_ds, test_ds)

    # Get input dimensions
    in_channels, height, width = utils.get_input_dimensions(train_dl)

    # Complile model
    # NOTE using PyTorch Default 'Kaiming Uniform' weights/bias initialization
    # Tensorflow Default is Xavier (used by Hoffman)
    # TODO If Kaiming is bad, use function in utils to apply Xavier initializtion
    model = Hoffman_CNN(
        in_channels, height, width).to(device)

    # Define regularization
    weight_decay = 1e-4 # L2 Norm Regularization, changed from 0.01 in TensorFlow
    # NOTE TensorFLow multiplies Regularization by 0.05 0.01*0.05 -> 5e-4

    # Define Learnig Rate
    lr = 1e-3

    # Initialize optimizer with weight decay (l2 regularization)
    opt = torch.optim.Adam(model.parameters(), lr = lr, weight_decay=0.01)

    # Define number of epochs
    num_epochs = 1 # Hoffman

    # Initialize losses
    train_losses = []
    val_losses = []

    # Train model
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for xb, yb in tqdm(trainData, desc=f"Train Epoch {epoch+1}/{num_epochs}", leave=False):
            # print(torch.isnan(xb).any(), torch.isinf(xb).any())
            optimizer.zero_grad()
            preds = model(xb)
            loss = NRMSEloss(preds, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0
        for xb, yb in tqdm(valData, desc=f"Val   Epoch {epoch+1}/{num_epochs}", leave=False):
            with torch.no_grad():
                preds = model(xb)
                loss = NRMSEloss(preds, yb)
                val_loss += loss.item()

        avg_train = train_loss / len(trainData)
        avg_val   = val_loss   / len(valData)

        
        train_losses.append(avg_train)
        val_losses.append(avg_val)
        
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train:.4f} - Val Loss: {avg_val:.4f}")

    # Load in model outputs destination path
    path_cnn_out = cfg.path_config.model_path('cnn_pt')

    # Make destination directory if missing
    cfg.path_config.makedir_if_missing(path_cnn_out)

    # Plot losses
    plot_losses(path_cnn_out, 'cnn_pt_lossses', num_epochs, train_losses, val_losses)

    # Save model weights
    torch.save(
        model.state_dict(), 
        path_cnn_out / 'weights.pt'
        )

    print('Model weights saved')

    # Evaluate trained model
    model.eval()

    # Get predictions on test set
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for xb, yb in testData:
            preds = model(xb)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(yb.cpu().numpy())

    # Concatenate all batches
    y_pred = np.concatenate(all_preds, axis=0)
    y_true = np.concatenate(all_targets, axis=0)

    print('y_pred shape:', y_pred.shape)
    print('y_true shape:', y_true.shape)

    # Save predictions and true values
    np.savez(
        path_cnn_out / 'preds.npz', 
        y_pred = y_pred, y_true = y_true)

    print("Predictions saved")


def set_seed(seed=42):
    torch.manual_seed(seed) # PyTorch Reproducibility
    torch.cuda.manual_seed(seed) # Required if using GPU
    torch.backends.cudnn.deterministic = True  # Reproducibility if using GPU
    torch.backends.cudnn.benchmark = False # Paired with above

    return


def plot_losses(path, filename, num_epochs, train_losses, val_losses):
    epochs = np.arange(1, num_epochs + 1)

    
    plt.figure()
    plt.plot(epochs, train_losses, label = 'Train')
    plt.plot(epochs, val_losses, label = 'Validation')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title(f"{MODEL_STR} Loss")

    plt.savefig(path / filename)

    # plt.show()

    return


def NRMSEloss(pred, true, eps=1e-6):
    """
    Norm Root Mean Squared Loss
    """

    mse = torch.mean((pred - true) ** 2)
    std = torch.std(true, unbiased = False) + eps # Unbiased=True To match default pop. std. in tf 

    return torch.sqrt(mse) / std


if __name__ == "__main__":
    from _00_config.load_config import load_config
    cfg = load_config()
    main(cfg)