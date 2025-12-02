import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import numpy as np
import os
import matplotlib.pyplot as plt
cuda_available = torch.cuda.is_available()

######################################################################
# NOTE Adapted from V5, but keeping flag values in uncertainty

######################################################################
# ****************************************************************** #
######################################################################

START_YEAR = 1992
END_YEAR = 2020
HEM = 'nh'

# Get current script directory path
script_dir = os.path.dirname(__file__)

# Navigate to data source directory from current path
PATH_SOURCE = os.path.join(script_dir, '..', '..', '..', 'data', HEM, 'cnn-inputs')

# Get absolute path to data source directory
PATH_SOURCE = os.path.abspath(PATH_SOURCE)

# Define destination path for inputs
PATH_DEST = os.path.join(script_dir, '..', '..', '..', 'data', HEM, 'outputs', 'cnn', 'weighted')

# Get absolute path to data destination directory
PATH_DEST = os.path.abspath(PATH_DEST)

# Create the direectory if it doesn't already exist
os.makedirs(PATH_DEST, exist_ok=True)


def set_seed(seed=42):
    torch.manual_seed(seed) # PyTorch Reproducibility
    torch.cuda.manual_seed(seed) # Required if using GPU
    torch.backends.cudnn.deterministic = True  # Reproducibility if using GPU
    torch.backends.cudnn.benchmark = False # Paired with above

    return

class WeightedCNN(nn.Module):
    """
    CNN architecture taken directly from Hoffman et al. (2023)
    
    """

    def __init__(self, in_channels, out_channels, height, width):
        super().__init__()

        self.out_channels = out_channels
        self.height = height
        self.width = width

        # Convolutional layers
        self.layer1 = nn.Conv2d(in_channels, 7, kernel_size=3, stride=1, padding='same')
        self.layer2 = nn.Conv2d(7, 14, kernel_size=3, stride=1, padding='same')
        self.layer3 = nn.Conv2d(14, 28, kernel_size=3, stride=1, padding='same')
        self.layer4 = nn.Conv2d(28, 56, kernel_size=3, stride=1, padding='same')
        self.layer5 = nn.Conv2d(56, 112, kernel_size=3, stride=1, padding='same')

        # Pooling, activation, dropout
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)

        # Dynamically compute flattened size using dummy input
        with torch.no_grad():
            dummy_input = torch.zeros(1, in_channels, height, width)
            dummy_out = self.forward_features(dummy_input)
            flat_size = dummy_out.view(1, -1).shape[1]

        # Final fully connected layer
        self.fc = nn.Linear(flat_size, out_channels * height * width)

    def forward_features(self, x):
        x = self.relu(self.layer1(x))
        x = self.pool(x)
        x = self.relu(self.layer2(x))
        x = self.pool(x)
        x = self.relu(self.layer3(x))
        x = self.pool(x)
        x = self.relu(self.layer4(x))
        x = self.pool(x)
        x = self.relu(self.layer5(x))
        x = self.pool(x)
        x = self.dropout(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        x = x.view(-1, self.out_channels, self.height, self.width)
        return x
    

# # NOTE 
# # 2. Change from mean to weighted average (dividing by sum(w))
def WeightedMSEloss(pred, true, r, eps=1e-4):

    # # Set NaN weights to zero (ignore in loss)
    # w = torch.where(torch.isnan(r), torch.tensor(0.0, device = r.device, dtype = r.dtype), (1 / r + eps))  

    # Compute weights
    w = 1 / (r + eps)  # [B, C, H, W]

    # DO NOT NORMALIZE, OR NORMALIZE BY THE MEAN OF THE VALID POINTS ONLY
    # TO AVOID INFLATION OF WEIGHTS

    # IF THIS WORKS, TRY NORMALIZING BY VALUES NOT FLAGGED, AND LEAVING
    # THE FLAGGED VALUES IN!

    # # # Normalize weights so mean is 1
    # w = w / torch.mean(w)

    # Compute square error
    se = (pred - true) ** 2                     # Square error [B, C, H, W]

    # Muliply square error by weights (batchwise)
    wse = w * se                                # [B, C, H, W]

    # Compute weighted mean for mse
    mse = torch.sum(wse) / (torch.sum(w) + eps)

    return mse

def plot_weighted_losses(num_epochs, train_losses, val_losses, model):
    epochs = np.arange(1, num_epochs + 1)

    
    plt.figure()
    plt.plot(epochs, train_losses, label = 'Train')
    plt.plot(epochs, val_losses, label = 'Validation')
    plt.xlabel('Epochs')
    plt.ylabel('Weighted MSE Loss')
    plt.legend()
    plt.title(f"{model}")

    plt.savefig(os.path.join(PATH_DEST, f'losses_{HEM}_{START_YEAR}_{END_YEAR}.png'))

    # plt.show()

    return

def main():

    # Set random seed for reproducibility
    set_seed(42)

    # Load input data
    fstr = f"{HEM}_{START_YEAR}_{END_YEAR}"
    x_train, y_train, r_train = torch.load(os.path.join(PATH_SOURCE,f'train_{fstr}.pt'))
    x_val, y_val, r_val = torch.load(os.path.join(PATH_SOURCE,f'val_{fstr}.pt'))
    x_test, y_test, r_test = torch.load(os.path.join(PATH_SOURCE,f'test_{fstr}.pt'))

    print("Input Data Loaded")

    # Use cuda if available
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        device = torch.device("cuda")
        print('Using CUDA')
    else:
        device = torch.device("cpu")
        print('USING CPU')


    # Move tensors to device
    x_train, y_train, r_train = x_train.to(device), y_train.to(device), r_train.to(device)
    x_val, y_val, r_val = x_val.to(device), y_val.to(device), r_val.to(device)
    x_test, y_test, r_test = x_test.to(device), y_test.to(device), r_test.to(device)

    # Define batch size
    batch_size = 365 # 365, Hoffman

    # Create Tensor Datasets
    trainData = DataLoader(TensorDataset(x_train, y_train, r_train), batch_size=batch_size, shuffle=True)
    valData = DataLoader(TensorDataset(x_val, y_val, r_val), batch_size=batch_size, shuffle=True)
    testData = DataLoader(TensorDataset(x_test, y_test, r_test), batch_size=batch_size, shuffle=False)

    print("Tensor datasets created")

    # Get input and output shapes for model
    _, n_in, ny, nx = x_train.shape
    n_out = y_train.shape[1]

    # Complile model
    model = WeightedCNN(n_in, n_out, ny, nx).to(device)

    print('model compiled')

    # Apply Xavier initialization to match TensorFlow default
    def init_weights(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            # Apply Xavier unniform to weights of Conv2d and Linear layers (TensorFlow default)
            nn.init.xavier_uniform_(m.weight)
            # Initialize bias to zero (TensorFlow default)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

        return

    # Apply 
    model.apply(init_weights)

    print('Xavier initialization complete') 

    # Define regularization
    weight_decay = 1e-4 # L2 Norm Regularization, changed from 0.01 in TensorFlow
    # NOTE TensorFLow multiplies Regularization by 0.05 0.01*0.05 -> 5e-4

    # Define Learning Rate
    # changed from 1e-3 in TensorFlow
    # NOTE changed back for scheduler
    lr = 1e-4

    # Initialize optimizer with weight decay (l2 regularization)
    optimizer = torch.optim.Adam(model.parameters(), lr = lr, weight_decay=weight_decay)

    # scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    # Define number of epochs
    num_epochs = 50 # Hoffman

    # Initialize losses
    train_losses = []
    val_losses = []

    # Train model
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for xb, yb, rb in tqdm(trainData, desc=f"Train Epoch {epoch+1}/{num_epochs}", leave=False):
            # print(torch.isnan(xb).any(), torch.isinf(xb).any())
            optimizer.zero_grad()
            preds = model(xb)
            loss = WeightedMSEloss(preds, yb, rb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0
        all_val_preds = []
        for xb, yb, rb in tqdm(valData, desc=f"Val   Epoch {epoch+1}/{num_epochs}", leave=False):
            with torch.no_grad():
                preds = model(xb)
                all_val_preds.append(preds.cpu())

                loss = WeightedMSEloss(preds, yb, rb)
                val_loss += loss.item()

        avg_train = train_loss / len(trainData)
        avg_val   = val_loss   / len(valData)

        train_losses.append(avg_train)
        val_losses.append(avg_val)


        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train:.4f} - Val Loss: {avg_val:.4f}")

        # # Step the scheduler
        # scheduler.step(avg_val)

        # # Print learning rate
        # for param_group in optimizer.param_groups:
        #     print(f"Epoch {epoch+1}: Learning Rate = {param_group['lr']}")

        # Test u and v preds
        all_val_preds = torch.cat(all_val_preds, dim=0)

        avg_u_pred = all_val_preds[:,0,:,:].mean().item()
        avg_v_pred = all_val_preds[:,1,:,:].mean().item()
        print(f"Epoch {epoch+1}/{num_epochs} - u Pred (Val) Avg: {avg_u_pred:.4f} - v Pred (Val) Avg: {avg_v_pred:.4f}")

    # Plot losses
    plot_weighted_losses(num_epochs, train_losses, val_losses, f"Weighted CNN")

    # Save model weights
    fnam = f'CNNweights_{HEM}_{START_YEAR}_{END_YEAR}.pth'
    torch.save(model.state_dict(), os.path.join(PATH_DEST,fnam))

    print('Model weights saved')

    # Evaluate trained model
    model.eval()

    # Get predictions on test set
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for xb, yb, rb in testData:
            preds = model(xb)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(yb.cpu().numpy())

    # Concatenate all batches
    y_pred = np.concatenate(all_preds, axis=0)
    y_true = np.concatenate(all_targets, axis=0)

    # Save to .npz
    fnam = f"CNNPreds_{HEM}_{START_YEAR}_{END_YEAR}.npz"
    np.savez(os.path.join(PATH_DEST, fnam), y_pred = y_pred, y_true = y_true)

    print("Predictions saved")

    return

if __name__ == "__main__":
    main()
