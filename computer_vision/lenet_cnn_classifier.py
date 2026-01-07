# python imports
import os
from tqdm import tqdm

# torch imports
import torch
import torch.nn as nn
import torch.optim as optim

# helper functions for computer vision
import torchvision
import torchvision.transforms as transforms


class LeNet(nn.Module):
    def __init__(self, input_shape=(32, 32), num_classes=100):
        super(LeNet, self).__init__()
        
        # Layer 1: Conv layer with 6 output channels, kernel_size=5, stride=1
        # Input: 3 channels (RGB), Output: 6 channels
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1)
        
        # Layer 2: Conv layer with 16 output channels, kernel_size=5, stride=1
        # Input: 6 channels, Output: 16 channels
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1)
        
        # Calculate the size after conv and pooling layers
        # After conv1: (32-5+1) = 28, after pool1: 28/2 = 14
        # After conv2: (14-5+1) = 10, after pool2: 10/2 = 5
        # So we have 16 channels * 5 * 5 = 400 features
        
        # Layer 4: Linear layer with output dimension = 256
        self.fc1 = nn.Linear(16 * 5 * 5, 256)
        
        # Layer 5: Linear layer with output dimension = 128
        self.fc2 = nn.Linear(256, 128)
        
        # Layer 6: Linear layer with output dimension = num_classes (100)
        self.fc3 = nn.Linear(128, num_classes)
        
        # Activation and pooling layers
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        shape_dict = {}
        
        # Stage 1: Conv1 -> ReLU -> MaxPool
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        shape_dict[1] = list(x.shape)
        
        # Stage 2: Conv2 -> ReLU -> MaxPool
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        shape_dict[2] = list(x.shape)
        
        # Stage 3: Flatten
        x = x.view(x.size(0), -1)  # Flatten the tensor
        shape_dict[3] = list(x.shape)
        
        # Stage 4: FC1 -> ReLU
        x = self.fc1(x)
        x = self.relu(x)
        shape_dict[4] = list(x.shape)
        
        # Stage 5: FC2 -> ReLU
        x = self.fc2(x)
        x = self.relu(x)
        shape_dict[5] = list(x.shape)
        
        # Stage 6: FC3 (output layer)
        out = self.fc3(x)
        shape_dict[6] = list(out.shape)
        
        return out, shape_dict


def count_model_params():
    '''
    return the number of trainable parameters of LeNet.
    '''
    model = LeNet()
    model_params = 0.0
    
    # Count all trainable parameters
    for name, param in model.named_parameters():
        if param.requires_grad:
            model_params += param.numel()
    
    # Convert to millions
    model_params = model_params / 1e6
    
    return model_params


def train_model(model, train_loader, optimizer, criterion, epoch):
    """
    model (torch.nn.module): The model created to train
    train_loader (pytorch data loader): Training data loader
    optimizer (optimizer.*): A instance of some sort of optimizer, usually SGD
    criterion (nn.CrossEntropyLoss) : Loss function used to train the network
    epoch (int): Current epoch number
    """
    model.train()
    train_loss = 0.0
    for input, target in tqdm(train_loader, total=len(train_loader)):
        ###################################
        # fill in the standard training loop of forward pass,
        # backward pass, loss computation and optimizer step
        ###################################

        # 1) zero the parameter gradients
        optimizer.zero_grad()
        # 2) forward + backward + optimize
        output, _ = model(input)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        # Update the train_loss variable
        # .item() detaches the node from the computational graph
        # Uncomment the below line after you fill block 1 and 2
        train_loss += loss.item()

    train_loss /= len(train_loader)
    print('[Training set] Epoch: {:d}, Average loss: {:.4f}'.format(epoch+1, train_loss))

    return train_loss


def test_model(model, test_loader, epoch):
    model.eval()
    correct = 0
    with torch.no_grad():
        for input, target in test_loader:
            output, _ = model(input)
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_acc = correct / len(test_loader.dataset)
    print('[Test set] Epoch: {:d}, Accuracy: {:.2f}%\n'.format(
        epoch+1, 100. * test_acc))

    return test_acc