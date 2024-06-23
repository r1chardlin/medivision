import os

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import global_vars
from dataset import NIHDataset
from model import ConvNet
from lib import test_classification_model, train_and_test_model

# hyperparameters
batch_size = 32
hidden_channels = 32
num_epochs = 30
lr = 1e-3

if __name__ == "__main__":
    # train_and_test_model(batch_size=batch_size, hidden_channels=hidden_channels, num_epochs=num_epochs, lr=lr)

    # this model was trained on both the training and validation sets,
    # but it is the first trained model so no hyperparameters were tuned in advance
    conv_model = torch.load(os.path.join(global_vars.model_path, "06-22-2024_12:28:59.pkl"))
    transform = transforms.Compose([transforms.Normalize((0.5,), (1.0,))])
    train_set = NIHDataset(csv_file=global_vars.csv_file, 
                           img_list=global_vars.train_list_file, 
                           img_path=global_vars.img_path, 
                           transform=transform)
    test_set = NIHDataset(csv_file=global_vars.csv_file, 
                          img_list=global_vars.test_list_file, 
                          img_path=global_vars.img_path, 
                          transform=transform)
    train_loader = DataLoader(dataset=train_set,
                              batch_size=batch_size,
                              shuffle=True)
    test_loader = DataLoader(dataset=test_set,
                             batch_size=batch_size,
                             shuffle=True)

    train_acc = test_classification_model(train_loader, conv_model)
    test_acc = test_classification_model(test_loader, conv_model)

    print(f"train accuracy: {train_acc}")
    print(f"test accuracy: {test_acc}")