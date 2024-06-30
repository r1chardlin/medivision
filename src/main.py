import os

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import global_vars
from dataset import NIHDataset
from model import ConvNet
import lib
# from lib import test_classification_model, train_and_test_model, split_train_val

if __name__ == "__main__":
    # lib.split_train_val()

    # train_loader, val_loader, test_loader = lib.get_dataset_loaders(transform=None)
    # mean, std = lib.get_mean_and_std(train_loader)
    # print(f"mean: {mean}")
    # print(f"std: {std}")

    lib.train_and_test_model(hidden_channels=global_vars.hidden_channels, num_epochs=global_vars.num_epochs, lr=global_vars.lr)

    # train_loader, val_loader, test_loader = lib.get_dataset_loaders()
    # model_file = os.path.join(global_vars.model_path, "2024-06-28_v2_mps_32_32_30_1.00e-03.pkl")
    # lib.log_test_metrics(model_file, train_loader, val_loader, test_loader, threshold=0.7, train=False, test=False)