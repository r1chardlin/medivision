import os

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import global_vars
from dataset import NIHDataset
from model import ConvNet3 as ConvNet
import lib
# from lib import test_classification_model, train_and_test_model, split_train_val

# TODO:
# 1. Fix normalization ^
# 2. Implement batch normalization ^
# 3. Implement weighted loss ^
# 4. Implement DenseNet

if __name__ == "__main__":
    # lib.split_train_val()
    # lib.edit_csv()
    # lib.split_csv()

    # train_set, val_set, test_set = lib.get_dataset_loaders(transform=None)
    # mean, std = lib.get_mean_and_std(train_loader)
    # print(f"mean: {mean}")
    # print(f"std: {std}")

    # lib.train_and_test_model(hidden_channels=global_vars.hidden_channels, num_epochs=global_vars.num_epochs, lr=global_vars.lr, weighted=True)

    # model_file = os.path.join(global_vars.model_path, "2024-07-25_v2_mps_32_32_10_1.00e-03.pkl")
    # model = torch.load(model_file)
    # lib.train_and_test_model(hidden_channels=global_vars.hidden_channels, num_epochs=global_vars.num_epochs, 
    #                          lr=global_vars.lr, weighted=True, model=model)

    train_set, val_set, test_set = lib.get_datasets()
    model_file = os.path.join(global_vars.model_path, "2024-07-26_v3_mps_32_32_5_1.00e-03.pkl")
    lib.log_test_metrics(model_file, train_set, val_set, test_set, threshold=0.7)