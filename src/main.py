import os

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import global_vars
from dataset import NIHDataset
from model import ConvNet
from lib import test_classification_model, train_and_test_model, split_train_val

if __name__ == "__main__":
    # split_train_val()

    # train_and_test_model(batch_size=global_vars.batch_size, hidden_channels=global_vars.hidden_channels, 
    #                      num_epochs=global_vars.num_epochs, lr=global_vars.lr)

    conv_model = torch.load(os.path.join(global_vars.model_path, "32_32_30_1.000000e-03.pkl"))
    transform = transforms.Compose([transforms.Normalize((0.5,), (1.0,))])
    train_set = NIHDataset(csv_file=global_vars.csv_file, 
                           img_list=global_vars.train_list_file, 
                           img_path=global_vars.img_path, 
                           transform=transform)
    val_set = NIHDataset(csv_file=global_vars.csv_file, 
                          img_list=global_vars.val_list_file, 
                          img_path=global_vars.img_path, 
                          transform=transform)
    test_set = NIHDataset(csv_file=global_vars.csv_file, 
                          img_list=global_vars.test_list_file, 
                          img_path=global_vars.img_path, 
                          transform=transform)
    train_loader = DataLoader(dataset=train_set,
                              batch_size=global_vars.batch_size,
                              shuffle=True)
    val_loader = DataLoader(dataset=val_set,
                              batch_size=global_vars.batch_size,
                              shuffle=True)
    test_loader = DataLoader(dataset=test_set,
                             batch_size=global_vars.batch_size,
                             shuffle=True)

    train_acc = test_classification_model(train_loader, conv_model)
    val_acc = test_classification_model(val_loader, conv_model)
    test_acc = test_classification_model(test_loader, conv_model)

    print(f"train accuracy: {train_acc}")
    print(f"validation accuracy: {val_acc}")
    print(f"test accuracy: {test_acc}")




    # print(len(train_set))
    # print(len(test_set))

    # print(train_set.__getitem__(0))
    # print(test_set.__getitem__(1))

    # train_set.__getitem__(0)
    # test_set.__getitem__(0)