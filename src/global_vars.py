import os
import torch
from datetime import datetime

# hyperparameters
batch_size = 32
hidden_channels = 32
num_epochs = 30
lr = 1e-3

# directories
model_path = "/Users/rhlin/gitStuffs/personal/medivision/saved_models/"
data_path = "/Users/rhlin/Documents/nih_xray_data/"
img_path = os.path.join(data_path, "images")

# files
# model_file = os.path.join(model_path, datetime.now().strftime("%m-%d-%Y_%H:%M:%S.pkl"))
model_file = os.path.join(model_path, f"{batch_size}_{hidden_channels}_{num_epochs}_{lr:e}.pkl")
csv_file = os.path.join(data_path, "Data_Entry_2017_v2020.csv")
train_val_list_file = os.path.join(data_path, "train_val_list.txt")
train_list_file = os.path.join(data_path, "train_list.txt")
val_list_file = os.path.join(data_path, "val_list.txt")
test_list_file = os.path.join(data_path, "test_list.txt")

# setting device
device = None
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

