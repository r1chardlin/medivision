import os
import torch
from datetime import date, datetime

from model import ConvNet

# hyperparameters
batch_size = 32
hidden_channels = 32
num_epochs = 30
lr = 1e-1
mean = 128.93533325195312
std = 59.14480209350586

# setting device
device = None
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

# directories
medivision_path = "/Users/rhlin/gitStuffs/personal/medivision/"
model_path = os.path.join(medivision_path, "saved_models")
log_path = os.path.join(medivision_path, "log")
data_path = "/Users/rhlin/Documents/nih_xray_data/"
img_path = os.path.join(data_path, "images")

# files
model_name = ConvNet().name
# model_file = os.path.join(model_path, datetime.now().strftime("%m-%d-%Y_%H:%M:%S.pkl"))
model_file = os.path.join(model_path, f"{date.today()}_{model_name}_{device.type}_{batch_size}_{hidden_channels}_{num_epochs}_{lr:.2e}.pkl")
csv_file = os.path.join(data_path, "Data_Entry_2017_v2020.csv")
train_val_list_file = os.path.join(data_path, "train_val_list.txt")
train_list_file = os.path.join(data_path, "train_list.txt")
val_list_file = os.path.join(data_path, "val_list.txt")
test_list_file = os.path.join(data_path, "test_list.txt")
metrics_log_file = os.path.join(log_path, "metrics_log.txt")