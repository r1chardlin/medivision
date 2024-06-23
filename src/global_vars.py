import os
import torch
from datetime import datetime

model_path = "/Users/rhlin/gitStuffs/personal/medivision/saved_models/"
data_path = "/Users/rhlin/Documents/nih_xray_data/"
img_path = os.path.join(data_path, "images")

model_file = os.path.join(model_path, datetime.now().strftime("%m-%d-%Y_%H:%M:%S.pkl"))
csv_file = os.path.join(data_path, "Data_Entry_2017_v2020.csv")
train_list_file = os.path.join(data_path, "train_val_list.txt") # TODO: split train and validation sets 
test_list_file = os.path.join(data_path, "test_list.txt")

device = None
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

