import os
from PIL import Image
import pandas as pd

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

# TODO: change normalization parameters
class NIHDataset(Dataset):
    def __init__(self, csv_file, img_list, img_path, transform=transforms.Compose([transforms.Normalize((0.5,), (1.0,))])):
        self.df = pd.read_csv(csv_file)
        self.data_list = [line[-1] for line in open(img_list)]
        self.img_path = img_path
        self.pil_to_tensor = transforms.Compose([transforms.PILToTensor()])
        self.transform = transform
        self.label_map = {"Atelectasis": 0, "Cardiomegaly": 1, "Effusion": 2, "Infiltration": 3, "Mass": 4, "Nodule": 5, "Pneumonia": 6,
                          "Pneumothorax": 7, "Consolidation": 8, "Edema": 9, "Emphysema": 10, "Fibrosis": 11,
                          "Pleural_Thickening": 12, "Hernia": 13}
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        # if torch.is_tensor(idx):
        #     idx = idx.tolist()

        img_name = self.df.iloc[idx]["Image Index"]
        img_file = os.path.join(self.img_path, img_name)
        str_labels = self.df.iloc[idx]["Finding Labels"].split('|')
        # label = self.df.iloc[idx]["Finding Labels"]

        labels = [0] * 14
        for diagnosis in str_labels:
            if diagnosis != "No Finding":
                labels[self.label_map[diagnosis]] = 1

        labels = torch.tensor(labels, dtype=torch.float32)

        # read image from file
        data = Image.open(img_file).convert("L") # convert image to grayscale if not already
        data = self.pil_to_tensor(data)
        data = data.to(torch.float32)

        if self.transform:
            features = self.transform(data)

        # if features.shape[0] > 1:
        #     print(idx, features.shape, len(label))
        
        return features, labels