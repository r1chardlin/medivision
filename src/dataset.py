import os
from PIL import Image
import pandas as pd

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class NIHDataset(Dataset):
    def __init__(self, csv_file, img_list, img_path, transform=None):
        self.df = pd.read_csv(csv_file)
        self.data_list = [line[:-1] for line in open(img_list)]
        self.img_path = img_path
        self.pil_to_tensor = transforms.Compose([transforms.PILToTensor()])
        self.transform = transform
        self.label_map = {"Atelectasis": 0, "Cardiomegaly": 1, "Effusion": 2, "Infiltration": 3, "Mass": 4, "Nodule": 5, "Pneumonia": 6,
                          "Pneumothorax": 7, "Consolidation": 8, "Edema": 9, "Emphysema": 10, "Fibrosis": 11,
                          "Pleural_Thickening": 12, "Hernia": 13}
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        img_name = self.data_list[idx] # get name of x-ray file
        img_file = os.path.join(self.img_path, img_name)
        str_labels = self.df[self.df["Image Index"] == img_name]["Finding Labels"].item().split('|') # get corresponding labels of x-ray

        # create the label tensor
        labels = [0] * 14
        for diagnosis in str_labels:
            if diagnosis != "No Finding":
                labels[self.label_map[diagnosis]] = 1
        labels = torch.tensor(labels, dtype=torch.float32)

        # read image from file
        data = Image.open(img_file).convert("L") # convert image to grayscale if not already
        data = self.pil_to_tensor(data)
        data = data.to(torch.float32)

        # apply transformation to image/features
        features = data
        if self.transform:
            features = self.transform(data)
        
        return features, labels