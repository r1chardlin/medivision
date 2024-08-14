import os
from PIL import Image

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch.nn.functional as F

import global_vars

class NIHDataset(Dataset):
    def __init__(self, csv_file, img_path, transform=None):
        self.df = pd.read_csv(csv_file)
        self.img_path = img_path
        self.transform = transforms.Compose([transforms.ToTensor(), transform])
        self.label_map = {"Atelectasis": 0, "Cardiomegaly": 1, "Effusion": 2, "Infiltration": 3, "Mass": 4, "Nodule": 5, "Pneumonia": 6,
                          "Pneumothorax": 7, "Consolidation": 8, "Edema": 9, "Emphysema": 10, "Fibrosis": 11,
                          "Pleural_Thickening": 12, "Hernia": 13}
        
        self.pathologies = [col for col in self.df.columns if col != "Image Index"]
        self.labels = self.df[self.pathologies].to_numpy().astype(np.float32)

        self.n_classes = len(self.pathologies)
        
        self.pos_weight, self.neg_weight = self.get_weights()

        self.label_ratio = torch.tensor((self.labels == 0).sum() / (self.labels == 1).sum(), dtype=torch.float32, device=global_vars.device)
    
    def __len__(self):
        return self.labels.shape[0]

    def get_weights(self):
        positives = (self.labels == 1).sum(axis=0)
        negatives = (self.labels == 0).sum(axis=0)
        total = positives + negatives

        # assign opposite weights to give weight positives higher and negatives lower
        pos_weight = negatives / total
        neg_weight = positives / total

        # convert to tensor
        pos_weight = torch.tensor(pos_weight, dtype=torch.float32, device=global_vars.device)
        neg_weight = torch.tensor(neg_weight, dtype=torch.float32, device=global_vars.device)

        return pos_weight, neg_weight
    
    def weighted_loss(self, pred, labels):
        weights = labels * self.pos_weight + (labels == 0) * self.neg_weight
        weights.to(global_vars.device)
        loss = 0.0
        for i in range(self.n_classes):
            # loss += F.binary_cross_entropy_with_logits(pred[:,i], labels[:,i], weight=weights[:,i])
            loss += F.binary_cross_entropy_with_logits(pred[:,i], labels[:,i], pos_weight=self.label_ratio)
        return loss / self.n_classes
    
    def __getitem__(self, idx):
        img_name = self.df["Image Index"][idx] # get name of x-ray file
        img_file = os.path.join(self.img_path, img_name)
        label = torch.tensor(self.labels[idx])

        # read image from file
        data = Image.open(img_file).convert("L") # convert image to grayscale if not already
        features = self.transform(data)
        
        return features, label


class NIHDatasetOLD2(Dataset):
    def __init__(self, csv_file, img_list, img_path, transform=None):
        self.df = pd.read_csv(csv_file)
        self.data_list = [line[:-1] for line in open(img_list)]
        self.img_path = img_path
        self.transform = transforms.Compose([transforms.ToTensor(), transform])
        self.label_map = {"Atelectasis": 0, "Cardiomegaly": 1, "Effusion": 2, "Infiltration": 3, "Mass": 4, "Nodule": 5, "Pneumonia": 6,
                          "Pneumothorax": 7, "Consolidation": 8, "Edema": 9, "Emphysema": 10, "Fibrosis": 11,
                          "Pleural_Thickening": 12, "Hernia": 13}
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        img_name = self.data_list[idx] # get name of x-ray file
        img_file = os.path.join(self.img_path, img_name)
        labels = torch.tensor(list(self.df[self.df["Image Index"] == img_name].iloc[0,1:]), dtype=torch.float32)

        # read image from file
        data = Image.open(img_file).convert("L") # convert image to grayscale if not already
        features = self.transform(data)
        
        return features, labels


class NIHDatasetOLD(Dataset):
    def __init__(self, csv_file, img_list, img_path, transform=None):
        self.df = pd.read_csv(csv_file)
        self.data_list = [line[:-1] for line in open(img_list)]
        self.img_path = img_path
        # self.pil_to_tensor = transforms.Compose([transforms.ToTensor()])
        self.transform = transforms.Compose([transforms.ToTensor(), transform])
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
        # data = self.pil_to_tensor(data)
        # data = data.to(torch.float32)

        # apply transformation to image/features
        # features = data
        # if self.transform:
        #     features = self.transform(data)

        features = self.transform(data)
        
        return features, labels