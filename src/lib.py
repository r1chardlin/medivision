import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn import metrics

import os
import random
import numpy as np
from tqdm import tqdm

import global_vars
from dataset import NIHDataset
from model import ConvNet3 as ConvNet

def split_train_val(train_val_file=global_vars.train_val_list_file, train_file=global_vars.train_list_file, 
                    val_file=global_vars.val_list_file, val_ratio=0.1):
    train_val_file = open(train_val_file, 'r')
    imgs = train_val_file.readlines()
    imgs[-1] += '\n'
    random.shuffle(imgs)
    split = int(val_ratio * len(imgs))
    val_list = imgs[:split]
    train_list = imgs[split:]
    with open(val_file, 'w') as f:
        for img in val_list:
            f.write(img)
    with open(train_file, 'w') as f:
        for img in train_list:
            f.write(img)
    train_val_file.close()

def edit_csv(csv_file=global_vars.csv_file, out_file=global_vars.edited_csv_file):
    out_file = open(out_file, 'w')
    out_file.write("Image Index,Atelectasis,Cardiomegaly,Effusion,Infiltration,Mass,Nodule,Pneumonia,\
                   Pneumothorax,Consolidation,Edema,Emphysema,Fibrosis,Pleural_Thickening,Hernia\n")
    
    label_map = {"Atelectasis": 0, "Cardiomegaly": 1, "Effusion": 2, "Infiltration": 3, "Mass": 4, 
                 "Nodule": 5, "Pneumonia": 6, "Pneumothorax": 7, "Consolidation": 8, "Edema": 9, 
                 "Emphysema": 10, "Fibrosis": 11, "Pleural_Thickening": 12, "Hernia": 13}
    
    with open(csv_file, 'r') as f:
        skip = True
        for line in f:
            if skip:
                skip = False
                continue
            row = line.split(',')
            img_name = row[0]
            labels = row[1].split('|')
            new_labels = ["0"] * 14
            for label in labels:
                if label != "No Finding":
                    new_labels[label_map[label]] = "1"
            new_labels = ','.join(new_labels)
            new_line = f"{img_name},{new_labels}\n"
            out_file.write(new_line)
    out_file.close()

def split_csv(train_lst=global_vars.train_list_file, val_lst=global_vars.val_list_file, test_lst=global_vars.test_list_file,
              train_csv=global_vars.train_csv_file, val_csv=global_vars.val_csv_file, test_csv=global_vars.test_csv_file,
              csv_file=global_vars.edited_csv_file):
    train_lst = {line.strip() for line in open(train_lst)}
    val_lst = {line.strip() for line in open(val_lst)}
    test_lst = {line.strip() for line in open(test_lst)}

    train_csv = open(train_csv, 'w')
    val_csv = open(val_csv, 'w')
    test_csv = open(test_csv, 'w')
    with open(csv_file, 'r') as f:
        header = True
        for line in f:
            if header:
                train_csv.write(line)
                val_csv.write(line)
                test_csv.write(line)
                header = False
                continue
            row = line.split(',')
            img = row[0]
            if img in train_lst:
                train_csv.write(line)
            elif img in val_lst:
                val_csv.write(line)
            elif img in test_lst:
                test_csv.write(line)
    train_csv.close()
    val_csv.close()
    test_csv.close()

def get_mean_and_std(dataset):
    mean = 0.0
    std = 0.0
    total_samples = 0

    data_loader = DataLoader(dataset=dataset,
                             batch_size=global_vars.batch_size,
                             shuffle=True)

    for images, _ in tqdm(data_loader):
        batch_samples = images.size(0) # get number of samples in the current batch
        images = images.view(batch_samples, images.size(1), -1) # reshape images to (batch_size, channels, height * width)
        mean += images.mean(2).sum(0) # calculate mean of each image and sum across batches
        std += images.std(2).sum(0)   # calculate std of each image and sum across batches
        
        total_samples += batch_samples

    mean /= total_samples
    std /= total_samples
    return mean.item(), std.item()

def keep_top_k_elements(tensor, k):
    # flatten the tensor and get the top k values and their indices
    values, indices = torch.topk(tensor.view(-1), k)
    
    # create a mask with all zeros
    mask = torch.zeros_like(tensor, dtype=torch.bool, device=global_vars.device)
    
    # set positions of top k elements in the mask to True
    mask.view(-1)[indices] = True
    
    # create a tensor where only the top k elements are kept all others are set to zero
    result = torch.where(mask, tensor, torch.tensor(0, dtype=tensor.dtype, device=global_vars.device))

    return result

def keep_top_k_elements_per_row(tensor, k):
    # Get the top k values and their indices for each row
    topk_values, topk_indices = torch.topk(tensor, k, dim=1)
    
    # Create a mask for the top k elements
    mask = torch.zeros_like(tensor, device=global_vars.device).scatter(1, topk_indices, 1)
    
    # Apply the mask to retain only the top k elements
    result = tensor * mask.to(torch.float32)
    
    return result

def get_datasets(transform=transforms.Compose([transforms.Normalize((global_vars.mean,), (global_vars.std,))])):
    train_set = NIHDataset(csv_file=global_vars.train_csv_file,
                           img_path=global_vars.img_path, 
                           transform=transform)
    
    val_set = NIHDataset(csv_file=global_vars.val_csv_file, 
                           img_path=global_vars.img_path, 
                           transform=transform)
    
    test_set = NIHDataset(csv_file=global_vars.test_csv_file, 
                          img_path=global_vars.img_path, 
                          transform=transform)

    return train_set, val_set, test_set

# TODO: implement weighted loss
def train_model(train_set, model, num_epochs, lr, weighted=False, print_freq=1):
    # optimizer = optim.SGD(model.parameters(), lr=lr)  # create an SGD optimizer for the model parameters
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # pos_weight = torch.tensor(20, device=global_vars.device)
    pos_weight = torch.tensor(21.40322216057374, device=global_vars.device)

    train_loader = DataLoader(dataset=train_set,
                              batch_size=global_vars.batch_size,
                              shuffle=True)
    
    for epoch in tqdm(range(num_epochs)):
        # Iterate through the dataloader for each epoch
        loss_acc = 0
        for batch_idx, (imgs, labels) in tqdm(enumerate(train_loader)):
            # imgs (torch.Tensor):    batch of input images
            # labels (torch.Tensor):  batch labels corresponding to the inputs

            imgs = imgs.to(global_vars.device)
            labels = labels.to(global_vars.device)
            
            # implement the training loop using imgs, labels, and cross entropy loss
            optimizer.zero_grad()  
            pred = model.forward(imgs)  # run the forward pass through the model to compute predictions
            

            # loss = F.cross_entropy(pred, labels) # for multiclass classification

            # loss = F.binary_cross_entropy_with_logits(pred, labels, pos_weight=pos_weight) # use this instead of F.cross_entropy for multilabel classification
            # loss = F.multilabel_soft_margin_loss(pred, labels)
            # loss = F.binary_cross_entropy_with_logits(pred, labels)
            # loss = F.multilabel_soft_margin_loss(pred, labels)

            loss = train_set.weighted_loss(pred, labels) if weighted else F.multilabel_soft_margin_loss(pred, labels)
            loss.backward()  # compute the gradient wrt loss
            optimizer.step()  # performs a step of gradient descent

            loss_acc += loss.item()      

        if (epoch + 1) % print_freq == 0:
            print('epoch {} loss {}'.format(epoch + 1, loss_acc / global_vars.batch_size))
                
    return model  # return trained model

def test_model(dataset, model, threshold=0.7):
    accuracy_acc = 0
    precision_acc = 0
    recall_acc = 0
    accuracy_total = 0
    precision_batch_count = 0
    recall_batch_count = 0

    data_loader = DataLoader(dataset=dataset,
                             batch_size=global_vars.batch_size,
                             shuffle=True)

    for batch_idx, (imgs, labels) in tqdm(enumerate(data_loader)):
        imgs = imgs.to(global_vars.device)
        labels = labels.to(global_vars.device)
        outputs = model(imgs)

        # # for single-label
        # _, preds = torch.max(outputs.data, 1)
        # accumulator += (preds == labels).sum().item()

        preds = outputs.data
        # preds = F.softmax(preds)
        preds = F.sigmoid(preds)
        # preds = keep_top_k_elements_per_row(preds, 9)
        preds = (preds > threshold).to(torch.float32) # apply threshold

        accuracy_acc += (preds == labels).sum().item()
        accuracy_total += labels.size(0) * labels.size(1)

        preds = preds.cpu().numpy()
        labels = labels.cpu().numpy()

        precision = metrics.precision_score(y_true=labels, y_pred=preds, average="samples", zero_division=np.nan)
        recall = metrics.recall_score(y_true=labels, y_pred=preds, average="samples", zero_division=np.nan)
        
        # print()
        if precision is not np.nan:
            # print(f"precision: {precision}")
            precision_acc += precision
            precision_batch_count += 1
        if recall is not np.nan:
            # print(f"recall: {recall}")
            recall_acc += recall
            recall_batch_count += 1
        

        # indices = np.where(preds.sum(axis=1) > 0)
        # if indices[0].shape[0] > 0:
        #     precision_acc += metrics.precision_score(y_true=labels[indices], y_pred=preds[indices], average="micro", zero_division=1.0)
        #     recall_acc += metrics.recall_score(y_true=labels[indices], y_pred=preds[indices], average="micro", zero_division=1.0)
            
        #     # print(precision_acc)
        #     # accuracy_total += labels.size(0) * labels.size(1)
        #     batch_count += 1

    print(precision_acc, precision_batch_count)
    print(recall_acc, recall_batch_count)
    
    return accuracy_acc / accuracy_total, precision_acc / precision_batch_count, recall_acc / recall_batch_count
    # return accuracy_acc / accuracy_total, recall_acc / recall_batch_count

def log_test_metrics(model_file, train_loader, val_loader, test_loader, threshold=0.7, log_file=global_vars.metrics_log_file, 
                     train=True, val=True, test=True, print_flag=True):
    if not train and not val and not test:
        return
    
    # train_loader, val_loader, test_loader = get_dataset_loaders()
    model = torch.load(model_file)
    model_name = model_file.split('/')[-1]

    with open(log_file, 'a') as f:
        f.write(f"{model_name}, threshold: {threshold}\n\n")
    
        if train:
            train_acc, train_precision, train_recall = test_model(train_loader, model, threshold)

            f.write(f"train accuracy: {train_acc}\n")
            f.write(f"train precision: {train_precision}\n")
            f.write(f"train recall: {train_recall}\n\n")

            if print_flag:
                print(f"train accuracy: {train_acc}")
                print(f"train precision: {train_precision}")
                print(f"train recall: {train_recall}\n")

        if val:
            val_acc, val_precision, val_recall = test_model(val_loader, model, threshold)
            # val_acc, val_recall = test_classification_model(val_loader, model, threshold)

            f.write(f"validation accuracy: {val_acc}\n")
            f.write(f"validation precision: {val_precision}\n")
            f.write(f"validation recall: {val_recall}\n\n")
            
            if print_flag:
                print(f"validation accuracy: {val_acc}")
                print(f"validation precision: {val_precision}")
                print(f"validation recall: {val_recall}\n")

        if test:
            test_acc, test_precision, test_recall = test_model(test_loader, model, threshold)

            f.write(f"test accuracy: {test_acc}\n")
            f.write(f"test precision: {test_precision}\n")
            f.write(f"test recall: {test_recall}\n\n")

            if print_flag:
                print(f"test accuracy: {test_acc}")
                print(f"test precision: {test_precision}")
                print(f"test recall: {test_recall}\n")

        f.write('\n')

def train_and_test_model(hidden_channels, num_epochs, lr, weighted=True, threshold=0.7, model=None):
    train_set, val_set, test_set = get_datasets()

    if not model:
        model = ConvNet(input_channels=1, hidden_channels=hidden_channels)
    model = model.to(global_vars.device)
    print('number of parameters:', sum(parameter.view(-1).size()[0] for parameter in model.parameters()))

    model = train_model(train_set, model, weighted=weighted, num_epochs=num_epochs, lr=lr)
    # conv_model = train_classification_model(val_loader, conv_model, num_epochs=num_epochs, lr=lr) # TODO: delete this line
    torch.save(model, global_vars.model_file)
    
    log_test_metrics(global_vars.model_file, train_set, val_set, test_set, threshold=threshold)