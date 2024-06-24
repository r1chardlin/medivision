import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import random
from tqdm import tqdm

import global_vars
from dataset import NIHDataset
from model import ConvNet

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

def train_classification_model(train_loader, model, num_epochs, lr=1e-1, print_freq=100):
    optimizer = optim.SGD(model.parameters(), lr=lr)  # create an SGD optimizer for the model parameters
    for epoch in tqdm(range(num_epochs)):

        # Iterate through the dataloader for each epoch
        for batch_idx, (imgs, labels) in tqdm(enumerate(train_loader)):
            # imgs (torch.Tensor):    batch of input images
            # labels (torch.Tensor):  batch labels corresponding to the inputs

            imgs = imgs.to(global_vars.device)
            labels = labels.to(global_vars.device)
            
            # implement the training loop using imgs, labels, and cross entropy loss
            optimizer.zero_grad()  
            pred = model.forward(imgs)  # run the forward pass through the model to compute predictions
            # print(pred)
            # print(labels)
            # loss = F.cross_entropy(pred, labels)
            loss = F.binary_cross_entropy_with_logits(pred, labels) # need use this instead of F.cross_entropy because labels are tensors
            loss.backward()  # compute the gradient wrt loss
            optimizer.step()  # performs a step of gradient descent      

            if (epoch + 1) % print_freq == 0:
                print('epoch {} loss {}'.format(epoch+1, loss.item()))
                
    return model  # return trained model

def test_classification_model(test_loader, model):
    accumulator = 0
    total = 0
    for batch_idx, (imgs, labels) in tqdm(enumerate(test_loader)):
        imgs = imgs.to(global_vars.device)
        labels = labels.to(global_vars.device)
        outputs = model(imgs)
        preds = outputs.data
        # preds = F.softmax(preds)
        preds = F.sigmoid(preds)
        # preds = keep_top_k_elements_per_row(preds, 9)
        preds = (preds > 0.7).to(torch.float32) # apply threshold
        accumulator += (preds == labels).sum().item()
        
        # # for single-label
        # _, preds = torch.max(outputs.data, 1)
        # accumulator += (preds == labels).sum().item()

        total += labels.size(0) * labels.size(1)
    return accumulator / total

def train_and_test_model(batch_size, hidden_channels, num_epochs, lr):
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
                              batch_size=batch_size,
                              shuffle=True)
    
    val_loader = DataLoader(dataset=val_set,
                              batch_size=batch_size,
                              shuffle=True)

    test_loader = DataLoader(dataset=test_set,
                             batch_size=batch_size,
                             shuffle=True)


    conv_model = ConvNet(input_channels=1, hidden_channels=hidden_channels)
    conv_model = conv_model.to(global_vars.device)
    print('the number of parameters:', sum(parameter.view(-1).size()[0] for parameter in conv_model.parameters()))

    conv_model = train_classification_model(train_loader, conv_model, num_epochs=num_epochs, lr=lr)
    torch.save(conv_model, global_vars.model_file)
    
    avg_train_acc = test_classification_model(train_loader, conv_model)
    avg_val_acc = test_classification_model(val_loader, conv_model)
    avg_test_acc = test_classification_model(test_loader, conv_model)

    print("train accuracy: ", avg_train_acc)
    print("validation accuracy:", avg_val_acc)
    print("test accuracy:", avg_test_acc)