import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import torchinfo
from torchvision import transforms
from torch import nn
import os
from pathlib import Path
from utils import data_setup
from utils import engine_convnext_small


def train_convnext_small(epochs,device):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    image_path = Path('data/')
    image_path.is_dir()
    
    train_dir = image_path/'train'
    test_dir = image_path/'test'
    
    weights = torchvision.models.ConvNeXt_Small_Weights.IMAGENET1K_V1
    
    auto_transforms = weights.transforms()
    
    train_dataloader,test_dataloader,class_names=data_setup.create_dataloaders(train_dir=train_dir,
                                                                               test_dir=test_dir,
                                                                               transform=auto_transforms,
                                                                               batch_size=32)
    
    model = torchvision.models.convnext_small(weights=weights)
    
    for param in model.parameters():
        param.requires_grad=False
        
    model.classifier[2] = nn.Linear(in_features=768,out_features=768,bias=True).to(device)
    model.classifier.append(nn.Linear(in_features=768,out_features=8,bias=True).to(device))
    
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(),lr = 0.001)
    results = engine_convnext_small.train(model = model,
                           train_dataloader=train_dataloader,
                           test_dataloader=test_dataloader,
                           optimizer=optimizer,
                           loss_fn=loss_fn,
                           epochs=epochs,
                           device=device)
    return results