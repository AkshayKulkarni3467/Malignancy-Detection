import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from torch import nn
import os
from pathlib import Path
from utils import data_setup
from utils import engine_regnet_y


def train_regnet_y(epochs,device):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    image_path = Path('data/')
    image_path.is_dir()

    train_dir = image_path/'train'
    test_dir = image_path/'test'

    weights = torchvision.models.RegNet_Y_16GF_Weights.IMAGENET1K_V1

    auto_transforms = weights.transforms()

    train_dataloader,test_dataloader,class_names=data_setup.create_dataloaders(train_dir=train_dir,
                                                                               test_dir=test_dir,
                                                                               transform=auto_transforms,
                                                                               batch_size=32)

    model = torchvision.models.regnet_y_16gf(weights=weights)

    for param in model.parameters():
        param.requires_grad=False

    model.fc = nn.Linear(in_features=3024,out_features=5,bias=True)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(),lr = 0.001)
    results = engine_regnet_y.train(model = model,
                           train_dataloader=train_dataloader,
                           test_dataloader=test_dataloader,
                           optimizer=optimizer,
                           loss_fn=loss_fn,
                           epochs=epochs,
                           device=device)
    return results