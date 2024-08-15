from torchvision import datasets,transforms
from PIL import Image
from torch.utils.data import DataLoader
from pathlib import Path
import os
from torch.utils.data import Dataset
import torch.nn as nn
import torch
from utils import engine_tinyVGG_base

class TinyVgg(nn.Module):
    def __init__(self,input_shape,output_shape,hidden_units):
        super().__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.MaxPool2d(kernel_size=2),
        )
        self.flatten_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units*8*8,
                      out_features=output_shape)
        )
    def forward(self,x):
        return self.flatten_layer(self.conv_block3(self.conv_block2(self.conv_block1(x))))


def find_classes(directory):
    class_names_found = sorted([entry.name for entry in list(os.scandir(directory))])
    class_to_idx = {class_name: i for i,class_name in enumerate(class_names_found)}
    return class_names_found,class_to_idx


def train_TinyVGG(epochs,device):
    train_dir = Path('data/train')
    test_dir = Path('data/test')
    
    data_transform = transforms.Compose([
        transforms.Resize(size=(64,64)),
        transforms.TrivialAugmentWide(num_magnitude_bins=31),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor()
    ])
    
    train_data = datasets.ImageFolder(root=train_dir,transform=data_transform,target_transform=None)
    test_data = datasets.ImageFolder(root=test_dir,transform=data_transform,target_transform=None)
    
    BATCH_SIZE = 32
    train_dataloader = DataLoader(dataset=train_data,batch_size=BATCH_SIZE,shuffle=True,num_workers=4)
    test_dataloader = DataLoader(dataset=test_data,batch_size=BATCH_SIZE,shuffle=False,num_workers=4)
    
    target_dir = train_dir
    class_names_found = sorted([entry.name for entry in list(os.scandir(target_dir))])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    loss_fn = nn.CrossEntropyLoss()
    model = TinyVgg(input_shape=3,hidden_units=20,output_shape=len(class_names_found)).to(device)
    optimizer = torch.optim.Adam(params=model.parameters(),lr=0.001)

    results = engine_tinyVGG_base.train(model = model,
                           train_dataloader=train_dataloader,
                           test_dataloader=test_dataloader,
                           optimizer=optimizer,
                           loss_fn=loss_fn,
                           epochs=epochs,
                           device=device)
    return results
