import os
import time
import copy
import numpy as np
import matplotlib.pyplot as plt
import torch

import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, models, transforms
from torchvision.models import resnet18, ResNet18_Weights

ddir = '/Users/a24/Desktop/pyskillup/insect'

batch_size = 4
num_workers = 0

data_transformers = {
    'train': transforms.Compose(
        [
         transforms.RandomResizedCrop(224), 
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor(),
         transforms.Normalize([0.490, 0.449, 0.411], [0.231, 0.221, 0.230])
        ]
    ),
    'val': transforms.Compose(
        [
         transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.490, 0.449, 0.411],[0.231, 0.221, 0.230])
        ]
    )
}

img_data = {
    k: datasets.ImageFolder(os.path.join(ddir, k), data_transformers[k])
    for k in ['train', 'val']
}
dloaders = {
    k: torch.utils.data.DataLoader(
        img_data[k], batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    for k in ['train', 'val']
}
dset_sizes = {x: len(img_data[x]) for x in ['train', 'val']}
classes = img_data['train'].classes

dvc = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
#dvc = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train(model, loss_func, optimizer, epochs=10):
   start = time.time()

   accuracy = 0.0

   for e in range(epochs):
        print(f'Epoch number {e}/{epochs - 1}')
        print('=' * 20)

        for dset in ['train', 'val']:
            if dset == 'train':
                model.train()  
            else:
                model.eval() 

            loss = 0.0
            successes = 0

            for imgs, tgts in dloaders[dset]:
                imgs = imgs.to(dvc)
                tgts = tgts.to(dvc)
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(dset == 'train'):
                    ops = model(imgs)
                    _, preds = torch.max(ops, 1)
                    loss_curr = loss_func(ops, tgts)
                    
                    if dset == 'train':
                        loss_curr.backward()
                        optimizer.step()

                loss += loss_curr.item() * imgs.size(0)
                successes += torch.sum(preds == tgts.data)
          
            loss_epoch = loss / dset_sizes[dset]
            accuracy_epoch = successes.float() / dset_sizes[dset]
            # accuracy_epoch = successes.double() / dset_sizes[dset] # window

            print(f'{dset} loss in this epoch: {loss_epoch}, accuracy in this epoch: {accuracy_epoch}')
            if dset == 'val' and accuracy_epoch > accuracy:
                accuracy = accuracy_epoch      

   time_delta = time.time() - start
   print(f'Training finished in {time_delta // 60}mins {time_delta % 60}secs')
   print(f'Best validation set accuracy: {accuracy}')


   return model

# fine-tuning : All
# model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
# num_fcls = model.fc.in_features
# model.fc = nn.Linear(num_fcls, 2)

# if torch.backends.mps.is_available():
#     model = model.to("mps")
    
# loss_func = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr = 0.0001)

# train(model, loss_func, optimizer, epochs = 5)

# fine-tuning: only classifier
# model_conv = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
# for param in model_conv.parameters():
#     param.requires_grad = False # freeze
    
# num_fcls = model_conv.fc.in_features
# model_conv.fc = nn.Linear(num_fcls, 2)

# if torch.backends.mps.is_available():
#     model_conv = model_conv.to("mps")
    
# loss_func = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model_conv.parameters(), lr = 0.0001)

# train(model_conv, loss_func, optimizer, epochs = 5)

# fine-tuning : back part of model and classifier
model_conv = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
for name, param in model_conv.named_parameters():
    param.requires_grad = False # freeze
    #print('name:', name)
    if 'layer4.1' in name:
        param.requires_grad = True

# for name, child in model_conv.named_children():
#     for param in child.parameters():
#         print(name, param)
    
num_fcls = model_conv.fc.in_features
model_conv.fc = nn.Linear(num_fcls, 2)

if torch.backends.mps.is_available():
    model_conv = model_conv.to("mps")
    
loss_func = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_conv.parameters(), lr = 0.0001)

train(model_conv, loss_func, optimizer, epochs = 5)
    