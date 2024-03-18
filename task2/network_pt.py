# network module
# adapted from: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.transforms as transforms
from typing import List
import torchvision.models as models


# VisonTransformer Network
# reference: https://huggingface.co/docs/transformers/model_doc/vit

class Net(nn.Module):
    def __init__(self, num_classes):
        super(Net, self).__init__()
        self.model = models.VisionTransformer(
            num_classes = num_classes,
            image_size = 32, # size(resolution) of each image
            patch_size = 16, # size(resolution) of each patch
            num_layers = 12, # number of hidden layers
            num_heads = 12, # number of a attention heads 
            hidden_dim = 384, # dimensionality of the encoder layers and the pooler layer (default 768)
            mlp_dim = 1536, # dimensionality of the intermediate mlp layer (default 3072)
        )
    
    def forward(self, x):
        return self.model(x)


# class Net(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv1 = nn.Conv2d(3, 6, 5)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(6, 16, 5)
#         self.fc1 = nn.Linear(16 * 5 * 5, 120)
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, 10)

#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = torch.flatten(x, 1) # flatten all dimensions except batch
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x