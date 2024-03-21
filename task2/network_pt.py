# network module

import torch
import torch.nn as nn
from torchvision.models import VisionTransformer


# VisonTransformer Network
# reference: https://huggingface.co/docs/transformers/model_doc/vit

class Net(nn.Module):
    def __init__(self, num_classes):
        super(Net, self).__init__()
        self.model = VisionTransformer(
            image_size = 32, # size of each CIFAR10 image
            patch_size = 16, # size of each patch
            num_layers = 12, # number of hidden layers
            num_heads = 12, # number of a attention heads 
            hidden_dim = 384, # dimensionality of the encoder layers and the pooler layer (default 768)
            mlp_dim = 1536, # dimensionality of the intermediate mlp layer (default 3072)
            num_classes = num_classes,
        )
    
    def forward(self, x):
        return self.model(x)

