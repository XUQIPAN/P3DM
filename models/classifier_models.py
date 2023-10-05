import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class MultiClassifier(nn.Module):
    def __init__(self):
        super(MultiClassifier, self).__init__()
        self.ConvLayer1 = nn.Sequential(
            nn.Conv2d(3, 64, 3), # 3, 256, 256
            nn.MaxPool2d(2), # op: 16, 127, 127
            nn.ReLU(), # op: 64, 127, 127
        )
        self.ConvLayer2 = nn.Sequential(
            nn.Conv2d(64, 128, 3), # 64, 127, 127   
            nn.MaxPool2d(2), #op: 128, 63, 63
            nn.ReLU() # op: 128, 63, 63
        )
        self.ConvLayer3 = nn.Sequential(
            nn.Conv2d(128, 256, 3), # 128, 63, 63
            nn.MaxPool2d(2), #op: 256, 30, 30
            nn.ReLU() #op: 256, 30, 30
        )
        
        self.Linear1 = nn.Linear(256*14*14, 64)
        self.Linear2 = nn.Linear(64, 2)
                
        
    def forward(self, x):
        x = self.ConvLayer1(x)
        x = self.ConvLayer2(x)
        x = self.ConvLayer3(x)

        x = x.view(x.size(0), -1)
        x = self.Linear1(x)
        x = self.Linear2(x)
        return F.softmax(x)
    

class CustomResNet18Model(nn.Module):
    def __init__(self, num_classes):
        super(CustomResNet18Model, self).__init__()
        
        # Load the pretrained ResNet-18 model + higher level layers
        self.resnet18 = models.resnet18(pretrained=True)
        
        # Remove the final layer (classification head of original ResNet18)
        # To keep the feature extraction layers only
        self.features = nn.Sequential(*list(self.resnet18.children())[:-1])
        
        # Freeze the parameters of ResNet18 (make them non-trainable)
        for param in self.features.parameters():
            param.requires_grad = False
        
        # Define new classification head
        self.class_head = nn.Sequential(
            nn.Linear(self.resnet18.fc.in_features, num_classes), 
            nn.Softmax(dim=1)
        )
        
    def forward(self, x):
        x = self.features(x)
        
        # Global Average Pooling (GAP) layer
        x = x.mean([2, 3])
        
        # Classification head
        x = self.class_head(x)
        return x