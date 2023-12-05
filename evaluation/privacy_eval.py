"""Calculate privacy metrics for the given model.

1.get_datatset, and get feature outputs after sending all images 
through the inception model.

2.compare generated image's feature output with feature outputs from dataset, 
and calculate the L2 distance between them.

3.find the minimum distance between the generated images and the feature outputs.

4. send both images to the classifier, if their labels are different, 
then the privacy is protected."""


import os
import argparse
import torch
import pathlib
import pickle
from models.classifier_models import *
from torch import nn

def is_private(image1, image2, attribute='gender'):
    if attribute == 'gender' or attribute == 'attractive':
        model = MultiClassifier()
        if attribute == 'gender':
            model.load_state_dict(torch.load('/data/local/qipan/exp_celeba/celeba_cls_gender.pth'))
        else:
            model.load_state_dict(torch.load('/data/local/qipan/exp_celeba/celeba_cls_attractive.pth'))
    elif attribute == 'smile':
        model = CustomResNet50Model(num_classes=2)
        model.load_state_dict(torch.load('/data/local/qipan/exp_celeba/celeba_cls_smile_50.pth'))
    model.eval()
    model.cuda()
    attr1 = model(image1)
    attr2 = model(image2)
    if torch.argmax(attr1) != torch.argmax(attr2):
        return True
    else:
        return False
    

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from PIL import Image
    from glob import glob
    import torchvision.transforms as transforms
    from torchvision import models
    import torch.nn.functional as F
    
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

    class CustomResNet50Model(nn.Module):
        def __init__(self, num_classes):
            super(CustomResNet50Model, self).__init__()
            
            # Load the pretrained ResNet-50 model + higher level layers
            self.resnet50 = models.resnet50(pretrained=True)
            
            # Remove the final layer (classification head of original ResNet50)
            # To keep the feature extraction layers only
            self.features = nn.Sequential(*list(self.resnet50.children())[:-1])
            
            # Freeze the parameters of ResNet50 (make them non-trainable)
            for param in self.features.parameters():
                param.requires_grad = False
            
            # Define new classification head
            self.class_head = nn.Sequential(
                nn.Linear(self.resnet50.fc.in_features, num_classes), 
                nn.Softmax(dim=1)
            )
            
        def forward(self, x):
            x = self.features(x)
            
            # Global Average Pooling (GAP) layer
            x = x.mean([2, 3])
            
            # Classification head
            x = self.class_head(x)
            return x
    
    
    model = CustomResNet50Model(num_classes=2)
    model.load_state_dict(torch.load('/data/local/qipan/exp_celeba/celeba_cls_smile_50.pth'))
    model.eval()
    model.cuda()
    image_dir = "/data/local/qipan/exp_celeba/dpdm_samples/nearest_image/smile"
    image_paths = glob(os.path.join(image_dir, "*.png"))

    transform=transforms.Compose([
        transforms.Resize((128*2, 128)),
        transforms.ToTensor()
    
    ])
    
    private_score = 0
    for image_path in image_paths:
        img = transform(Image.open(image_path))
        ground_truth = img[:, :128, :]
        generated = img[:, 128:, :]
        print(image_path)
        print(torch.argmax(model(ground_truth.unsqueeze(0).cuda())))
        print(torch.argmax(model(generated.unsqueeze(0).cuda())))
        private_score += is_private(ground_truth.unsqueeze(0).cuda(), generated.unsqueeze(0).cuda(), attribute='smile')
    print(private_score/len(image_paths))
        