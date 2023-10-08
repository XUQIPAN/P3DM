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
from torchvision.utils import save_image
from .inception import InceptionV3
from .fid_score import get_activations
import pathlib
import pickle
from models.classifier_guidance import MultiClassifier, CustomResNet18Model
from torch import nn

def is_private(image1, image2, attribute='gender'):
    if attribute == 'gender' or attribute == 'attractive':
        model = MultiClassifier()
        if attribute == 'gender':
            model.load_state_dict(torch.load('/data/local/qipan/exp_celeba/celeba_cls_gender.pth'))
        else:
            model.load_state_dict(torch.load('/data/local/qipan/exp_celeba/celeba_cls_attractive.pth'))
    elif attribute == 'smile':
        model = CustomResNet18Model(num_classes=2)
        model.load_state_dict(torch.load('/data/local/qipan/exp_celeba/celeba_cls_smile.pth'))
    model.cuda()
    attr1 = model(image1)
    attr2 = model(image2)
    if torch.argmax(attr1) != torch.argmax(attr2):
        return True
    else:
        return False
    

if __name__ == "__main__":
    model = MultiClassifier()
    model.load_state_dict(torch.load('/data/local/qipan/exp_celeba/celeba_cls_attractive.pth'))