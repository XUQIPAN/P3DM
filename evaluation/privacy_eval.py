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
import sys
sys.path.append(str(pathlib.Path(__file__).parent.parent))
from models.guided_diffusion.script_util import create_classifier
from torch import nn
path = '/data/local/ml01/qipan/exp_celeba/logs/NOISY_CLASSIFIER_SMILE/checkpoint_51000.pth'


def is_private(image1, image2, args=None, config=None):

    if image1.shape != image2.shape:
        raise ValueError("The two images must have the same shape.")

    model = create_classifier(image_size=64,
        classifier_use_fp16=False,
        classifier_width=128,
        classifier_depth=2,
        classifier_attention_resolutions="32,16,8",  # 16
        classifier_use_scale_shift_norm=True,  # False
        classifier_resblock_updown=True,  # False
        classifier_pool="attention",).to(config.device)
    
    labels = torch.tensor([0] * image1.shape[0], device='cuda')
    model.load_state_dict(torch.load(args.classifier_state_dict)[0])
    # model.load_state_dict(torch.load(path)[0])
    attr1 = model(image1, labels)
    attr2 = model(image2, labels)
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
    
    image_dir = "/data/local/ml01/qipan/exp_celeba/CG_smile_samples/nearest_image/smile"
    image_paths = glob(os.path.join(image_dir, "*.png"))

    transform=transforms.Compose([
        transforms.Resize((64*2, 64)),
        transforms.ToTensor()
    
    ])
    
    model = create_classifier(image_size=64,
        classifier_use_fp16=False,
        classifier_width=128,
        classifier_depth=2,
        classifier_attention_resolutions="32,16,8",  # 16
        classifier_use_scale_shift_norm=True,  # False
        classifier_resblock_updown=True,  # False
        classifier_pool="attention",).to('cuda')
    
    model.load_state_dict(torch.load(path)[0])
    labels = torch.tensor([0] * 64, device='cuda')

    private_score = 0
    for image_path in image_paths:
        img = transform(Image.open(image_path))
        ground_truth = img[:, :64, :]
        generated = img[:, 64:, :]
        print(image_path)
        print(torch.argmax(model(ground_truth.unsqueeze(0).cuda(), labels)))
        print(torch.argmax(model(generated.unsqueeze(0).cuda(), labels)))
        private_score += is_private(ground_truth.unsqueeze(0).cuda(), generated.unsqueeze(0).cuda())
    print(private_score/len(image_paths))
        