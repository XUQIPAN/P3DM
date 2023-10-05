import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
from models.classifier_models import *


def check_cuda():
    _cuda = False
    if torch.cuda.is_available():
        _cuda = True
    return _cuda


def grad_classifier(scale:int, x:torch.tensor, y:torch.tensor, attribute:str)->float:
    # compute the gradient of classifier
    train_transform = transforms.Resize((128,128))
    x = train_transform(x)
    target = torch.eye(2)[y].squeeze().cuda()

    is_cuda = check_cuda()
    if attribute == 'gender' or attribute == 'attractive':
        model = MultiClassifier()
        if attribute == 'gender':
            model.load_state_dict(torch.load('/data/local/qipan/exp_celeba/celeba_cls_gender.pth'))
        else:
            model.load_state_dict(torch.load('/workspace/celeba_cls_attractive.pth'))
    elif attribute == 'smile':
        model = CustomResNet18Model(num_classes=2)
        model.load_state_dict(torch.load('/workspace/celeba_cls_smile.pth'))
    if is_cuda:
        model.cuda()

    criterion = nn.BCELoss()
    pred = model(x)
    loss = criterion(pred, target)

    gradient = torch.autograd.grad(outputs=loss, inputs=x)
    grad_transform = transforms.Resize((64,64))
    gradient = grad_transform(gradient[0])
    # gradient shape: tuple(tensor)
    # print(gradient.shape)
    variance = torch.var(x, dim=(1, 2, 3)).cuda()
    expanded_variance = variance.unsqueeze(1).unsqueeze(1).unsqueeze(1)
    return scale * expanded_variance * gradient


if __name__ == "__main__":
    device = torch.device('cuda')
    init_samples = torch.rand(36, 3, 64, 64, requires_grad=True)
    print(init_samples.shape)
    print(init_samples.view(-1).detach().numpy().shape[0])
    label = torch.randint(0, 2, (init_samples.shape[0],))
    print(label.shape)
    grad = grad_classifier(1, init_samples.cuda(), label, 'smile')
    print(grad.shape)