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


def grad_classifier(scale:int, x:torch.tensor, y:torch.tensor, t, model)->float:
    # compute the gradient of classifier
    target = y.cuda()
    bsz = y.shape[0]
    t = torch.tensor([t]*bsz, device=y.device)

    criterion = nn.CrossEntropyLoss()
    pred = model(x, timesteps=t)
    loss = criterion(pred, target)

    gradient = torch.autograd.grad(outputs=loss, inputs=x)[0]

    out = scale * gradient
    out2 = out.detach()

    del gradient, out
    return out2


if __name__ == "__main__":
    device = torch.device('cuda')
    init_samples = torch.rand(36, 3, 64, 64, requires_grad=True)
    print(init_samples.shape)
    print(init_samples.view(-1).detach().numpy().shape[0])
    label = torch.randint(0, 2, (init_samples.shape[0],))
    print(label.shape)
    grad = grad_classifier(1, init_samples.cuda(), label, 'smile')
    print(grad.shape)