import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import numpy as np


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
    

def check_cuda():
    _cuda = False
    if torch.cuda.is_available():
        _cuda = True
    return _cuda


def grad_classifier(scale:int, x:torch.tensor, y:torch.tensor)->float:
    # compute the gradient of classifier
    train_transform = transforms.Compose([
        transforms.Resize((128,128)),
        transforms.ToTensor()
    ])
    # x = train_transform(x).cuda()
    target = torch.eye(2)[y].squeeze().cuda()

    is_cuda = check_cuda()
    model = MultiClassifier()
    model.load_state_dict(torch.load('/common/home/qx67/Desktop/dpgen/DPgan_model/celeba_cls.pth'))
    if is_cuda:
        model.cuda()

    criterion = nn.BCELoss()
    pred = model(x)
    loss = criterion(pred, target)

    model.zero_grad()
    loss.backward()
    total_gradients = 0.0
    param_num = 0
    for param in model.parameters():
        if param.grad is not None:
            total_gradients += param.grad.sum().item()
            param_num += 1

    return scale * total_gradients/param_num/x.view(-1).cpu().detach().numpy().shape[0]


if __name__ == "__main__":
    device = torch.device('cuda')
    init_samples = torch.rand(36, 3, 128, 128)
    print(init_samples.shape)
    print(init_samples.view(-1).numpy().shape[0])
    label = torch.randint(0, 2, (init_samples.shape[0],))
    print(label.shape)
    grad = grad_classifier(1, init_samples.cuda(), label)
    print(grad)