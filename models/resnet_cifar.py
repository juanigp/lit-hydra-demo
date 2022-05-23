from torchvision.models import resnet18
import torch

def resnet18_cifar10():
    model = resnet18(num_classes=10, pretrained=False)
    model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    model.maxpool = torch.nn.Identity()
    return model