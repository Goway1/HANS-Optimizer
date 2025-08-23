from .cnn import CIFAR10CNN
from .resnet import ResNet18

def get_model(architecture: str, num_classes: int = 10):
    if architecture == 'resnet':
        return ResNet18()
    return CIFAR10CNN(num_classes=num_classes)
