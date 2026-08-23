import torch.nn as nn
import torchvision.models as models


def get_model(architecture: str, num_classes: int = 10) -> nn.Module:
    architecture = architecture.lower()
    if architecture != "resnet18":
        raise ValueError(f"Unsupported architecture: {architecture}")

    model = models.resnet18(weights=None, num_classes=num_classes)
    # ResNet-18's default stem (7x7 stride-2 conv + maxpool) is built for
    # 224x224 ImageNet input; on CIFAR-10's 32x32 images it throws away too
    # much spatial resolution before the residual blocks even start. Swap
    # in the standard CIFAR stem instead: a 3x3 stride-1 conv, no maxpool.
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model
