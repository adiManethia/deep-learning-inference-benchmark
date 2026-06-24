import torch 
import torch.nn as nn 
import torchvision.models as models

class SmallCNN(nn.Module):
    def __init__(self):
        super(SmallCNN, self).__init__()
        # feature extractor: takes 3 color channels, outputs 16
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Classifier flatten the image and maps to 10 output classes
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 112 * 112, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def build_model(model_name: str):
    """
    Model registry to easily swap architectures.
    """
    if model_name == "SmallCNN":
        return SmallCNN()
    elif model_name == "ResNet18":
        return models.resnet18(weights=None)
    elif model_name == "MobileNetV2":
        return models.mobilenet_v2(weights=None)
    else:
        raise ValueError(f"Model {model_name} not supported yet!")

def count_parameters(model):
    """
    Returns the total number of trainable parameters in millions.
    """
    total_params = sum(p.numel() for p in model.parameters())
    return total_params / 1_000_000 # Convert to Millions