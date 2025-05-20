import torch
import torch.nn as nn
import torchvision.models as models

class CNNBranch(nn.Module):
    def __init__(self, feature_dim=64):
        super(CNNBranch, self).__init__()
        
        # Load pretrained ResNet18 backbone
        self.backbone = models.resnet18(pretrained=True)
        
        # Remove the final fully connected layer
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])  # Remove last FC
        
        # Add a new custom head
        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
            nn.Linear(256, feature_dim),  # Output 32-dimensional feature vector
            nn.BatchNorm1d(feature_dim),
            nn.LeakyReLU(0.1)
        )

    def forward(self, x):
        x = self.backbone(x)  # Pass through ResNet18 convolution layers
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)  # Project to 32-dimensional features
        return x
