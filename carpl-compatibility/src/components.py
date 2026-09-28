import torch
import torch.nn as nn
from torchxrayvision.models import ResNet

class ClassifierHead(nn.Module):
    """Two-layer MLP classification head with ReLU and dropout.

    Args:
        input_dim (int): Size of the input feature vector.
        num_classes (int): Number of output classes.
        hidden_dim (int): Hidden layer width.
        dropout (float): Dropout probability between layers.

    Attributes:
        head (nn.Sequential): Sequential MLP that maps inputs to logits.
    """
    def __init__(self, input_dim, num_classes, hidden_dim, dropout):
        super().__init__()
        self.head = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                  nn.ReLU(),
                                  nn.Dropout(dropout),
                                  nn.Linear(hidden_dim, num_classes))
        
    def forward(self, x):
        """Compute class probability from input features.

        Args:
            x (torch.Tensor): Input tensor of shape ``(batch_size, input_dim)``.

        Returns:
            torch.Tensor: Class probabilities of shape ``(batch_size, num_classes)``.
        """
        logits = self.head(x)
        return logits


class ResNetClassifier(nn.Module):
    """ResNet50 feature extractor with tabular fusion for classification."""
    def __init__(self, tabular_dim, tabular_proj, num_classes, hidden_dim, dropout, input_size=512):
        super().__init__()
        resnet = ResNet(weights="resnet50-res512-all", cache_dir="models/xrv_weights/")
        self.encoder = nn.Sequential(
            resnet.model.conv1,
            resnet.model.bn1,
            resnet.model.relu,
            resnet.model.maxpool,
            resnet.model.layer1,
            resnet.model.layer2,
            resnet.model.layer3,
            resnet.model.layer4,
        ) # remove the 18 feature classifier head
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.tabular_proj = nn.Linear(tabular_dim, tabular_proj)

        with torch.no_grad():
            dummy = torch.zeros(1, 1, input_size, input_size)  # 1 channel for X-ray
            out = self.encoder(dummy)
            out = self.global_pool(out)
            flat_dim = out.numel()  # total elements
            print(f"CNN Extracted Feature Dim: {flat_dim}")

        self.head = ClassifierHead(flat_dim + tabular_proj, num_classes, hidden_dim, dropout)
    
    def forward(self, x, demos):
        """Forward pass combining CNN features and tabular features."""
        x = self.encoder(x)
        x = self.global_pool(x)
        x = self.flatten(x)
        demos = self.tabular_proj(demos)
        combined = torch.cat([x, demos], dim=1)
        out = self.head(combined)
        return out