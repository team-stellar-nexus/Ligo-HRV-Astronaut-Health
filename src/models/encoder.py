import torch
import torch.nn as nn
import torch.nn.functional as F

# 1D Residual Block with Conv1D, BatchNorm, and ReLU.
class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 5, stride: int = 1):
        super().__init__()
        padding = kernel_size // 2

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm1d(out_channels)

        # Shortcut to match dimensions 
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return self.relu(out)


# 1D-CNN Encoder for raw time-series sensor windows (ECG, PPG, BVP).
class BiosignalEncoder(nn.Module):
    def __init__(self, in_channels: int = 1, embedding_dim: int = 128):
        super().__init__()

        # Initial feature extractor
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )

        # Residual blocks with increasing channels
        self.layer1 = ResidualBlock1D(32, 64, kernel_size=7, stride=2)
        self.layer2 = ResidualBlock1D(64, 128, kernel_size=5, stride=2)
        self.layer3 = ResidualBlock1D(128, embedding_dim, kernel_size=3, stride=2)

        # Global pooling
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(1)

        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.pool(x)       
        x = x.squeeze(-1)      
        return x


# Complete model: Encoder + Linear Classification Head for Stress Detection.
class HealthClassifier(nn.Module):
    def __init__(self, embedding_dim: int = 128, num_classes: int = 3):
        super().__init__()
        self.encoder = BiosignalEncoder(in_channels=1, embedding_dim=embedding_dim)

        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(embedding_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embeddings = self.encoder(x)
        logits = self.classifier(embeddings)
        return logits, embeddings


if __name__ == "__main__":
    # Testing model 
    model = HealthClassifier(num_classes=3)
    dummy_input = torch.randn(16, 750)  
    logits, embeddings = model(dummy_input)

    print("Model test successful!")
    print(f"  Input shape: {dummy_input.shape}")
    print(f"  Logits shape: {logits.shape}")
    print(f"  Embeddings shape: {embeddings.shape}")
