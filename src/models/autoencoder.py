import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

#1D Convolutional Autoencoder for time-series / spectrogram windows.
class ConvAutoencoder1D(nn.Module):
    def __init__(self, in_channels: int = 1, bottleneck_dim: int = 32):
        super().__init__()

        # --- ENCODER ---
        # Input: (batch, in_channels, length) e.g., (32, 1, 750)
        self.enc_conv1 = nn.Conv1d(in_channels, 32, kernel_size=7, stride=2, padding=3)  # -> (32, 375)
        self.enc_bn1   = nn.BatchNorm1d(32)

        self.enc_conv2 = nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2)         # -> (64, 188)
        self.enc_bn2   = nn.BatchNorm1d(64)

        self.enc_conv3 = nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1)        # -> (128, 94)
        self.enc_bn3   = nn.BatchNorm1d(128)

        # Bottleneck projection
        self.bottleneck = nn.Linear(128 * 94, bottleneck_dim)
        self.de_bottleneck = nn.Linear(bottleneck_dim, 128 * 94)

        # --- DECODER ---
        self.dec_conv1 = nn.ConvTranspose1d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1) # -> (64, 188)
        self.dec_bn1   = nn.BatchNorm1d(64)

        self.dec_conv2 = nn.ConvTranspose1d(64, 32, kernel_size=5, stride=2, padding=2, output_padding=0)  # -> (32, 375)
        self.dec_bn2   = nn.BatchNorm1d(32)

        self.dec_conv3 = nn.ConvTranspose1d(32, in_channels, kernel_size=7, stride=2, padding=3, output_padding=1) # -> (1, 750)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(1)

        x = F.relu(self.enc_bn1(self.enc_conv1(x)))
        x = F.relu(self.enc_bn2(self.enc_conv2(x)))
        x = F.relu(self.enc_bn3(self.enc_conv3(x)))

        b = x.size(0)
        x = x.reshape(b, -1)
        z = self.bottleneck(x)
        return z

    def decode(self, z: torch.Tensor, target_length: int = 750) -> torch.Tensor:
        b = z.size(0)
        x = self.de_bottleneck(z)
        x = x.reshape(b, 128, 94)

        x = F.relu(self.dec_bn1(self.dec_conv1(x)))
        x = F.relu(self.dec_bn2(self.dec_conv2(x)))
        x = self.dec_conv3(x)

        # Ensure exact target length
        if x.size(-1) != target_length:
            x = F.interpolate(x, size=target_length, mode='linear', align_corners=False)
        return x

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        target_len = x.size(-1) if x.ndim == 3 else x.shape[-1]
        z = self.encode(x)
        reconstructed = self.decode(z, target_length=target_len)
        return reconstructed, z


#Compute MSE reconstruction error for each window.
def compute_reconstruction_errors(model: nn.Module, data_windows: np.ndarray, device: str = "cpu") -> np.ndarray:

    model.eval()
    errors = []
    batch_size = 64

    with torch.no_grad():
        for i in range(0, len(data_windows), batch_size):
            batch = torch.from_numpy(data_windows[i:i+batch_size]).float().to(device)
            recon, _ = model(batch)
            # MSE per window
            mse = torch.mean((batch.unsqueeze(1) - recon) ** 2, dim=[1, 2])
            errors.extend(mse.cpu().numpy())

    return np.array(errors)


if __name__ == "__main__":
    model = ConvAutoencoder1D()
    x = torch.randn(16, 750)
    recon, z = model(x)
    print("Autoencoder test passed!")
    print(f"  Input: {x.shape}")
    print(f"  Bottleneck latent z: {z.shape}")
    print(f"  Reconstructed: {recon.shape}")
