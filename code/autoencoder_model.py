import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Set random seed for reproducibility
torch.manual_seed(42)

# Define the Autoencoder Architecture
class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super(Autoencoder, self).__init__()
        
        # Encoder Layers
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),  # Dense layer 1 (encoder)
            nn.ReLU(),
            nn.Linear(128, 64),       # Dense layer 2 (encoder)
            nn.ReLU(),
            nn.Linear(64, 32),        # Bottleneck layer (compressed representation)
            nn.ReLU()
        )
        
        # Decoder Layers
        self.decoder = nn.Sequential(
            nn.Linear(32, 64),       # Dense layer 1 (decoder)
            nn.ReLU(),
            nn.Linear(64, 128),      # Dense layer 2 (decoder)
            nn.ReLU(),
            nn.Linear(128, input_dim) # Output layer (reconstructed input)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded











