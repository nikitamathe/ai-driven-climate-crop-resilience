import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import os

# =========================================================
# 1. Device
# =========================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# =========================================================
# 2. Load data
# =========================================================
df = pd.read_csv("crop_resilience_full_dataset.csv")

LAT = "Latitude"
LON = "Longitude"
RAIN = "Rainfall_mm"

# =========================================================
# 3. Convert points → grid
# =========================================================
lats = np.sort(df[LAT].unique())
lons = np.sort(df[LON].unique())

lat_i = {v: i for i, v in enumerate(lats)}
lon_i = {v: i for i, v in enumerate(lons)}

grid = np.zeros((len(lats), len(lons)), dtype=np.float32)

for _, r in df.iterrows():
    grid[lat_i[r[LAT]], lon_i[r[LON]]] = r[RAIN]

# Normalize
mean, std = grid.mean(), grid.std()
grid = (grid - mean) / (std + 1e-6)

x = torch.tensor(grid).unsqueeze(0).unsqueeze(0).to(device)

# =========================================================
# 4. CNN Model
# =========================================================
class DownscaleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, 3, padding=1)
        )

    def forward(self, x):
        return self.net(x)

model = DownscaleCNN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

# =========================================================
# 5. Training
# =========================================================
epochs = 6

for epoch in range(epochs):