import torch
import torch.nn as nn
import torch.nn.functional as F

class URDMU(nn.Module):
    def __init__(self):
        super(URDMU, self).__init__()
        # Input size for I3D features
        self.feature_dim = 2048
        self.num_class = 2  # Normal vs Anomaly
        
        # Feature encoder
        self.feature_encoder = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU()
        )
        
        # Dual Memory Units
        self.normal_memory = nn.Parameter(torch.randn(128, 10))
        self.abnormal_memory = nn.Parameter(torch.randn(128, 10))
        
        # Uncertainty regulation
        self.uncertainty_encoder = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Classification head
        self.classifier = nn.Linear(128, self.num_class)

    def forward(self, x):
        # x shape: [batch_size, 2048] (I3D features)
        features = self.feature_encoder(x)
        
        # Memory attention
        normal_attention = torch.softmax(torch.matmul(features, self.normal_memory), dim=1)
        abnormal_attention = torch.softmax(torch.matmul(features, self.abnormal_memory), dim=1)
        
        # Memory readout
        normal_mem = torch.matmul(normal_attention, self.normal_memory.t())
        abnormal_mem = torch.matmul(abnormal_attention, self.abnormal_memory.t())
        
        # Combine features
        combined_features = features + normal_mem + abnormal_mem
        
        # Uncertainty estimation
        uncertainty = self.uncertainty_encoder(combined_features)
        
        # Classification
        logits = self.classifier(combined_features)
        
        return logits, uncertainty

# File: main.py modifications

# Add to imports section:
from archs.video_anomaly import urdmu

# Add to model selection section:
elif args.arch_type == "urdmu":
    model = urdmu.URDMU().to(device)

# Add to dataset loading section:
elif args.dataset == "ucf_crime":
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    traindataset = VideoAnomalyDataset(
        root='../data/UCF_Crime/i3d_features',
        split='train',
        transform=transform
    )
    testdataset = VideoAnomalyDataset(
        root='../data/UCF_Crime/i3d_features',
        split='test',
        transform=transform
    )
elif args.dataset == "xd_violence":
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    traindataset = VideoAnomalyDataset(
        root='../data/XD_Violence/i3d_features',
        split='train',
        transform=transform
    )
    testdataset = VideoAnomalyDataset(
        root='../data/XD_Violence/i3d_features',
        split='test',
        transform=transform
    )