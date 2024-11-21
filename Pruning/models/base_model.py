from abc import ABC, abstractmethod
import torch.nn as nn

class BaseModel(nn.Module, ABC):
    @abstractmethod
    def __init__(self):
        super().__init__()
        
    @abstractmethod
    def forward(self, x):
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        pass