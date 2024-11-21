import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from typing import Tuple, Dict

class DatasetFactory:
    @staticmethod
    def get_dataset(name: str, root: str = '../data') -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        dataset_map = {
            'mnist': (datasets.MNIST, 1),
            'cifar10': (datasets.CIFAR10, 3),
            'fashionmnist': (datasets.FashionMNIST, 1),
            'cifar100': (datasets.CIFAR100, 3)
        }
        
        if name not in dataset_map:
            raise ValueError(f"Dataset {name} not supported. Available datasets: {list(dataset_map.keys())}")
            
        dataset_class, channels = dataset_map[name]
        
        train_dataset = dataset_class(root, train=True, download=True, transform=transform)
        test_dataset = dataset_class(root, train=False, transform=transform)
        
        return train_dataset, test_dataset

    @staticmethod
    def get_data_loaders(train_dataset, test_dataset, batch_size: int, num_workers: int = 0):
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, 
            num_workers=num_workers, drop_last=False
        )
        
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, drop_last=True
        )
        
        return train_loader, test_loader