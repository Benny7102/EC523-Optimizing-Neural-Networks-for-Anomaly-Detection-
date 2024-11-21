# File: trainer.py
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import utils
import matplotlib.pyplot as plt
import pickle

class LotteryTicketTrainer:
    def __init__(self, args, model, train_loader, test_loader, criterion, device):
        self.args = args
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.device = device
        self.mask = None
        self.initial_state_dict = None
        
    def initialize_training(self):
        """Initialize masks and save initial state"""
        self.make_mask()
        self.initial_state_dict = self.model.state_dict().copy()
        utils.checkdir(f"saves/{self.args.arch_type}/{self.args.dataset}/")
        torch.save(self.model, f"saves/{self.args.arch_type}/{self.args.dataset}/initial_state_dict_{self.args.prune_type}.pth.tar")
        
    def train_epoch(self, optimizer):
        """Train for one epoch"""
        EPS = 1e-6
        self.model.train()
        for batch_idx, (imgs, targets) in enumerate(self.train_loader):
            optimizer.zero_grad()
            imgs, targets = imgs.to(self.device), targets.to(self.device)
            output = self.model(imgs)
            train_loss = self.criterion(output, targets)
            train_loss.backward()

            # Freeze pruned weights
            for name, p in self.model.named_parameters():
                if 'weight' in name:
                    tensor = p.data.cpu().numpy()
                    grad_tensor = p.grad.data.cpu().numpy()
                    grad_tensor = np.where(tensor < EPS, 0, grad_tensor)
                    p.grad.data = torch.from_numpy(grad_tensor).to(self.device)
            optimizer.step()
        return train_loss.item()

    def evaluate(self):
        """Evaluate model accuracy"""
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
        return 100. * correct / total

    def make_mask(self):
        """Initialize masks for pruning"""
        step = 0
        for name, param in self.model.named_parameters():
            if 'weight' in name:
                step += 1
        self.mask = [None] * step
        
        step = 0
        for name, param in self.model.named_parameters():
            if 'weight' in name:
                tensor = param.data.cpu().numpy()
                self.mask[step] = np.ones_like(tensor)
                step += 1

    def prune_by_percentile(self, percent):
        """Prune model by percentile"""
        step = 0
        for name, param in self.model.named_parameters():
            if 'weight' in name:
                tensor = param.data.cpu().numpy()
                alive = tensor[np.nonzero(tensor)]
                percentile_value = np.percentile(abs(alive), percent)
                
                weight_dev = param.device
                new_mask = np.where(abs(tensor) < percentile_value, 0, self.mask[step])
                
                param.data = torch.from_numpy(tensor * new_mask).to(weight_dev)
                self.mask[step] = new_mask
                step += 1

    def train_with_pruning(self):
        """Main training loop with pruning"""
        best_accuracy = 0
        comp = np.zeros(self.args.prune_iterations, float)
        bestacc = np.zeros(self.args.prune_iterations, float)
        
        for _ite in range(self.args.start_iter, self.args.prune_iterations):
            if _ite > 0:
                self.prune_by_percentile(self.args.prune_percent)
                if self.args.prune_type == "reinit":
                    self.reset_weights()
                else:
                    self.restore_initial_weights()
                    
            print(f"\n--- Pruning Level [{_ite}/{self.args.prune_iterations}]: ---")
            
            # Initialize optimizer for this pruning iteration
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
            
            # Training loop for this pruning iteration
            pbar = tqdm(range(self.args.end_iter))
            for iter_ in pbar:
                if iter_ % self.args.valid_freq == 0:
                    accuracy = self.evaluate()
                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        self.save_model(_ite)
                
                loss = self.train_epoch(optimizer)
                pbar.set_description(
                    f'Train Epoch: {iter_}/{self.args.end_iter} Loss: {loss:.6f} Accuracy: {accuracy:.2f}%')
            
            # Save statistics
            comp[_ite] = self.compute_compression()
            bestacc[_ite] = best_accuracy
            best_accuracy = 0
            
        return comp, bestacc

    def compute_compression(self):
        """Compute compression ratio"""
        nonzero = 0
        total = 0
        for name, p in self.model.named_parameters():
            tensor = p.data.cpu().numpy()
            nz_count = np.count_nonzero(tensor)
            total_params = np.prod(tensor.shape)
            nonzero += nz_count
            total += total_params
        return 100 * (nonzero / total)

    def save_model(self, iteration):
        """Save model checkpoint"""
        utils.checkdir(f"saves/{self.args.arch_type}/{self.args.dataset}/")
        torch.save(self.model, 
                  f"saves/{self.args.arch_type}/{self.args.dataset}/{iteration}_model_{self.args.prune_type}.pth.tar")
