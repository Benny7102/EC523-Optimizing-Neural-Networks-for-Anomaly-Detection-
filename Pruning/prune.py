import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import copy
from memory import Memory_Unit


def count_parameters(model):
    total_params = 0
    zero_params = 0
    
    # Track each prunable layer
    for name, module in model.named_modules():
        if isinstance(module, Memory_Unit):
            if hasattr(module, 'memory_block'):
                param = module.memory_block
                mask = getattr(module, 'memory_block_mask', None)
                if param is not None:
                    this_total = param.numel()
                    total_params += this_total
                    if mask is not None:
                        zero_params += torch.sum(mask == 0).item()
                        print(f"Memory layer {name}: {torch.sum(mask == 0).item()}/{this_total}")
                        
        elif isinstance(module, (nn.Linear, nn.Conv1d)):
            if hasattr(module, 'weight'):
                param = module.weight
                mask = getattr(module, 'weight_mask', None)
                if param is not None:
                    this_total = param.numel()
                    total_params += this_total
                    if mask is not None:
                        zero_params += torch.sum(mask == 0).item()
                        print(f"Layer {name}: {torch.sum(mask == 0).item()}/{this_total}")

    sparsity = 100 * zero_params / total_params if total_params > 0 else 0
    
    return total_params, zero_params

class ModelPruner:
    def __init__(self, model, total_prune_rounds=5, target_sparsity=0.8):
        self.model = model
        self.total_rounds = total_prune_rounds
        self.target_sparsity = target_sparsity
        self.sparsity_per_round = 1 - (1 - target_sparsity) ** (1/total_prune_rounds)
        self.initial_state_dict = copy.deepcopy(model.state_dict())
        
        # Initialize pruning for all layers
        self.current_masks = {}
        for name, module in self.model.named_modules():
            if isinstance(module, Memory_Unit):
                prune.random_unstructured(module, name='memory_block', amount=0)
                self.current_masks[name + '.memory_block'] = module.memory_block_mask
            elif isinstance(module, (nn.Linear, nn.Conv1d)):
                prune.random_unstructured(module, name='weight', amount=0)
                self.current_masks[name + '.weight'] = module.weight_mask

    def prune_model(self, round_num):
        before_params, before_zeros = count_parameters(self.model)
        current_target_sparsity = 1 - (1 - self.sparsity_per_round) ** (round_num + 1)
        
        print(f"\nStarting pruning with target sparsity: {current_target_sparsity:.4f}")
        
        prunable_layers = []
        for name, module in self.model.named_modules():
            if isinstance(module, Memory_Unit):
                prunable_layers.append((name, module, 'memory_block'))
                print(f"Found Memory_Unit layer: {name}")
            elif isinstance(module, (nn.Linear, nn.Conv1d)):
                prunable_layers.append((name, module, 'weight'))
                print(f"Found prunable layer: {name}, type: {type(module).__name__}")
                
        print(f"\nTotal prunable layers found: {len(prunable_layers)}")
        
        for name, module, param_type in prunable_layers:
            param_name = f"{name}.{param_type}"
            print(f"\nProcessing layer: {param_name}")
            
            try:
                # Get the parameter and its current mask
                if param_type == 'memory_block':
                    weights = module.memory_block.data
                    if not hasattr(module, 'memory_block_mask'):
                        module.memory_block_mask = nn.Parameter(torch.ones_like(weights), requires_grad=False)
                    current_mask = module.memory_block_mask
                else:
                    weights = module.weight.data
                    if not hasattr(module, 'weight_mask'):
                        module.weight_mask = nn.Parameter(torch.ones_like(weights), requires_grad=False)
                    current_mask = module.weight_mask

                # Calculate absolute values of weights
                weights_abs = torch.abs(weights * current_mask)
                
                # Calculate number of weights to prune this round
                total_weights = weights.numel()
                current_zeros = (current_mask == 0).sum().item()
                target_zeros = int(total_weights * current_target_sparsity)
                need_to_prune = target_zeros - current_zeros

                if need_to_prune > 0:
                    # Find threshold value that gives desired sparsity
                    # Only sort non-zero weights
                    non_zero_weights = weights_abs[weights_abs > 0]
                    if len(non_zero_weights) > 0:
                        sorted_weights = torch.sort(non_zero_weights.view(-1))[0]
                        threshold_idx = min(need_to_prune, len(sorted_weights) - 1)
                        threshold = sorted_weights[threshold_idx]
                    else:
                        threshold = 0.0
                    
                    # Create new mask based on threshold 
                    new_mask = (weights_abs > threshold).float()
                    
                    # Store and apply new mask and zero out pruned weights
                    if param_type == 'memory_block':
                        module.memory_block_mask.data = new_mask
                        module.memory_block.data.mul_(new_mask)
                    else:
                        module.weight_mask.data = new_mask
                        module.weight.data.mul_(new_mask)
                    
                    self.current_masks[param_name] = new_mask.clone()

                # Print statistics
                zeros_after = (new_mask == 0).sum().item() if 'new_mask' in locals() else current_zeros
                print(f"  Total weights: {total_weights}")
                print(f"  Current zeros: {current_zeros}")
                print(f"  Target zeros: {target_zeros}")
                print(f"  Need to prune: {need_to_prune}")
                print(f"  Threshold value: {threshold if 'threshold' in locals() else 0.0:.6f}")
                print(f"  Zeros after new mask: {zeros_after}")

            except Exception as e:
                print(f"Error processing layer {param_name}: {str(e)}")
                continue

    def reset_weights(self):
        with torch.no_grad():
            for name, module in self.model.named_modules():
                if isinstance(module, Memory_Unit):
                    param_name = f"{name}.memory_block"
                    if param_name in self.current_masks:
                        module.memory_block.data = (
                            self.initial_state_dict[param_name] * 
                            self.current_masks[param_name]
                        )
                elif isinstance(module, (nn.Linear, nn.Conv1d)):
                    param_name = f"{name}.weight"
                    if param_name in self.current_masks:
                        module.weight.data = (
                            self.initial_state_dict[param_name] * 
                            self.current_masks[param_name]
                        )

    def add_gradient_mask(self):
        """Add gradient masking to prevent pruned weights from being updated"""
        def make_hook(mask):
            def hook(grad):
                return grad * mask
            return hook

        for name, module in self.model.named_modules():
            if isinstance(module, Memory_Unit):
                if hasattr(module, 'memory_block_mask'):
                    module.memory_block.register_hook(
                        make_hook(module.memory_block_mask)
                    )
            elif isinstance(module, (nn.Linear, nn.Conv1d)):
                if hasattr(module, 'weight_mask'):
                    module.weight.register_hook(
                        make_hook(module.weight_mask)
                    )

    def make_permanent(self):
        for name, module in self.model.named_modules():
            if isinstance(module, Memory_Unit):
                if hasattr(module, 'memory_block_mask'):
                    prune.remove(module, 'memory_block')
            elif isinstance(module, (nn.Linear, nn.Conv1d)):
                if hasattr(module, 'weight_mask'):
                    prune.remove(module, 'weight')