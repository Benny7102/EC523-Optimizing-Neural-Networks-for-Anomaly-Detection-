import torch
import os
from model import WSAD

class Args:
    def __init__(self):
        self.model_path = "/projectnb/ec523kb/projects/teams_Fall_2024/Team_9/Pruning_UR_DMU/Fast-UR-DMU/models"
        self.output_path = "/projectnb/ec523kb/projects/teams_Fall_2024/Team_9/Pruning_UR_DMU/Fast-UR-DMU/models"
        self.root_dir = "/projectnb/ec523kb/projects/teams_Fall_2024/Team_9/Pruning_UR_DMU/Fast-UR-DMU"
        self.modal = "rgb"

def apply_masks_to_model(model):
    for name, module in model.named_modules():
        if hasattr(module, 'weight_mask'):
            # Apply mask to the weights
            module.weight.data *= module.weight_mask
            print(f"Applied mask to {name}")

def count_parameters(model):
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    zeros = sum((p == 0).sum().item() for p in model.parameters() if p.requires_grad)
    return total, zeros

def convert_state_dict(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        # Handle _orig suffix
        if key.endswith('_orig'):
            new_key = key.replace('_orig', '')
            new_state_dict[new_key] = value
        # Handle special memory cases
        elif key in ['Amemory.weight', 'Nmemory.weight']:
            new_key = key.replace('weight', 'memory_block')
            new_state_dict[new_key] = value
        # Keep other keys as they are
        elif not key.endswith('_mask'):
            new_state_dict[key] = value
    return new_state_dict

def main():
    args = Args()
    
    # Initialize model
    len_feature = 1024
    net = WSAD(len_feature, flag="Train", a_nums=60, n_nums=60)
    
    if torch.cuda.is_available():
        net = net.cuda()
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU")

    # Load the saved model
    model_path = os.path.join(args.model_path, f"target_90/model_round_4.pkl")
    print(f"Loading model from: {model_path}")
    saved_model = torch.load(model_path)
    
    # Print model keys for debugging
    if 'state_dict' in saved_model:
        print("\nKeys in state_dict:", saved_model['state_dict'].keys())
    else:
        print("\nKeys in model:", saved_model.keys())

    # Print initial statistics
    total, zeros = count_parameters(net)
    print("\nBefore applying masks:")
    print(f"Total parameters: {total:,}")
    print(f"Zero parameters: {zeros:,}")
    print(f"Sparsity: {100.0 * zeros / total:.2f}%\n")

    # Convert and load the state dict
    if 'state_dict' in saved_model:
        converted_state_dict = convert_state_dict(saved_model['state_dict'])
    else:
        converted_state_dict = convert_state_dict(saved_model)
    
    # Load state dict with strict=False to ignore missing keys
    net.load_state_dict(converted_state_dict, strict=False)

    # Create masks from the saved weights
    masks = {}
    for key, value in saved_model['state_dict'].items() if 'state_dict' in saved_model else saved_model.items():
        if key.endswith('_mask'):
            masks[key] = value

    # Apply masks to the model
    for name, module in net.named_modules():
        if hasattr(module, 'weight'):
            mask_name = name + '.weight_mask'
            if mask_name in masks:
                module.weight_mask = masks[mask_name]
                module.weight.data *= module.weight_mask

    # Print final statistics
    total, zeros = count_parameters(net)
    print("\nAfter applying masks:")
    print(f"Total parameters: {total:,}")
    print(f"Zero parameters: {zeros:,}")
    print(f"Sparsity: {100.0 * zeros / total:.2f}%\n")

    # Save the properly pruned model
    print(args.output_path)
    output_path = os.path.join(args.output_path, f"target_90_model_round_4_masked.pkl")
    print(f"Saving masked model to: {output_path}")
    
    torch.save({
        'state_dict': net.state_dict(),
        'masks': masks
    }, output_path)

if __name__ == "__main__":
    main()
