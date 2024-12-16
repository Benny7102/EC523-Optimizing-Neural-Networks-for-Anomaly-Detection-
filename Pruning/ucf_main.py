import pdb
import numpy as np
import torch.utils.data as data
from torch.utils.data import Subset
import utils
from options import *
from config import *
from train import *
from ucf_test import test
from model import *
from utils import Visualizer
import os
from dataset_loader import *
from tqdm import tqdm
import copy
from prune import ModelPruner, count_parameters

USE_VISDOM = False

if __name__ == "__main__":
    args = parse_args()
    if args.debug:
        pdb.set_trace()

    config = Config(args)
    worker_init_fn = None
    gpus = [0]
    if torch.cuda.is_available():
        device = 'cuda:0'
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = 'cpu'
        print("Using CPU: No GPU available")

    # Set the device
    torch.cuda.set_device(device if device.startswith('cuda') else 0)

    config.len_feature = 1024
    net = WSAD(config.len_feature, flag = "Train", a_nums = 60, n_nums = 60)
    net = net.cuda()

    pruner = ModelPruner(net, total_prune_rounds=5, target_sparsity=0.8)
    pruner.add_gradient_mask()
    
    total, zeros = count_parameters(net)
    print(f"\nBefore pruning:")
    print(f"Total parameters: {total:,}")
    print(f"Zero parameters: {zeros:,}")
    print(f"Sparsity: {100.0 * zeros / total:.2f}%\n")

    normal_train_loader = data.DataLoader(
        UCF_crime(root_dir = config.root_dir, mode = 'Train', modal = config.modal, num_segments = 200, len_feature = config.len_feature, is_normal = True),
            batch_size = 64,
            shuffle = True, num_workers = config.num_workers,
            worker_init_fn = worker_init_fn, drop_last = True)
    abnormal_train_loader = data.DataLoader(
        UCF_crime(root_dir = config.root_dir, mode = 'Train', modal = config.modal, num_segments = 200, len_feature = config.len_feature, is_normal = False),
            batch_size = 64,
            shuffle = True, num_workers = config.num_workers,
            worker_init_fn = worker_init_fn, drop_last = True)
    test_loader = data.DataLoader(
        UCF_crime(root_dir = config.root_dir, mode = 'Test', modal = config.modal, num_segments = config.num_segments, len_feature = config.len_feature),
            batch_size = 1,
            shuffle = False, num_workers = config.num_workers,
            worker_init_fn = worker_init_fn)

    test_info = {"step": [], "auc": [],"ap":[],"ac":[]}
    
    best_auc = 0

    best_state_dict = None

    criterion = AD_Loss()
    
    optimizer = torch.optim.Adam(net.parameters(), lr = config.lr[0],
        betas = (0.9, 0.999), weight_decay = 0.00005)

    test(net, config, test_loader, test_info, 0)
    for prune_round in range(pruner.total_rounds):
        print(f"\nStarting pruning round {prune_round + 1}/{pruner.total_rounds}")
        
        # Reset weights to initial values while maintaining masks
        if prune_round > 0:
            pruner.reset_weights()
        
        # Prune the network
        pruner.prune_model(prune_round)
        
        # Validate after pruning
        test(net, config, test_loader, test_info, f"round_{prune_round}_start")
        
        # Training iterations for this pruning round
        for step in tqdm(range(1, config.num_iters + 1), total = config.num_iters, dynamic_ncols = True):
            try:
                if step > 1 and config.lr[step - 1] != config.lr[step - 2]:
                    for param_group in optimizer.param_groups:
                        param_group["lr"] = config.lr[step - 1]
                if (step - 1) % len(normal_train_loader) == 0:
                    normal_loader_iter = iter(normal_train_loader)

                if (step - 1) % len(abnormal_train_loader) == 0:
                    abnormal_loader_iter = iter(abnormal_train_loader)
                train(net, normal_loader_iter, abnormal_loader_iter, optimizer, criterion, step)
                
                # Validate every 10 steps
                if step % 10 == 0 and step > 10:
                    test(net, config, test_loader, test_info, f"round_{prune_round}_step_{step}")
                    if test_info["auc"][-1] > best_auc:
                        best_auc = test_info["auc"][-1]
                        utils.save_best_record(test_info, 
                            os.path.join(config.output_path, "ucf_best_record_{}.txt".format(config.seed)))

                        torch.save(net.state_dict(), os.path.join(args.model_path, \
                            "pruned_ucf_trans_{}.pkl".format(config.seed)))
            except Exception as e:
                print(f"Error during training: {str(e)}")
                continue
        
        # Save model after each pruning round
        torch.save(net.state_dict(), 
                  os.path.join(config.model_path, f"model_round_{prune_round}.pkl"))
        
        # Print pruning statistics
        total, zeros = count_parameters(net)
        print(f"\nAfter pruning round {prune_round + 1}:")
        print(f"Total parameters: {total:,}")
        print(f"Zero parameters: {zeros:,}")
        print(f"Sparsity: {100.0 * zeros / total:.2f}%\n")

    # Save final model
    save_dict = {
        'original_state_dict': pruner.initial_state_dict,
        'final_state_dict': best_state_dict,
        'mask': {name + '.weight_mask': module.weight_mask 
                for name, module in net.named_modules() 
                if hasattr(module, 'weight_mask')}
    }
    torch.save(save_dict, os.path.join(args.model_path, f"iterative_pruned_ucf_trans_{config.seed}.pkl"))