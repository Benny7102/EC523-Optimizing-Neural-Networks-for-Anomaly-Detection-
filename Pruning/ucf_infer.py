import torch
import numpy as np
from dataset_loader import UCF_crime
from options import parse_args
import pdb
from config import Config
import utils
import os
from model import WSAD
from dataset_loader import data
from sklearn.metrics import precision_recall_curve, roc_curve, auc
import time
from torch.utils.benchmark import Timer

def load_pruned_model(net, model_file):
    # Load the pruned model state_dict
    pruned_state_dict = torch.load(model_file, map_location='cuda:0')

    # Rename the keys in the pruned state dict to match the original model's keys
    renamed_state_dict = {}
    for key, value in pruned_state_dict.items():
        if key == "Amemory.weight_orig":
            renamed_key = "Amemory.weight_orig"
        elif key == "Nmemory.weight_orig":
            renamed_key = "Nmemory.weight_orig"
        elif key.endswith(".weight_orig"):
            renamed_key = key.replace(".weight_orig", ".weight")
        elif key.endswith(".weight_mask"):
            continue  # Skip mask weights if not needed
        elif key.endswith(".memory_block_orig"):
            renamed_key = key.replace(".memory_block_orig", ".memory_block")
        elif key.endswith(".memory_block_mask"):
            renamed_key = key.replace(".memory_block_mask", ".weight_mask")
        else:
            renamed_key = key
        
        renamed_state_dict[renamed_key] = value

    # Apply the mask to the weights (pruning)
    for key in renamed_state_dict:
        if '.weight' in key:
            weight = renamed_state_dict[key]
            mask_key = key.replace('.weight', '.weight_mask')
            
            if mask_key in pruned_state_dict:  # Ensure the mask exists
                mask = pruned_state_dict[mask_key]
                # Apply the mask to the weights: zero out the weights where the mask is 0
                renamed_state_dict[key] = weight * mask
                print(key, ((renamed_state_dict[key] == 0).sum().item() / renamed_state_dict[key].numel()) * 100)

    # Load the renamed and pruned state_dict into the model
    net.load_state_dict(renamed_state_dict)
    print("Done loading..")

def valid(net, config, test_loader, model_file=None):
    with torch.no_grad():
        net.eval()
        net.flag = "Test"
        if model_file is not None:
            load_pruned_model(net, model_file)

        # Warmup
        dummy_input = torch.randn(1, *next(iter(test_loader))[0].shape[1:]).cuda()
        for _ in range(10):
            _ = net(dummy_input)
        torch.cuda.synchronize()

        load_iter = iter(test_loader)
        frame_gt = np.load("frame_label/gt-ucf.npy")
        frame_predict = None
        ucf_pdict = {"Abuse":{}, "Arrest":{}, "Arson":{}, "Assault":{}, "Burglary":{},
                     "Explosion":{}, "Fighting":{}, "RoadAccidents":{}, "Robbery":{},
                     "Shooting":{}, "Shoplifting":{}, "Stealing":{}, "Vandalism":{}, "Normal":{}}
        ucf_gdict = {"Abuse":{}, "Arrest":{}, "Arson":{}, "Assault":{}, "Burglary":{},
                     "Explosion":{}, "Fighting":{}, "RoadAccidents":{}, "Robbery":{},
                     "Shooting":{}, "Shoplifting":{}, "Stealing":{}, "Vandalism":{}, "Normal":{}}
        cls_label = []
        cls_pre = []
        temp_predict = torch.zeros((0)).cuda()
        count = 0

        # Added for counting total inference time
        total_inference_time = 0
        inference_count = 0

        for i in range(len(test_loader.dataset)):
            _data, _label, _name = next(load_iter)
            _name = _name[0]
            _data = _data.cuda()
            _label = _label.cuda()

            torch.cuda.synchronize()
            timer = Timer(
                stmt='net(_data)',
                globals={'net': net, '_data': _data}
            )
            inference_time = timer.timeit(1).mean
            
            res = net(_data)  # Actually run the inference
            
            total_inference_time += inference_time
            inference_count += 1

            a_predict = res["frame"]
            temp_predict = torch.cat([temp_predict, a_predict], dim=0)
            if (i + 1) % 10 == 0:
                cls_label.append(int(_label))
                a_predict = temp_predict.mean(0).cpu().numpy()
                pl = len(a_predict) * 16
            
                if "Normal" in _name:
                    ucf_pdict["Normal"][_name] = np.repeat(a_predict, 16)
                    ucf_gdict["Normal"][_name] = frame_gt[count:count + pl]
                else:
                    ucf_pdict[_name[:-3]][_name] = np.repeat(a_predict, 16)
                    ucf_gdict[_name[:-3]][_name] = frame_gt[count:count + pl]
                count = count + pl
                cls_pre.append(1 if a_predict.max() > 0.5 else 0)
                fpre_ = np.repeat(a_predict, 16)
                if frame_predict is None:         
                    frame_predict = fpre_
                else:
                    frame_predict = np.concatenate([frame_predict, fpre_])  
                temp_predict = torch.zeros((0)).cuda()

        # 4
        average_inference_time = total_inference_time / inference_count
        print(f"Average inference time (ms): {average_inference_time * 1000:.2f}")
        
        frame_gt = np.load("frame_label/gt-ucf.npy")
        np.save('frame_label/ucf_pre.npy', frame_predict)
        np.save('frame_label/ucf_pre_dict.npy', ucf_pdict)
        np.save('frame_label/ucf_gt_dict.npy', ucf_gdict)

        fpr, tpr, _ = roc_curve(frame_gt, frame_predict)
        auc_score = auc(fpr, tpr)
        print(auc_score)
        precision, recall, th = precision_recall_curve(frame_gt, frame_predict)
        ap_score = auc(recall, precision)
        print(ap_score)

if __name__ == "__main__":
    if torch.cuda.is_available():
        device = 'cuda:0'
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = 'cpu'
        print("Using CPU: No GPU available")

    args = parse_args()
    if args.debug:
        pdb.set_trace()
    config = Config(args)
    worker_init_fn = None
    config.len_feature = 1024
    if config.seed >= 0:
        utils.set_seed(config.seed)
        worker_init_fn = np.random.seed(config.seed)
    net = WSAD(input_size=config.len_feature, flag="Test", a_nums=60, n_nums=60)
    net = net.cuda()
    test_loader = data.DataLoader(
        UCF_crime(root_dir=config.root_dir, mode='Test', modal=config.modal, num_segments=config.num_segments, len_feature=config.len_feature),
        batch_size=1, shuffle=False, num_workers=config.num_workers,
        worker_init_fn=worker_init_fn)
    valid(net, config, test_loader, model_file=os.path.join(args.model_path, "target_99/model_round_4.pkl"))  # Change model here
