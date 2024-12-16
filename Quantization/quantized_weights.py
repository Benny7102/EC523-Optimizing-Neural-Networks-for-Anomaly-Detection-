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

def valid(net, config, test_loader, model_file=None):
    with torch.no_grad():
        net.eval()
        net.flag = "Test"

        device = torch.device('cuda')
        net.to(device)

        if model_file is not None:
            state_dict = torch.load(model_file, map_location='cuda')
            for k,v in state_dict.items():
                if v.dtype == torch.float32:
                    state_dict[k] = v.half()
            net.load_state_dict(state_dict)

        load_iter = iter(test_loader)
        frame_gt = np.load("frame_label/gt-ucf.npy")
        frame_predict = None
        ucf_pdict = {
            "Abuse":{}, "Arrest":{}, "Arson":{}, "Assault":{}, "Burglary":{},
            "Explosion":{}, "Fighting":{}, "RoadAccidents":{}, "Robbery":{},
            "Shooting":{}, "Shoplifting":{}, "Stealing":{}, "Vandalism":{},
            "Normal":{}
        }
        ucf_gdict = {
            "Abuse":{}, "Arrest":{}, "Arson":{}, "Assault":{}, "Burglary":{},
            "Explosion":{}, "Fighting":{}, "RoadAccidents":{}, "Robbery":{},
            "Shooting":{}, "Shoplifting":{}, "Stealing":{}, "Vandalism":{},
            "Normal":{}
        }

        cls_label = []
        cls_pre = []
        temp_predict = torch.zeros((0), dtype=torch.float16, device=device)

        count = 0
        total_inference_time = 0
        inference_count = 0

        for i in range(len(test_loader.dataset)):
            _data, _label, _name = next(load_iter)
            _name = _name[0]

            _data = _data.to(device).half()
            _label = _label.to(device)  

            start_time = time.time()

            res = net(_data)
            end_time = time.time()

            inference_time = end_time - start_time
            total_inference_time += inference_time
            inference_count += 1

            a_predict = res["frame"]  
            temp_predict = torch.cat([temp_predict, a_predict], dim=0)

            if (i + 1) % 10 == 0:
                cls_label.append(int(_label))
                a_predict_cpu = temp_predict.mean(0).float().cpu().numpy()
                pl = len(a_predict_cpu) * 16

                if "Normal" in _name:
                    ucf_pdict["Normal"][_name] = np.repeat(a_predict_cpu, 16)
                    ucf_gdict["Normal"][_name] = frame_gt[count:count + pl]
                else:
                    ucf_pdict[_name[:-3]][_name] = np.repeat(a_predict_cpu, 16)
                    ucf_gdict[_name[:-3]][_name] = frame_gt[count:count + pl]

                count += pl
                cls_pre.append(1 if a_predict_cpu.max() > 0.5 else 0)

                fpre_ = np.repeat(a_predict_cpu, 16)
                if frame_predict is None:
                    frame_predict = fpre_
                else:
                    frame_predict = np.concatenate([frame_predict, fpre_])

                temp_predict = torch.zeros((0), dtype=torch.float16, device=device)

        average_inference_time = total_inference_time / inference_count
        print("average inference time (ms): ", average_inference_time * 1000)

        frame_gt = np.load("frame_label/gt-ucf.npy")
        np.save('frame_label/ucf_pre.npy', frame_predict)
        np.save('frame_label/ucf_pre_dict.npy', ucf_pdict)
        np.save('frame_label/ucf_gt_dict.npy', ucf_gdict)

        fpr, tpr, _ = roc_curve(frame_gt, frame_predict)
        auc_score = auc(fpr, tpr)
        print("auc score: ", auc_score)
        precision, recall, th = precision_recall_curve(frame_gt, frame_predict)
        ap_score = auc(recall, precision)
        print("ap_score: ", ap_score)

if __name__ == "__main__":
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
    device = torch.device('cuda')
    net.to(device)

    test_loader = data.DataLoader(
        UCF_crime(
            root_dir=config.root_dir, mode='Test', modal=config.modal,
            num_segments=config.num_segments, len_feature=config.len_feature
        ),
        batch_size=1,
        shuffle=False,
        num_workers=config.num_workers,
        worker_init_fn=worker_init_fn
    )

    valid(net, config, test_loader, model_file=os.path.join(args.model_path, "ucf_trans_2022.pkl"))
