import os
import math
import argparse
import torch.nn as nn
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import json
from torchvision import transforms
import warnings
import pandas as pd
from utils.eval_linear_probe import eval_linear_probe
from utils.fewshot import eval_knn,eval_fewshot
from utils.get_encoder import get_pathology_encoder
from utils.roi_dataset import ROIDataSet,FeatDataSet
from utils.get_transforms import get_transforms
from utils.metrics import get_eval_metrics
from utils.protonet import ProtoNet
from utils.common_utils import save_topk_retrieval_imgs,save_results_as_txt
from utils.extract_patch_features import extract_patch_features_from_dataloader
from utils.linear_train_utils import train_one_epoch,evaluate
warnings.filterwarnings("ignore")
import random
import numpy as np
import torch

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    

def main(args):
    TASK = args.TASK
    os.makedirs(args.log_dir,exist_ok=True)
    save_description_path = os.path.join(args.log_dir,'description.txt')
    if args.log_description is not None:
        save_results_as_txt(args.log_description,save_description_path)
    for task in TASK:
        os.makedirs(os.path.join(args.log_dir,task),exist_ok=True)
    if len(TASK) == 0:
        raise ValueError("No task specified")
    device = torch.device(args.device)
    train_assert = torch.load(args.train_feature_path,weights_only=False)
    test_assert = torch.load(args.test_feature_path,weights_only=False)
    train_feats = torch.Tensor(train_assert['embeddings'])

    train_labels = torch.Tensor(train_assert['labels']).type(torch.long)
    test_feats = torch.Tensor(test_assert['embeddings'])
    test_labels = torch.Tensor(test_assert['labels']).type(torch.long)
    val_feats,val_labels = None,None
    if 'Linear-Probe' in TASK:
        linprobe_eval_metrics, linprobe_dump = eval_linear_probe(
            train_feats = train_feats,
            train_labels = train_labels,
            valid_feats = val_feats ,
            valid_labels = val_labels,
            test_feats = test_feats,
            test_labels = test_labels,
            device = device,
            max_iter = args.max_iteration,
            use_sklearn = args.use_sklearn,
            verbose= True,
            random_state = args.seed)
        print("------------Linear Probe Evaluation------------")
        print(linprobe_eval_metrics)
        save_path = os.path.join(args.log_dir,'Linear-Probe','results.txt')
        save_path_dump = os.path.join(args.log_dir,'Linear-Probe','dump.csv')
        save_results_as_txt(str(linprobe_eval_metrics),save_path)
        # save_results_as_txt(str(plinprobe_dum),save_path_dump)
        # linprobe_dump.to_csv(save_path_dump)
        linprobe_dump = {key: linprobe_dump[key] for key in ['preds_all', 'targets_all','probs_all']}
        linprobe_dump = {key: value.tolist() for key, value in linprobe_dump.items()}
        linprobe_dump = pd.DataFrame(linprobe_dump)
        linprobe_dump.to_csv(save_path_dump)

    
    if 'KNN-Proto' in TASK:
        knn_eval_metrics, knn_dump, proto_eval_metrics, proto_dump = eval_knn(
            train_feats = train_feats,
            train_labels = train_labels,
            valid_feats = val_feats ,
            valid_labels = val_labels,
            test_feats = test_feats,
            test_labels = test_labels,
            center_feats = True,
            normalize_feats = True,
            n_neighbors = args.n_neighbors,)
        print("------------KNN Evaluation------------")
        print(knn_eval_metrics)
        save_path_knn = os.path.join(args.log_dir,'KNN-Proto','KNN_results.txt')
        print("------------Proto Evaluation------------")
        print(proto_eval_metrics)
        save_path_proto = os.path.join(args.log_dir,'KNN-Proto','Proto_results.txt')
        save_results_as_txt(str(knn_eval_metrics),save_path_knn)
        save_results_as_txt(str(proto_eval_metrics),save_path_proto)
        knn_dump_path = os.path.join(args.log_dir,'KNN-Proto','KNN_dump.txt')
        proto_dump_path = os.path.join(args.log_dir,'KNN-Proto','Proto_dump.txt')
        save_results_as_txt(str(knn_dump),knn_dump_path)
        save_results_as_txt(str(proto_dump),proto_dump_path)
    
    if 'Few-shot' in TASK:
        n_way = args.n_way
        ways = [int(way) for way in n_way.split(',')]
        for way in ways:
            save_dir = os.path.join(args.log_dir,'Few-shot',f'way_{way}')
            os.makedirs(save_dir,exist_ok=True)
            for shot in args.n_shot:
                print(f"Few-shot evaluation with {shot} examples per class")
                fewshot_episodes, fewshot_dump = eval_fewshot(
                train_feats = train_feats,
                train_labels = train_labels,
                valid_feats = val_feats ,
                valid_labels = val_labels,
                test_feats = test_feats,
                test_labels = test_labels,
                n_iter = args.n_iter, # draw 500 few-shot episodes
                n_way = way, # use all class examples
                n_shot = shot, # 4 examples per class (as we don't have that many)
                n_query = test_feats.shape[0], # evaluate on all test samples
                center_feats = True,
                normalize_feats = True,
                average_feats = True,)
                print("------------Fewshot Evaluation------------")
                print(fewshot_dump)
                save_path = os.path.join(save_dir,f'results_{shot}_shot.txt')
                save_results_as_txt(str(fewshot_dump),save_path)
                save_path_episodes = os.path.join(save_dir,f'results_{shot}_shot_episodes.csv')
                fewshot_episodes.to_csv(save_path_episodes)
    



    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    General_args = parser.add_argument_group('General')
    Linear_probe_args = parser.add_argument_group('Linear_probe')
    Linear_train_args = parser.add_argument_group('Linear_train')
    KNN_and_proto_args = parser.add_argument_group('KNN_and_proto')
    Few_shot_args = parser.add_argument_group('Few-shot')
    ROI_retrieval_args = parser.add_argument_group('ROI_retrieval')
    # General
    General_args.add_argument('--TASK', type=list, default=['Linear-Probe','KNN-Proto','Few-shot'],choices=['Linear-Probe','KNN-Proto','Few-shot'])
    General_args.add_argument('--train_feature_path', type=str, default='/path/to/CAMEL-train-Digepath.pt')
    General_args.add_argument('--test_feature_path', type=str, default='/path/to/CAMEL-test-Digepath.pt')
    General_args.add_argument('--device', default='cuda:7', help='device id')
    General_args.add_argument('--log_dir', default='/path/to/log_dir', help='path where to save')
    General_args.add_argument('--log_description', type=str, default='test code') 
    General_args.add_argument('--seed', type=int, default=2024) 
    # Linear_probe
    Linear_probe_args.add_argument('--max_iteration', type=int, default=1000)
    Linear_probe_args.add_argument('--use_sklearn', default=False, help='use sklearn logistic regression')
    # KNN_and_proto
    KNN_and_proto_args.add_argument('--n_neighbors', type=int, default=20)
    # Few_shot
    Few_shot_args.add_argument('--n_iter', type=int, default=100) 
    Few_shot_args.add_argument('--n_way', type=str, default='2') 
    Few_shot_args.add_argument('--n_shot', type=list, default=[1,2,4,8,16,32,64,128,256]) 
    # ROI_retrieval
    opt = parser.parse_args()
    seed = opt.seed
    set_seed(seed)
    main(opt)


 
