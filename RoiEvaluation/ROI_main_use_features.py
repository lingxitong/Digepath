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

# 设置全局随机种子
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
    model = get_pathology_encoder(args.model_name,ctranspath_size = args.resize_size)
    device = torch.device(args.device)
    model = model.to(device)
    model.eval()
    data_transform = get_transforms(args.resize_size)
    train_dataset = ROIDataSet(csv_path=args.dataset_split_csv,domain='train',transform=data_transform["train"],class2id_txt=args.class2id_txt)
    val_dataset = ROIDataSet(csv_path=args.dataset_split_csv,domain='val',transform=data_transform["val"],class2id_txt=args.class2id_txt)
    test_dataset = ROIDataSet(csv_path=args.dataset_split_csv,domain='test',transform=data_transform["test"],class2id_txt=args.class2id_txt)
    if val_dataset.__len__() == 0:
        val_dataset = None
        val_loader = None
    else:
        val_loader = torch.utils.data.DataLoader(val_dataset,batch_size=args.batch_size,shuffle=False,num_workers=8,collate_fn=ROIDataSet.collate_fn)
    train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=args.batch_size,shuffle=True,num_workers=8,collate_fn=ROIDataSet.collate_fn)
    test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=args.batch_size,shuffle=False,num_workers=8,collate_fn=ROIDataSet.collate_fn)
    train_assert = extract_patch_features_from_dataloader(model,train_loader,device,args.model_name)
    test_assert = extract_patch_features_from_dataloader(model,test_loader,device,args.model_name)
    if val_loader is not None:
        val_assert = extract_patch_features_from_dataloader(model,val_loader,device,args.model_name)
    else:
        val_assert = None,None
    train_feats = torch.Tensor(train_assert['embeddings'])
    train_labels = torch.Tensor(train_assert['labels']).type(torch.long)
    test_feats = torch.Tensor(test_assert['embeddings'])
    test_labels = torch.Tensor(test_assert['labels']).type(torch.long)
    if val_loader is not None:
        val_feats = torch.Tensor(val_assert['embeddings'])
        val_labels = torch.Tensor(val_assert['labels']).type(torch.long)
    else:
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
    
    if 'Proto-ROI-retrieval' in TASK:
        if args.combine_trainval and val_feats is not None:
            train_feats = torch.cat([train_feats, val_feats], dim=0)
            train_labels = torch.cat([train_labels, val_labels], dim=0)
        proto_clf = ProtoNet(metric='L2', center_feats=True, normalize_feats=True)
        proto_clf.fit(train_feats, train_labels)
        print('What our prototypes look like', proto_clf.prototype_embeddings.shape)
        test_pred = proto_clf.predict(test_feats)
        eval_metrics = get_eval_metrics(test_labels, test_pred, get_report=False)
        ans_df = pd.DataFrame({'test_lables':test_labels,'test_pred':test_pred})
        save_path_csv = os.path.join(args.log_dir,'Proto-ROI-retrieval','results.csv')
        ans_df.to_csv(save_path_csv)
        save_path = os.path.join(args.log_dir,'Proto-ROI-retrieval','results.txt')
        save_results_as_txt(str(eval_metrics),save_path)
        print("------------ROI Retrieval Evaluation------------")
        dist, topk_inds = proto_clf._get_topk_queries_inds(test_feats, topk=args.topk)
        num_classes = len(torch.unique(test_labels))
        for now_class_id in range(0,num_classes):
            adi_topk_inds = topk_inds[now_class_id]
            class_name = test_dataset.id2class_dict[now_class_id]
            img_paths, labels = test_dataset.get_imgs_from_idxs(adi_topk_inds)
            labels = [test_dataset.id2class_dict[label] for label in labels]
            save_topk_dir = os.path.join(args.log_dir,'Proto-ROI-retrieval',f'ROI_retrieval_top{args.topk}',f'{class_name}')
            os.makedirs(save_topk_dir,exist_ok=True)
            save_topk_retrieval_imgs(save_topk_dir,img_paths,labels)
            


    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    General_args = parser.add_argument_group('General')
    Linear_probe_args = parser.add_argument_group('Linear_probe')
    Linear_train_args = parser.add_argument_group('Linear_train')
    KNN_and_proto_args = parser.add_argument_group('KNN_and_proto')
    Few_shot_args = parser.add_argument_group('Few-shot')
    ROI_retrieval_args = parser.add_argument_group('ROI_retrieval')
    # General
    General_args.add_argument('--TASK', type=list, default=['Proto-ROI-retrieval'],choices=['Linear-Probe','KNN-Proto','Few-shot','Proto-ROI-retrieval'])
    General_args.add_argument('--resize_size', type=int, default=224)
    General_args.add_argument('--batch-size', type=int, default=256)
    General_args.add_argument('--dataset_split_csv', type=str, default='/mnt/net_sda/lxt/Supervised_Classification/datasets/CRC-100K.csv')
    General_args.add_argument('--class2id_txt', type=str, default='/mnt/net_sda/lxt/Supervised_Classification/datasets/CRC-100K.txt')
    General_args.add_argument('--model_name', default='DigePath16-11-28-33w-iter', help='create model name')
    General_args.add_argument('--device', default='cuda:2', help='device id')
    General_args.add_argument('--log_dir', default='/mnt/net_sda/lxt/Supervised_Classification/digepath-downstream/logs/1000-digepath-test-Linear-use_sklearn', help='path where to save')
    General_args.add_argument('--log_description', type=str, default='测试代码') # 详细的实验描述为了后续好区分
    General_args.add_argument('--seed', type=int, default=2024) 
    # Linear_probe
    Linear_probe_args.add_argument('--max_iteration', type=int, default=1000)
    Linear_probe_args.add_argument('--use_sklearn', default=False, help='use sklearn logistic regression')
    # KNN_and_proto
    KNN_and_proto_args.add_argument('--n_neighbors', type=int, default=20)
    # Few_shot
    Few_shot_args.add_argument('--n_iter', type=int, default=100) # 多少个事件
    Few_shot_args.add_argument('--n_way', type=str, default='2,3,4,5,6,7,8,9,10,11') # 不能大于类别数
    Few_shot_args.add_argument('--n_shot', type=list, default=[1,2,4,8,16,32,64,128,256]) # 每个类别的样本数
    # ROI_retrieval
    ROI_retrieval_args.add_argument('--combine_trainval', type=bool, default=True)
    ROI_retrieval_args.add_argument('--topk', type=int, default=5)
    opt = parser.parse_args()
    model_name = opt.model_name
    seed = opt.seed
    set_seed(seed)
    main(opt)


 