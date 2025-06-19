import os
import argparse
import time
from datetime import datetime
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from seg_datasets import Seg_Dataset,Seg_Dataset_2,PanNukeDataset
from TransUnet_PFM.PFM_Seg_Models import PFM_Seg_Model
from Utils.general_utils import set_global_seed
from Utils.loop_utils import train_one_epoch,evaluate,create_lr_scheduler
#from train_utils import train_one_epoch, evaluate, create_lr_scheduler
import transforms as T
_MODEL2WEIGHTS = {
    'Gigapath':'/path/to/gigapath_weights.bin',
    'UNI':'/path/to/UNI_weights.bin',
    'Ctranspath':'/path/to/Ctranspath_weights.pth', 
    'Conch_V1_5':'/path/to/Conch_V1_5_weights.bin',
    'Digepath':'/path/to/Digepath_weights.pth',
   
}

_MODEL2DIM = {
    'Gigapath':1536,
    'UNI':1024,
    'Ctranspath':768,
    'Conch_V1_5':1024,
    'Digepath':1024,
}


class SegmentationPresetTrain:
    def __init__(self, base_size, crop_size, hflip_prob=0.5, vflip_prob=0.5,
                 mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        min_size = int(0.5 * base_size)
        max_size = int(1.2 * base_size)

        # trans = [T.RandomResize(min_size, max_size)]
        trans = []
        if hflip_prob > 0:
            trans.append(T.RandomHorizontalFlip(hflip_prob))
        if vflip_prob > 0:
            trans.append(T.RandomVerticalFlip(vflip_prob))
        trans.extend([
            T.RandomCrop(crop_size),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
        self.transforms = T.Compose(trans)

    def __call__(self, img, target):
        return self.transforms(img, target)


class SegmentationPresetEval:
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225),resize_size = None):
        if resize_size != None:
            self.transforms = T.Compose([
            T.Resize(resize_size),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
        else:
            self.transforms = T.Compose([
                T.ToTensor(),
                T.Normalize(mean=mean, std=std),
            ])

    def __call__(self, img, target):
        return self.transforms(img, target)


def get_transform(train, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225),resize_size =None):
    base_size = 224
    crop_size = 224

    if train:
        return SegmentationPresetTrain(base_size, crop_size, mean=mean, std=std)
    else:
        return SegmentationPresetEval(mean=mean, std=std,resize_size=resize_size)

def main(args):
    dataset_name = args.dataset_name
    log_dir = args.log_dir
    os.makedirs(log_dir,exist_ok=True)
    set_global_seed(args.seed)
    device = torch.device(args.device)
    batch_size = args.batch_size
    num_classes = args.num_classes

    mean = (0.709, 0.381, 0.224)
    std = (0.127, 0.079, 0.043)

    results_file = "results{}.txt".format(datetime.now().strftime("%Y%m%d-%H%M%S"))
    results_file = os.path.join(log_dir,results_file)

    if args.has_split == True:
        train_dataset = Seg_Dataset(args.dataset_csv,
                                    group='train',
                                    transforms=get_transform(train=True, mean=mean, std=std))

        if dataset_name == 'ESD':
            test_dataset = Seg_Dataset(args.dataset_csv,
                                    group='test',
                                    transforms=get_transform(train=False, mean=mean, std=std,resize_size=224),slide_window_val=False)
        elif dataset_name in ['FJSL','GLAS','CRAG']:
            test_dataset = Seg_Dataset(args.dataset_csv,
                                    group='test',
                                    transforms=get_transform(train=False, mean=mean, std=std),slide_window_val=True)        
        
        else:
            test_dataset = Seg_Dataset(args.dataset_csv,
                                    group='test',
                                    transforms=get_transform(train=False, mean=mean, std=std),slide_window_val=False)
    

    num_workers = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               num_workers=num_workers,
                                               shuffle=True,
                                               pin_memory=True,
                                               collate_fn=train_dataset.collate_fn)

    # val_loader = torch.utils.data.DataLoader(val_dataset,
    #                                          batch_size=16,
    #                                          num_workers=num_workers,
    #                                          pin_memory=True,
    #                                          collate_fn=val_dataset.collate_fn)
    if dataset_name in ['FJSL','GLAS','CRAG']:
        test_loader = torch.utils.data.DataLoader(test_dataset,
                                             batch_size=1,
                                             num_workers=num_workers,
                                             pin_memory=True,
                                             collate_fn=test_dataset.collate_fn)
    else:
        test_loader = torch.utils.data.DataLoader(test_dataset,
                                             batch_size=16,
                                             num_workers=num_workers,
                                             pin_memory=True,
                                             collate_fn=test_dataset.collate_fn)
    model = PFM_Seg_Model(PFM_name=args.pfm_model,PFM_weights_path=_MODEL2WEIGHTS[args.pfm_model],
                          num_classes=args.num_classes,emb_dim=_MODEL2DIM[args.pfm_model])
    model.to(device)

    params_to_optimize = [p for p in model.parameters() if p.requires_grad]

    optimizer = torch.optim.SGD(
        params_to_optimize,
        lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay
    )

    scaler = None

    lr_scheduler = create_lr_scheduler(optimizer, len(train_loader), args.epochs, warmup=True)
    best_dice = 0.
    start_time = time.time()
    for epoch in range(0, args.epochs):
        mean_loss, lr = train_one_epoch(model, optimizer, train_loader, device, epoch, num_classes,
                                        lr_scheduler=lr_scheduler, print_freq=10, scaler=scaler)
        val_iou, val_dice = evaluate(model, test_loader, device=device, num_classes=num_classes)
        print(f"dice coefficient: {val_dice:.3f}")
        print(f"iou coefficient: {val_iou:.3f}")
        # print(val_dice)
        # print(val_iou)
        # write into txt
        with open(results_file, "a") as f:
            train_info = f"[epoch: {epoch}]\n" \
                         f"train_loss: {mean_loss:.4f}\n" \
                         f"lr: {lr:.6f}\n" \
                         f"val dice coefficient: {val_dice:.5f}\n" \
                         f"val iou coefficient: {val_iou:.5f}\n"
            f.write(train_info + "\n\n")

        if best_dice < val_dice:
            best_dice = val_dice


        save_file = {"model": model.state_dict(),
                     "optimizer": optimizer.state_dict(),
                     "lr_scheduler": lr_scheduler.state_dict(),
                     "epoch": epoch,
                     "args": args}
        save_path = os.path.join(log_dir,'best_model.pth')
        torch.save(save_file, save_path)
    best_path = os.path.join(log_dir,'best_model.pth')
    model.load_state_dict(torch.load(best_path)['model'],strict=True)
    test_iou, test_dice = evaluate(model, test_loader, device=device, num_classes=num_classes)
    with open(results_file, "a") as f:
        test_info = f"test dice coefficient: {test_dice:.5f}\n" \
                    f"test iou: {test_iou:.5f}"
        f.write(test_info + "\n\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="pytorch unet training")
    parser.add_argument('--dataset_name',default='GLAS')
    parser.add_argument("--dataset_csv", default="/mnt/net_sda/fumingxi_fmx/大模型分割下游/Datasets_csv_fmx/GLAS_split.csv", help="dataset csv")
    parser.add_argument('--has_split',action='store_false')
    parser.add_argument("--seed",type=int,default=2025)
    parser.add_argument("--pfm_model",default='Ctranspath',help='pfm')
    parser.add_argument("--max_num",type=int,default=-1)
    parser.add_argument("--log_dir",default='/path/to/log_dir',help='log_dir')
    parser.add_argument("--num_classes", default=2, type=int,help='num_classes')
    parser.add_argument("--device", default="cuda:7", help="training device")
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument('--val_batch_size',default=32)
    parser.add_argument("--epochs", default=40, type=int, metavar="N",
                        help="number of total epochs to train")
    parser.add_argument('--lr', default=0.01, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    args = parser.parse_args()

    main(args)
    
    
    

