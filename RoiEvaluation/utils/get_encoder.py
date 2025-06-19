import os
import logging
import timm
import torch
import torch.nn as nn
import warnings 
from collections import OrderedDict


warnings.filterwarnings("ignore")
from torchvision import transforms
from .common_utils import create_DigePath_based_model

Name2Chkpt = {
'UNI':'/path/to/UNI_weights.bin', 
'Gigapath': '/path/to/Gigapath_weights.bin', 
'Conch-v1_5': '/path/to/Conch-v1_5_weights.bin', 
'Ctranspath':'/path/to/Ctranspath_weights.pth',
'Digepath':'/path/to/Digepath_weights.pth', 
def get_pathology_encoder(model_name: str,num_classes: int = 0,trainable_head: bool = False):
    if model_name == 'Digepath':
        uni_kwargs = {
        'model_name': 'vit_large_patch16_224',
        'img_size': 224, 
        'patch_size': 16, 
        'init_values': 1e-5, 
        'num_classes': num_classes, 
        'dynamic_img_size': True}
        model = timm.create_model(**uni_kwargs)
        state_dict = torch.load(Name2Chkpt[model_name], map_location="cpu")
        new_state_dict = OrderedDict({k.replace('backbone.', ''): v for k, v in state_dict['student'].items()})
        model.load_state_dict(new_state_dict,strict=False)
        return model
    if model_name == 'UNI':
        uni_kwargs = {
        'model_name': 'vit_large_patch16_224',
        'img_size': 224, 
        'patch_size': 16, 
        'init_values': 1e-5, 
        'num_classes': num_classes, 
        'dynamic_img_size': True}
        model = timm.create_model(**uni_kwargs)
        state_dict = torch.load(Name2Chkpt[model_name], map_location="cpu")
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=True)
        if trainable_head:
            for name, param in model.named_parameters():
                if name != 'head.weight' and name != 'head.bias':
                    param.requires_grad = False
                else:
                    param.requires_grad = True
        print(f"Missing Keys: {missing_keys}")
        print(f"Unexpected Keys: {unexpected_keys}")
        print(f'--------successfully load UNI with {num_classes} classes head and {trainable_head} trainable head--------')
        return model
    elif 'DigePath' in model_name:
        patch_size = int(model_name.split('DigePath')[1][:2])
        model =  create_DigePath_based_model(Name2Chkpt[model_name], patch_size, num_classes)
        if trainable_head:
            for name, param in model.named_parameters():
                if name != 'head.weight' and name != 'head.bias':
                    param.requires_grad = False
                else:
                    param.requires_grad = True
        print(f'--------successfully load {model_name} with {num_classes} classes head and {trainable_head} trainable head--------')
        return model
    elif model_name == 'Gigapath':
        gig_config = {
        "architecture": "vit_giant_patch14_dinov2",
        "num_classes": 0,
        "num_features": 1536,
        "global_pool": "token",
        "model_args": {
        "img_size": 224,
        "in_chans": 3,
        "dynamic_img_size": True,
        "patch_size": 16,
        "embed_dim": 1536,
        "depth": 40,
        "num_heads": 24,
        "init_values": 1e-05,
        "mlp_ratio": 5.33334,
        "num_classes": num_classes}} 
        model = timm.create_model("vit_giant_patch14_dinov2", pretrained=False, **gig_config['model_args'])
        state_dict = torch.load(Name2Chkpt[model_name], map_location="cpu")
        model.load_state_dict(state_dict, strict=True)
        if trainable_head:
            for name, param in model.named_parameters():
                if name != 'head.weight' and name != 'head.bias':
                    param.requires_grad = False
                else:
                    param.requires_grad = True
        return model
    elif model_name == 'Conch-v1_5':
        from .conch_v1_5_config import ConchConfig
        from .build_conch_v1_5 import build_conch_v1_5
        checkpoint_path = Name2Chkpt[model_name]
        conch_v1_5_config = ConchConfig()
        model = build_conch_v1_5(conch_v1_5_config, checkpoint_path)
        if trainable_head:
            for name, param in model.named_parameters():
                if name != 'head.weight' and name != 'head.bias':
                    param.requires_grad = False
                else:
                    param.requires_grad = True
        return model
    elif model_name == 'Ctranspath':
        from .ctrans import ctranspath
        checkpoint_path = Name2Chkpt[model_name]
        model = ctranspath()
        model.head = nn.Identity()
        state_dict = torch.load(checkpoint_path,weights_only=True)['model']
        state_dict = {key: val for key, val in state_dict.items() if 'attn_mask' not in key}
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if trainable_head:
            for name, param in model.named_parameters():
                if name != 'head.weight' and name != 'head.bias':
                    param.requires_grad = False
                else:
                    param.requires_grad = True
        return model
    
