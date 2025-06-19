import timm
import torch
from collections import OrderedDict
from PIL import Image
import os
def create_DigePath_based_model(checkpoint_path, patch_size, num_classes: int = 0):
    dige_kwargs = {
    'model_name': f'vit_large_patch{patch_size}_224',
    'img_size': 224, 
    'patch_size': patch_size, 
    'init_values': 1e-5, 
    'num_classes': num_classes, 
    'dynamic_img_size': True}
    model = timm.create_model(**dige_kwargs)
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    new_state_dict = OrderedDict({k.replace('backbone.', ''): v for k, v in state_dict['teacher'].items()})
    missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
    print(f"Missing Keys: {missing_keys}")
    print(f"Unexpected Keys: {unexpected_keys}")
    return model
        
def load_class2id_mapping(filepath):
    class2id = {}
    with open(filepath, 'r') as file:
        for line in file:
            class_name, class_id = line.strip().split(',')
            class2id[class_name] = int(class_id)
    return class2id

def save_topk_retrieval_imgs(save_topk_dir,img_paths,labels):
    for i in range(len(img_paths)):
        img_path = img_paths[i]
        img_name = os.path.basename(img_path).split('.')[0]
        img = Image.open(img_path)
        img.save(os.path.join(save_topk_dir,f'top{i}_{img_name}_{labels[i]}.png'))
        
def save_results_as_txt(results_text, save_path):
    with open(save_path, 'w') as f:
        f.write(results_text)
