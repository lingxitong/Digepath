if __name__ == '__main__':
    timm_kwargs = {
    "img_size": 224,
    "in_chans": 3,
    "patch_size": 16,
    "embed_dim": 1536,
    "depth": 40,
    "num_heads": 24,
    "mlp_ratio": 5.33334,
    "num_classes": 0,
    "dynamic_img_size":True
}
    import timm
    model = timm.create_model("vit_giant_patch14_dinov2", pretrained=False, **timm_kwargs)
    import torch
    x = torch.randn(1,3,448,448)
    y = model(x)
    print(y.shape)