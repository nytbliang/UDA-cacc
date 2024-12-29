# ---------------------------------------------------------------
# Copyright (c) 2021-2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

# Baseline UDA
uda = dict(
    type='MultiCA',
    alpha=0.99,
    pseudo_threshold=0.968,
    pseudo_weight_ignore_top=0,
    pseudo_weight_ignore_bottom=0,
    imnet_feature_dist_lambda=0,
    imnet_feature_dist_classes=None,
    imnet_feature_dist_scale_min_ratio=None,
    mix='class',
    blur=True,
    color_jitter_strength=0.2,
    color_jitter_probability=0.2,
    debug_img_interval=1000,
    print_grad_magnitude=False,

    # MultiCA
    power=0.9,

    # Pixel level
    lr_px_d=1e-4,
    enable_px_d_feat=False,
    enable_px_d_out=True,
    px_adv_feat_lambda=0.001,
    px_adv_out_lambda=0.001,
    # Image level
    lr_img_d=1e-4,
    enable_img_d_feat=False,
    enable_img_d_out=True,
    img_adv_feat_lambda=0.001,
    img_adv_out_lambda=0.001,

    # Image classifier
    enable_cls=True,
    cls_pretrained='/root/autodl-tmp/DAFormer/pretrained/ep50.pth',
    cls_thred=0.5,
)
use_ddp_wrapper = True
