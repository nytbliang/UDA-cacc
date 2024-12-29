# ---------------------------------------------------------------
# Copyright (c) 2021-2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

# Baseline UDA
uda = dict(
    type='MultiDA',
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
    # Discriminator
    power=0.9,
    # Pixel level without class in output space
    enable_px_wo_cls_d=False,
    px_wo_cls_adv_lambda=0.01,
    lr_px_wo_cls_d=1e-4,
    # Pixel level in feature space
    enable_px_d=False,
    px_adv_lambda=0.01,
    lr_px_d=1e-4,
    # Image level in feature space
    enable_img_d=False,
    img_adv_lambda=0.01,
    lr_img_d=1e-4,
    # Image classifier
    enable_cls=False,
    cls_pretrained='/root/autodl-tmp/DAFormer/pretrained/ClsEp50.pth',
    cls_thred=0.5,
    # Style transfer
    enable_fft=False,
    enable_style_gan=False,
    fft_beta=0.01,
    enable_src_in_tgt = False,
    enable_tgt_in_src = False,
    enable_st_consistency = False,
    st_consistency_lambda = 0.5,
    enable_src_in_tgt_b4_train = False,
    # Mix setting
    enable_mix_ss_tt = False,
    enable_mix_st_tt = False,
    enable_mix_ss_ts = False,
    mix_consistency_lambda = 1,
    # Rare class mix
    enable_rcm=False,
    rare_class_mix=[],
    # Normalize outside pipeline
    to_rgb = True,
    norm_cfg = dict(
        mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]),
    # Category contrast
    temperature=0.2,
    mix_proto_alpha=0.5,
    enable_ctrst_feat = False,
    enable_ctrst_out = False,
    ctrst_feat_lambda = 0.5,
    ctrst_out_lambda = 0.5,
)
use_ddp_wrapper = True
