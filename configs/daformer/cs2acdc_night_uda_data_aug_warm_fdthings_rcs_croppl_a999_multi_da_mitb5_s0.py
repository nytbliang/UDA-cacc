# ---------------------------------------------------------------
# Copyright (c) 2021-2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

_base_ = [
    '../_base_/default_runtime.py',
    # DAFormer Network Architecture
    '../_base_/models/daformer_sepaspp_mitb5.py',
    # Cityscapes->ACDC Night Data Loading
    '../_base_/datasets/uda_cityscapes_to_acdc_night_512x512_wo_norm.py',
    # Basic UDA Self-Training
    '../_base_/uda/multi_da.py',
    # AdamW Optimizer
    '../_base_/schedules/adamw.py',
    # Linear Learning Rate Warmup with Subsequent Linear Decay
    '../_base_/schedules/poly10warm.py'
]
# Random Seed
seed = 0
# Modifications to Basic UDA
uda = dict(
    # Increased Alpha
    alpha=0.999,
    # Thing-Class Feature Distance
    imnet_feature_dist_lambda=0.005,
    imnet_feature_dist_classes=[6, 7, 11, 12, 13, 14, 15, 16, 17, 18],
    imnet_feature_dist_scale_min_ratio=0.75,
    # Pseudo-Label Crop
    pseudo_weight_ignore_top=15,
    pseudo_weight_ignore_bottom=120,
    # Style transfer
    enable_fft=True,
    enable_style_gan=False,
    fft_beta=0.01,
    enable_src_in_tgt = False,
    enable_tgt_in_src = False,
    enable_st_consistency = False,
    st_consistency_lambda = 0.5,
    enable_src_in_tgt_b4_train = True,
    # Mix setting
    enable_mix_ss_tt = False,
    enable_mix_st_tt = False,
    enable_mix_ss_ts = False,
    mix_consistency_lambda = 1,
    # Normalize outside pipeline
    to_rgb = True,
    norm_cfg = dict(
        mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]),
    )
data = dict(
    train=dict(
        # Rare Class Sampling
        rare_class_sampling=dict(
            min_pixels=3000, class_temp=0.01, min_crop_ratio=0.5)))
# Optimizer Hyperparameters
optimizer_config = None
optimizer = dict(
    lr=6e-05,
    paramwise_cfg=dict(
        custom_keys=dict(
            head=dict(lr_mult=10.0),
            pos_block=dict(decay_mult=0.0),
            norm=dict(decay_mult=0.0))))
n_gpus = 1
runner = dict(type='IterBasedRunner', max_iters=40000)
# Logging Configuration
checkpoint_config = dict(by_epoch=False, interval=4000, max_keep_ckpts=2)
evaluation = dict(interval=4000, metric='mIoU')
# Meta Information for Result Analysis
name = 'cs2acdc_night_uda_data_aug_warm_fdthings_rcs_croppl_a999_multi_da_mitb5_s0'
exp = 'basic'
name_dataset = 'cityscapes2acdcnight'
name_architecture = 'daformer_sepaspp_mitb5'
name_encoder = 'mitb5'
name_decoder = 'daformer_sepaspp'
name_uda = 'dacs_a999_fd_things_rcs0.01_cpl'
name_opt = 'adamw_6e-05_pmTrue_poly10warm_1x2_40k'