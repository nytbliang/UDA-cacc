# ---------------------------------------------------------------
# Copyright (c) 2021-2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

_base_ = [
    '../_base_/default_runtime.py',
    # DAFormer Network Architecture
    '../_base_/models/daformer_sepaspp_mitb5.py',
    # GTA->Cityscapes Data Loading
    # '../_base_/datasets/uda_gta_to_cityscapes_512x512_wo_norm.py',
    '../_base_/datasets/uda_gta_to_cityscapes_512x512.py',
    # Basic UDA Self-Training
    '../_base_/uda/multi_da.py',
    # AdamW Optimizer
    '../_base_/schedules/adamw.py',
    # Linear Learning Rate Warmup with Subsequent Linear Decay
    '../_base_/schedules/poly10warm.py'
]
# Random Seed
seed = 0
model = dict(
    decode_head=dict(
        channels=512,
        decoder_params=dict(
            embed_dims=512,
        ),
    )
)
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
    enable_src_in_tgt=False,
    enable_tgt_in_src=False,
    enable_src_in_tgt_b4_train=False,
    enable_st_consistency=False,
    st_consistency_lambda=0.1,
    # Normalize outside pipeline
    to_rgb = True,
    norm_cfg = dict(
        mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]),
    # Category contrast
    enable_ctrst = True,
    ctrst_lambda = 0.05,
    rare_class_id = [4,5,6,7,11,12,13,14,15,16,17,18],
    temperature=0.2,
    mix_proto_alpha=0.5)
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
runner = dict(type='IterBasedRunner', max_iters=60000)
# Logging Configuration
checkpoint_config = dict(by_epoch=False, interval=2000, max_keep_ckpts=2)
evaluation = dict(interval=2000, metric='mIoU')
resume_from = '/root/autodl-tmp/DAFormer/work_dirs/local-basic/221215_1906_gta2cs_uda_ctrst_warm_fdthings_rcs_croppl_a999_multi_da_mitb5_s0_f6656/iter_40000.pth'
# Meta Information for Result Analysis
name = 'gta2cs_uda_ctrst_warm_fdthings_rcs_croppl_a999_multi_da_mitb5_s0'
exp = 'basic'
name_dataset = 'gta2cityscapes'
name_architecture = 'daformer_sepaspp_mitb5'
name_encoder = 'mitb5'
name_decoder = 'daformer_sepaspp'
name_uda = 'dacs_a999_fd_things_rcs0.01_cpl'
name_opt = 'adamw_6e-05_pmTrue_poly10warm_1x2_40k'