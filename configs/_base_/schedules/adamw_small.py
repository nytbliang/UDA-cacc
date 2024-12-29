# Obtained from: https://github.com/open-mmlab/mmsegmentation/tree/v0.16.0

# optimizer
# optimizer = dict(
#     type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01)
# optimizer_config = dict()
optimizer = dict(
    type='AdamW',
    lr=7e-5,
    weight_decay=0.0001,
    paramwise_cfg=dict(custom_keys={'decode_head': dict(lr_mult=10.0)}),
)
optimizer_config = dict()