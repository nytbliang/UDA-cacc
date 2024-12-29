# ---------------------------------------------------------------
# Copyright (c) 2021-2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

# The ema model update and the domain-mixing are based on:
# https://github.com/vikolss/DACS
# Copyright (c) 2020 vikolss. Licensed under the MIT License.
# A copy of the license is available at resources/license_dacs

import math
import os
import random
from copy import deepcopy

import mmcv
import numpy as np
import torch
from matplotlib import pyplot as plt
from timm.models.layers import DropPath
from torch.nn.modules.dropout import _DropoutNd
import torch.optim as optim
import torch.nn.functional as F

from mmseg.core import add_prefix
from mmseg.models import UDA, build_segmentor
from mmseg.models.uda.uda_decorator import UDADecorator, get_module
from mmseg.models.utils.dacs_transforms import (denorm, get_class_masks,
                                                get_mean_std, strong_transform,
                                                lbl_retain, weight_retain)
from mmseg.models.utils.visualization import subplotimg
from mmseg.utils.utils import downscale_label_ratio

from mmseg.models.uda.clsnet.network.resnet38_cls import ClsNet
from mmseg.models.uda.discriminator import PixelDiscriminator, ImageDiscriminator

def _params_equal(ema_model, model):
    for ema_param, param in zip(ema_model.named_parameters(),
                                model.named_parameters()):
        if not torch.equal(ema_param[1].data, param[1].data):
            # print("Difference in", ema_param[0])
            return False
    return True

def calc_grad_magnitude(grads, norm_type=2.0):
    norm_type = float(norm_type)
    if norm_type == math.inf:
        norm = max(p.abs().max() for p in grads)
    else:
        norm = torch.norm(
            torch.stack([torch.norm(p, norm_type) for p in grads]), norm_type)
    return norm

def lr_poly(base_lr, iter, max_iters, power):
    return base_lr * ((1 - float(iter) / max_iters) ** (power))

def get_one_hot(label, N):
    b,_,h,w = label.shape
    label = torch.where(label==255, N, label)
    label = label.squeeze(1).view(-1)
    ones = torch.sparse.torch.eye(N)
    ones = torch.cat((ones, torch.zeros(1, N)), dim=0).cuda()
    ones = ones.index_select(0, label)
    return ones.view(b, h, w, N).permute(0, 3, 1, 2)

def get_one_hot_cls(label, N):
    assert label.dim() == 2
    b,c = label.shape
    label_one_hot = [None] * b
    ones = torch.sparse.torch.eye(N).cuda()
    for i in range(b):
        label_one_hot[i] = (label[i].unsqueeze(-1)*ones).unsqueeze(0)
    # (B,C,C)
    label_one_hot = torch.cat(label_one_hot)
    return label_one_hot

@UDA.register_module()
class MultiCA(UDADecorator):

    def __init__(self, **cfg):
        super(MultiCA, self).__init__(**cfg)
        self.local_iter = 0
        self.max_iters = cfg['max_iters']
        # Pseudo label configs
        self.alpha = cfg['alpha']
        self.pseudo_threshold = cfg['pseudo_threshold']
        self.psweight_ignore_top = cfg['pseudo_weight_ignore_top']
        self.psweight_ignore_bottom = cfg['pseudo_weight_ignore_bottom']
        # Feature distance configs
        self.fdist_lambda = cfg['imnet_feature_dist_lambda']
        self.fdist_classes = cfg['imnet_feature_dist_classes']
        self.fdist_scale_min_ratio = cfg['imnet_feature_dist_scale_min_ratio']
        self.enable_fdist = self.fdist_lambda > 0
        # Mix configs
        self.mix = cfg['mix']
        self.blur = cfg['blur']
        self.color_jitter_s = cfg['color_jitter_strength']
        self.color_jitter_p = cfg['color_jitter_probability']
        # Debug configs
        self.debug_img_interval = cfg['debug_img_interval']
        self.print_grad_magnitude = cfg['print_grad_magnitude']

        assert self.mix == 'class'

        self.debug_fdist_mask = None
        self.debug_gt_rescale = None

        self.class_probs = {}
        ema_cfg = deepcopy(cfg['model'])
        self.ema_model = build_segmentor(ema_cfg)

        if self.enable_fdist:
            self.imnet_model = build_segmentor(deepcopy(cfg['model']))
        else:
            self.imnet_model = None
            
        # MutiCA configs
        dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_classes = 19
        self.power = cfg['power']

        self.src_lbl = 0
        self.tgt_lbl = 1
        
        # Pixel level discriminator
        self.enable_px_d_feat = cfg['enable_px_d_feat']
        self.lr_px_d = cfg['lr_px_d']
        if self.enable_px_d_feat:
            self.px_d_feat_model = PixelDiscriminator().to(dev)
            self.px_d_feat_optim = optim.Adam(self.px_d_feat_model.parameters(), lr=self.lr_px_d, betas=(0.9, 0.99))
            self.px_adv_feat_lambda = cfg['px_adv_feat_lambda']
        self.enable_px_d_out = cfg['enable_px_d_out']
        if self.enable_px_d_out:
            self.px_d_out_model = PixelDiscriminator(input_nc=19).to(dev)
            self.px_d_out_optim = optim.Adam(self.px_d_out_model.parameters(), lr=self.lr_px_d, betas=(0.9, 0.99))
            self.px_adv_out_lambda = cfg['px_adv_out_lambda']

        # Image level discriminator
        self.enable_img_d_feat = cfg['enable_img_d_feat']
        self.lr_img_d = cfg['lr_img_d']
        if self.enable_img_d_feat:
            self.img_d_feat_model = ImageDiscriminator().to(dev)
            self.img_d_feat_optim = optim.Adam(self.img_d_feat_model.parameters(), lr=self.lr_img_d, betas=(0.9, 0.99))
            self.img_adv_feat_lambda = cfg['img_adv_feat_lambda']
        self.enable_img_d_out = cfg['enable_img_d_out']
        if self.enable_img_d_out:
            self.img_d_out_model = ImageDiscriminator(input_nc=19).to(dev)
            self.img_d_out_optim = optim.Adam(self.img_d_out_model.parameters(), lr=self.lr_img_d, betas=(0.9, 0.99))
            self.img_adv_out_lambda = cfg['px_adv_out_lambda']

        # Image Classifier
        self.enable_cls = cfg['enable_cls']
        if self.enable_cls:
            self.cls_model = ClsNet().to(dev)
            self.cls_model.load_state_dict(torch.load(cfg['cls_pretrained']))
            self.cls_thred = cfg['cls_thred']
            self.cls_model.eval()

    def get_ema_model(self):
        return get_module(self.ema_model)
    
    def get_imnet_model(self):
        return get_module(self.imnet_model)
    
    def _init_ema_weights(self):
        for param in self.get_ema_model().parameters():
            param.detach_()
        mp = list(self.get_model().parameters())
        mcp = list(self.get_ema_model().parameters())
        for i in range(0, len(mp)):
            if not mcp[i].data.shape:  # scalar tensor
                mcp[i].data = mp[i].data.clone()
            else:
                mcp[i].data[:] = mp[i].data[:].clone()

    def _update_ema(self, iter):
        alpha_teacher = min(1 - 1 / (iter + 1), self.alpha)
        for ema_param, param in zip(self.get_ema_model().parameters(),
                                    self.get_model().parameters()):
            if not param.data.shape:  # scalar tensor
                ema_param.data = \
                    alpha_teacher * ema_param.data + \
                    (1 - alpha_teacher) * param.data
            else:
                ema_param.data[:] = \
                    alpha_teacher * ema_param[:].data[:] + \
                    (1 - alpha_teacher) * param[:].data[:]
                
    def masked_feat_dist(self, f1, f2, mask=None):
        feat_diff = f1 - f2
        # mmcv.print_log(f'fdiff: {feat_diff.shape}', 'mmseg')
        pw_feat_dist = torch.norm(feat_diff, dim=1, p=2)
        # mmcv.print_log(f'pw_fdist: {pw_feat_dist.shape}', 'mmseg')
        if mask is not None:
            # mmcv.print_log(f'fd mask: {mask.shape}', 'mmseg')
            pw_feat_dist = pw_feat_dist[mask.squeeze(1)]
            # mmcv.print_log(f'fd masked: {pw_feat_dist.shape}', 'mmseg')
        return torch.mean(pw_feat_dist)
    
    def calc_feat_dist(self, img, gt, feat=None):
        assert self.enable_fdist
        with torch.no_grad():
            self.get_imnet_model().eval()
            feat_imnet = self.get_imnet_model().extract_feat(img)
            feat_imnet = [f.detach() for f in feat_imnet]
        lay = -1
        if self.fdist_classes is not None:
            fdclasses = torch.tensor(self.fdist_classes, device=gt.device)
            scale_factor = gt.shape[-1] // feat[lay].shape[-1]
            gt_rescaled = downscale_label_ratio(gt, scale_factor,
                                                self.fdist_scale_min_ratio,
                                                self.num_classes,
                                                255).long().detach()
            fdist_mask = torch.any(gt_rescaled[..., None] == fdclasses, -1)
            feat_dist = self.masked_feat_dist(feat[lay], feat_imnet[lay],
                                              fdist_mask)
            self.debug_fdist_mask = fdist_mask
            self.debug_gt_rescale = gt_rescaled
        else:
            feat_dist = self.masked_feat_dist(feat[lay], feat_imnet[lay])
        feat_dist = self.fdist_lambda * feat_dist
        feat_loss, feat_log = self._parse_losses(
            {'loss_imnet_feat_dist': feat_dist})
        feat_log.pop('loss', None)
        return feat_loss, feat_log
    
    def adjust_learning_rate_d(self, base_lr, optimizer):
        lr = lr_poly(base_lr, self.local_iter, self.max_iters, self.power)
        optimizer.param_groups[0]['lr'] = lr
        if len(optimizer.param_groups) > 1:
            optimizer.param_groups[1]['lr'] = lr * 10

    def train_step(self, data_batch, optimizer, **kwargs):
        # The iteration step during training.
        if self.local_iter == 0:
            if self.enable_px_d_feat:
                self.px_d_feat_model.train()
            if self.enable_px_d_out:
                self.px_d_out_model.train()
            if self.enable_img_d_feat:
                self.img_d_feat_model.train()
            if self.enable_img_d_out:
                self.img_d_out_model.train()

        # Discriminator
        if self.enable_px_d_feat:
            self.px_d_feat_optim.zero_grad()
            self.adjust_learning_rate_d(self.lr_px_d, self.px_d_feat_optim)
        if self.enable_px_d_out:
            self.px_d_out_optim.zero_grad()
            self.adjust_learning_rate_d(self.lr_px_d, self.px_d_out_optim)
        if self.enable_img_d_feat:
            self.img_d_feat_optim.zero_grad()
            self.adjust_learning_rate_d(self.lr_img_d, self.img_d_feat_optim)
        if self.enable_img_d_out:
            self.img_d_out_optim.zero_grad()
            self.adjust_learning_rate_d(self.lr_img_d, self.img_d_out_optim)

        # Segmentor
        optimizer.zero_grad()

        log_vars = self(**data_batch)

        if self.enable_px_d_feat:
            self.px_d_feat_optim.step()
        if self.enable_px_d_out:
            self.px_d_out_optim.step()
        if self.enable_img_d_feat:
            self.img_d_feat_optim.step()
        if self.enable_img_d_out:
            self.img_d_out_optim.step()

        optimizer.step()

        log_vars.pop('loss', None)  # remove the unnecessary 'loss'
        outputs = dict(
            log_vars=log_vars, num_samples=len(data_batch['img_metas']))
        return outputs
        
    def forward_train(self, img, img_metas, gt_semantic_seg, target_img,
                      target_img_metas):
        """Forward function for training.
        Args:
            img (Tensor): Input images.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.
        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        log_vars = {}
        batch_size = img.shape[0]
        dev = img.device
        
        # Init/update ema model
        if self.local_iter == 0:
            self._init_ema_weights()
            # assert _params_equal(self.get_ema_model(), self.get_model())
        if self.local_iter > 0:
            self._update_ema(self.local_iter)
            # assert not _params_equal(self.get_ema_model(), self.get_model())
            # assert self.get_ema_model().training

        means, stds = get_mean_std(img_metas, dev)
        strong_parameters = {
            'mix': None,
            'color_jitter': random.uniform(0, 1),
            'color_jitter_s': self.color_jitter_s,
            'color_jitter_p': self.color_jitter_p,
            'blur': random.uniform(0, 1) if self.blur else 0,
            'mean': means[0].unsqueeze(0),  # assume same normalization
            'std': stds[0].unsqueeze(0)
        }

        with torch.no_grad():
            # Original seg label (B,1,H,W)
            # One-hot seg label (B,19,H,W)
            src_seg_gt = get_one_hot(gt_semantic_seg, self.num_classes)
            b, c, _, _ = src_seg_gt.shape 

        # Train G
        # Don't accumulate grads in D
        if self.enable_px_d_feat:
            for param in self.px_d_feat_model.parameters():
                param.requires_grad = False
        if self.enable_px_d_out:
            for param in self.px_d_out_model.parameters():
                param.requires_grad = False
        if self.enable_img_d_feat:
            for param in self.img_d_feat_model.parameters():
                param.requires_grad = False
        if self.enable_img_d_out:
            for param in self.img_d_out_model.parameters():
                param.requires_grad = False

        # Train on source images
        clean_losses = self.get_model().forward_train(
            img, img_metas, gt_semantic_seg, return_feat=True)
        src_feat = clean_losses.pop('features') # [2, 512, 16, 16]
        clean_losses = add_prefix(clean_losses, 'src')
        clean_loss, clean_log_vars = self._parse_losses(clean_losses)
        log_vars.update(clean_log_vars)
        # Source Segmentation loss
        clean_loss.backward(retain_graph=self.enable_fdist)
        
        if self.print_grad_magnitude:
            params = self.get_model().backbone.parameters()
            seg_grads = [
                p.grad.detach().clone() for p in params if p.grad is not None
            ]
            grad_mag = calc_grad_magnitude(seg_grads)
            mmcv.print_log(f'Seg. Grad.: {grad_mag}', 'mmseg')

        # ImageNet feature distance
        if self.enable_fdist:
            feat_loss, feat_log = self.calc_feat_dist(img, gt_semantic_seg,
                                                      src_feat)
            feat_loss.backward(retain_graph=True)
            log_vars.update(add_prefix(feat_log, 'src'))
            if self.print_grad_magnitude:
                params = self.get_model().backbone.parameters()
                fd_grads = [
                    p.grad.detach() for p in params if p.grad is not None
                ]
                fd_grads = [g2 - g1 for g1, g2 in zip(seg_grads, fd_grads)]
                grad_mag = calc_grad_magnitude(fd_grads)
                mmcv.print_log(f'Fdist Grad.: {grad_mag}', 'mmseg')

        for m in self.get_ema_model().modules():
            if isinstance(m, _DropoutNd):
                m.training = False
            if isinstance(m, DropPath):
                m.training = False

        # Generate pseudo-label
        ema_logits = self.get_ema_model().encode_decode(
            target_img, target_img_metas)
        ema_softmax = torch.softmax(ema_logits.detach(), dim=1)

        # (B, H, W)
        pseudo_prob, pseudo_label = torch.max(ema_softmax, dim=1)
        ps_large_p = pseudo_prob.ge(self.pseudo_threshold).long() == 1
        ps_size = np.size(np.array(pseudo_label.cpu()))
        pseudo_weight = torch.sum(ps_large_p).item() / ps_size
        pseudo_weight = pseudo_weight * torch.ones(
            pseudo_prob.shape, device=dev)
        
        if self.psweight_ignore_top > 0:
            # Don't trust pseudo-labels in regions with potential
            # rectification artifacts. This can lead to a pseudo-label
            # drift from sky towards building or traffic light.
            pseudo_weight[:, :self.psweight_ignore_top, :] = 0
        if self.psweight_ignore_bottom > 0:
            pseudo_weight[:, -self.psweight_ignore_bottom:, :] = 0
        # (B,H,W)
        gt_pixel_weight = torch.ones((pseudo_weight.shape), device=dev)
        # Apply mixing
        mixed_img, mixed_lbl = [None] * batch_size, [None] * batch_size
        mix_masks = get_class_masks(gt_semantic_seg)
        
        src_lbl_retain, tgt_lbl_retain = [None] * batch_size, [None] * batch_size

        for i in range(batch_size):
            strong_parameters['mix'] = mix_masks[i]
            mixed_img[i], mixed_lbl[i] = strong_transform(
                strong_parameters,
                data=torch.stack((img[i], target_img[i])),
                target=torch.stack((gt_semantic_seg[i][0], pseudo_label[i])))
            _, pseudo_weight[i] = strong_transform(
                strong_parameters,
                target=torch.stack((gt_pixel_weight[i], pseudo_weight[i])))
            # Calculate retained label of source and target
            # (1,1,H,W)
            src_lbl_retain[i], tgt_lbl_retain[i] = lbl_retain(
                strong_parameters['mix'],
                torch.stack((gt_semantic_seg[i][0], pseudo_label[i]))
            )

        mixed_img = torch.cat(mixed_img)
        mixed_lbl = torch.cat(mixed_lbl)
        # (B,1,H,W)
        src_lbl_retain = torch.cat(src_lbl_retain)
        tgt_lbl_retain = torch.cat(tgt_lbl_retain)
        # Segmentation retained label -> (B,C,H,W)
        src_lbl_retain = get_one_hot(src_lbl_retain, self.num_classes).detach()
        tgt_lbl_retain = get_one_hot(tgt_lbl_retain, self.num_classes).detach()
        mix_lbl_onehot = get_one_hot(mixed_lbl, self.num_classes).detach()
        # Classification retained label -> (B,C)
        src_sum = src_lbl_retain.view(b,c,-1).sum(dim=2).float()
        tgt_sum = tgt_lbl_retain.view(b,c,-1).sum(dim=2).float()
        # (B,C)
        src_cls_retain = (src_sum > 0).float().nan_to_num(nan=0.0, posinf=0.0, neginf=0.0).detach()
        tgt_cls_retain = (tgt_sum > 0).float().nan_to_num(nan=0.0, posinf=0.0, neginf=0.0).detach()
        mix_cls_lbl = (src_cls_retain + tgt_cls_retain).detach()
        # (B,C,C)
        src_cls_retain = get_one_hot_cls(src_cls_retain, src_cls_retain.shape[1]).detach()
        tgt_cls_retain = get_one_hot_cls(tgt_cls_retain, tgt_cls_retain.shape[1]).detach()
        # (B,C,H,W)
        mix_lbl_soft = mix_lbl_onehot * pseudo_weight.unsqueeze(1)  
        # (B,C)
        mix_sum = mix_lbl_soft.view(b,c,-1).sum(dim=2).float() 
        mix_cnt = mix_lbl_onehot.view(b,c,-1).sum(dim=2).float()
        cls_weights = (mix_sum / mix_cnt).float().nan_to_num(nan=0.0, posinf=0.0, neginf=0.0).detach()

        with torch.no_grad():
            # y_19 -- multi-hot cls prediction (B, 19)
            # _, _, y_19 = self.cls_model(mixed_img)
            if self.enable_cls:
                mix_cam_19, _ = self.cls_model.forward_cam(mixed_img)
                mix_cam_19 = torch.nn.ReLU(inplace=True)(mix_cam_19)
                mix_cam_19 = mix_cam_19 / torch.max(mix_cam_19)
            # mask = (y_19 > self.cls_thred).float()
            # mix_cls_pred = y_19 * mask  # (B, 19)
            # mix_cam_19 = mix_cam_19 * mask.unsqueeze(-1).unsqueeze(-1)  # (B, 19, H, W)
        
        # Train on mixed images
        mix_losses = self.get_model().forward_train(
            mixed_img, img_metas, mixed_lbl, pseudo_weight, return_feat=True, return_fusefeat=True, return_out=True)
        mix_feat = mix_losses.pop('features')
        mix_fusefeat = mix_losses.pop('fusefeatures') # [2, 256, 128, 128]
        mix_out = mix_losses.pop('out') # [2, 19, 128, 12]
        mix_out = F.log_softmax(mix_out, dim=1)
        
        mix_losses = add_prefix(mix_losses, 'mix')
        mix_loss, mix_log_vars = self._parse_losses(mix_losses)
        log_vars.update(mix_log_vars)
        mix_loss.backward(retain_graph=True)
        
        if self.enable_px_d_feat:
            px_adv_losses = self.px_d_feat_model.forward(
                mix_fusefeat, torch.cat((src_lbl_retain, tgt_lbl_retain), dim=1), 
                weights=pseudo_weight, return_inv=True)
            
            px_adv_losses = add_prefix(px_adv_losses, 'adv.mix.feat')
            px_adv_loss, px_adv_log_vars = self._parse_losses(px_adv_losses)
            log_vars.update(px_adv_log_vars)
            px_adv_loss = self.px_adv_feat_lambda * (px_adv_loss / 2)
            px_adv_loss.backward(retain_graph=True)
        
        if self.enable_px_d_out:
            px_adv_losses = self.px_d_out_model.forward(
                mix_out, torch.cat((src_lbl_retain, tgt_lbl_retain), dim=1), 
                weights=pseudo_weight, return_inv=True)
            
            px_adv_losses = add_prefix(px_adv_losses, 'adv.mix.out')
            px_adv_loss, px_adv_log_vars = self._parse_losses(px_adv_losses)
            log_vars.update(px_adv_log_vars)
            px_adv_loss = self.px_adv_out_lambda * (px_adv_loss / 2)
            px_adv_loss.backward(retain_graph=True)
        
        if self.enable_img_d_feat:
            img_adv_losses = self.img_d_feat_model.forward(
                mix_fusefeat, mix_cam_19, mix_cls_lbl,
                torch.cat((src_cls_retain, tgt_cls_retain), dim=2), # (B,C,2*C)
                weights=cls_weights, return_inv=True
            )

            img_adv_losses = add_prefix(img_adv_losses, 'adv.mix.feat')
            img_adv_loss, img_adv_log_vars = self._parse_losses(img_adv_losses)
            log_vars.update(img_adv_log_vars)
            img_adv_loss = self.img_adv_feat_lambda * (img_adv_loss / 2)
            img_adv_loss.backward(retain_graph=True)
        
        if self.enable_img_d_out:
            img_adv_losses = self.img_d_out_model.forward(
                mix_out, mix_cam_19, mix_cls_lbl,
                torch.cat((src_cls_retain, tgt_cls_retain), dim=2), # (B,C,2*C)
                weights=cls_weights, return_inv=True
            )

            img_adv_losses = add_prefix(img_adv_losses, 'adv.mix.out')
            img_adv_loss, img_adv_log_vars = self._parse_losses(img_adv_losses)
            log_vars.update(img_adv_log_vars)
            img_adv_loss = self.img_adv_out_lambda * (img_adv_loss / 2)
            img_adv_loss.backward()
        
        # Train D
        # Bring back requires_grad
        if self.enable_px_d_feat:
            for param in self.px_d_feat_model.parameters():
                param.requires_grad = True
        if self.enable_px_d_out:
            for param in self.px_d_out_model.parameters():
                param.requires_grad = True
        if self.enable_img_d_feat:
            for param in self.img_d_feat_model.parameters():
                param.requires_grad = True   
        if self.enable_img_d_out:
            for param in self.img_d_out_model.parameters():
                param.requires_grad = True  

        # Train on mix images
        # Block gradients back to the segmentation network
        mix_fusefeat = mix_fusefeat.detach()
        mix_out = mix_out.detach()

        if self.enable_px_d_feat:
            px_losses = self.px_d_feat_model.forward(
                mix_fusefeat, torch.cat((src_lbl_retain, tgt_lbl_retain), dim=1),
                weights=pseudo_weight
            )
            px_loss, px_log_vars = self._parse_losses(px_losses)
            px_loss.backward(retain_graph=True)

        if self.enable_px_d_out:
            px_losses = self.px_d_out_model.forward(
                mix_out, torch.cat((src_lbl_retain, tgt_lbl_retain), dim=1),
                weights=pseudo_weight
            )
            px_loss, px_log_vars = self._parse_losses(px_losses)
            px_loss.backward()

        if self.enable_img_d_feat:
            img_losses = self.img_d_feat_model.forward(
                mix_fusefeat, mix_cam_19, mix_cls_lbl,
                torch.cat((src_cls_retain, tgt_cls_retain), dim=2),
                weights=cls_weights
            )
            img_loss, img_log_vars = self._parse_losses(img_losses)
            img_loss.backward(retain_graph=True)

        if self.enable_img_d_out:
            img_losses = self.img_d_out_model.forward(
                mix_out, mix_cam_19, mix_cls_lbl,
                torch.cat((src_cls_retain, tgt_cls_retain), dim=2),
                weights=cls_weights
            )
            img_loss, img_log_vars = self._parse_losses(img_losses)
            img_loss = img_loss
            img_loss.backward()

        if self.local_iter % self.debug_img_interval == 0:
            out_dir = os.path.join(self.train_cfg['work_dir'],
                                   'class_mix_debug')
            os.makedirs(out_dir, exist_ok=True)
            vis_img = torch.clamp(denorm(img, means, stds), 0, 1)
            vis_trg_img = torch.clamp(denorm(target_img, means, stds), 0, 1)
            vis_mixed_img = torch.clamp(denorm(mixed_img, means, stds), 0, 1)
            for j in range(batch_size):
                rows, cols = 2, 5
                fig, axs = plt.subplots(
                    rows,
                    cols,
                    figsize=(3 * cols, 3 * rows),
                    gridspec_kw={
                        'hspace': 0.1,
                        'wspace': 0,
                        'top': 0.95,
                        'bottom': 0,
                        'right': 1,
                        'left': 0
                    },
                )
                subplotimg(axs[0][0], vis_img[j], 'Source Image')
                subplotimg(axs[1][0], vis_trg_img[j], 'Target Image')
                subplotimg(
                    axs[0][1],
                    gt_semantic_seg[j],
                    'Source Seg GT',
                    cmap='cityscapes')
                subplotimg(
                    axs[1][1],
                    pseudo_label[j],
                    'Target Seg (Pseudo) GT',
                    cmap='cityscapes')
                subplotimg(axs[0][2], vis_mixed_img[j], 'Mixed Image')
                subplotimg(
                    axs[1][2], mix_masks[j][0], 'Domain Mask', cmap='gray')
                # subplotimg(axs[0][3], pred_u_s[j], "Seg Pred",
                #            cmap="cityscapes")
                subplotimg(
                    axs[1][3], mixed_lbl[j], 'Seg Targ', cmap='cityscapes')
                subplotimg(
                    axs[0][3], pseudo_weight[j], 'Pseudo W.', vmin=0, vmax=1)
                if self.debug_fdist_mask is not None:
                    subplotimg(
                        axs[0][4],
                        self.debug_fdist_mask[j][0],
                        'FDist Mask',
                        cmap='gray')
                if self.debug_gt_rescale is not None:
                    subplotimg(
                        axs[1][4],
                        self.debug_gt_rescale[j],
                        'Scaled GT',
                        cmap='cityscapes')
                for ax in axs.flat:
                    ax.axis('off')
                plt.savefig(
                    os.path.join(out_dir,
                                 f'{(self.local_iter + 1):06d}_{j}.png'))
                plt.close()
        self.local_iter += 1

        return log_vars
    