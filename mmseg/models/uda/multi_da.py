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
from torchvision.utils import save_image

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
from mmseg.models.utils.fda import FDA_source_to_target

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
class MultiDA(UDADecorator):

    def __init__(self, **cfg):
        super(MultiDA, self).__init__(**cfg)
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

        self.dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_classes = cfg['model']['decode_head']['num_classes']
        self.power = cfg['power']

        self.src_lbl = 0
        self.tgt_lbl = 1

        self.enable_adver = False

        # TODO: Task level discriminator

        # Pixel level discriminator
        self.enable_px_d = cfg['enable_px_d']
        if self.enable_px_d:
            self.px_d_model = PixelDiscriminator().to(self.dev)
            self.lr_px_d = cfg['lr_px_d']
            self.px_d_optim = optim.Adam(self.px_d_model.parameters(), lr=self.lr_px_d, betas=(0.9, 0.99))
            self.px_adv_lambda = cfg['px_adv_lambda']

        # Image level discriminator
        self.enable_img_d = cfg['enable_img_d']
        if self.enable_img_d:
            self.img_d_model = ImageDiscriminator().to(self.dev)
            self.lr_img_d = cfg['lr_img_d']
            self.img_d_optim = optim.Adam(self.img_d_model.parameters(), lr=self.lr_img_d, betas=(0.9, 0.99))
            self.img_adv_lambda = cfg['img_adv_lambda']

        # Image Classifier
        self.enable_cls = cfg['enable_cls']
        if self.enable_cls:
            self.cls_model = ClsNet().to(self.dev)
            self.cls_model.load_state_dict(torch.load(cfg['cls_pretrained']))
            self.cls_thred = cfg['cls_thred']
            self.cls_model.eval()

        # Style transfer
        self.enable_fft = cfg['enable_fft']
        self.enable_style_gan = cfg['enable_style_gan']
        self.fft_beta = cfg['fft_beta']
        self.enable_src_in_tgt = cfg['enable_src_in_tgt']
        self.enable_tgt_in_src = cfg['enable_tgt_in_src']
        self.enable_src_in_tgt_b4_train = cfg['enable_src_in_tgt_b4_train']

        # Style consistency regularization
        self.enable_st_consistency = cfg['enable_st_consistency']
        self.st_consistency_lambda = cfg['st_consistency_lambda']

        # Normalize
        self.to_rgb = cfg['to_rgb']
        self.norm_cfg = cfg['norm_cfg']

        # Rare class mix
        self.enable_rcm = cfg['enable_rcm']
        if self.enable_rcm:
            self.rare_class_mix = cfg['rare_class_mix']

        # Category contrast
        self.enable_ctrst_feat = cfg['enable_ctrst_feat']
        if self.enable_ctrst_feat:
            self.protos_feat = torch.rand(self.num_classes, 256).cuda()
            self.ctrst_feat_lambda = cfg['ctrst_feat_lambda']

        self.enable_ctrst_out = cfg['enable_ctrst_out']
        if self.enable_ctrst_out:
            self.protos_out = torch.rand(self.num_classes, 19).cuda()
            self.ctrst_out_lambda = cfg['ctrst_out_lambda']

        if self.enable_ctrst_feat or self.enable_ctrst_out:
            # self.rare_class_id = cfg['rare_class_id']
            self.temperature = cfg['temperature']
            self.mix_proto_alpha = cfg['mix_proto_alpha']

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

    def adjust_temperature(self):
        self.temperature = 1.0 / (1.0 + math.log(self.local_iter))

    def init_norm_param(self, img_metas):
        for i in range(len(img_metas)):
            img_metas[i]['img_norm_cfg']['mean'] = np.array(self.norm_cfg['mean'], dtype=np.float32)
            img_metas[i]['img_norm_cfg']['std'] = np.array(self.norm_cfg['std'], dtype=np.float32)
            img_metas[i]['img_norm_cfg']['to_rgb'] = self.to_rgb
        return img_metas

    def normalize(self, img, mean, std):
        mean = torch.tensor(mean).reshape(1,3,1,1).cuda()
        std = torch.tensor(std).reshape(1,3,1,1).cuda()
        stdinv = 1.0 / std
        img = img - mean
        img = torch.mul(img, stdinv)    
        return img

    def calculate_cl_loss(self, feat_seg, feat_cls, cam_19, gt):
        # (B,512,H1,W1) (B,512,H2,W2) (B,19,H2,W2) (B,1,H,W)
        assert feat_seg.shape[1] == feat_cls.shape[1], feat_seg.shape[1]
        b, ch, _, _ = feat_seg.shape
        c = cam_19.shape[1]
        # gt = gt.clone().un
        # Positive samples
        # (b,19,512)
        Vsamples = torch.sum(cam_19.unsqueeze(2) * feat_cls.unsqueeze(1), dim=[-2,-1]) / torch.sum(cam_19, dim=[-2,-1]).view(b, c, 1)
        Vsamples = Vsamples.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0).cuda()
        # Downsample gt
        gt = downscale_label_ratio(gt,gt.shape[-1]//feat_seg.shape[-1],0.75,19)
        gt = gt.squeeze()   # (B,H,W)
        loss = []
        for i in range(b):
            classes = torch.unique(gt[i])
            rare_class_cnt=0
            cl_loss=0.0
            for ci in classes:
                ci_cl_loss = 0.0
                if ci in self.rare_class_id:
                    feat_seg_ci = feat_seg[i]  # (512,H1,W1)
                    mask_ci = (gt[i]==ci)
                    feat_seg_ci = feat_seg_ci[:, mask_ci]   # (512, n)
                    if feat_seg_ci.shape[1] <= 1:
                        continue
                    # Calculate vi * v+
                    vi_vplus = torch.zeros(feat_seg_ci.shape[1]).cuda() # (n)
                    vi_vplus = torch.cosine_similarity(Vsamples[i,ci,:].unsqueeze(0), feat_seg_ci.t(), dim=1)
                    vi_vplus = torch.exp(vi_vplus / self.temperature)
                    # Calculate vi * v-
                    v_negtive = Vsamples[i].clone() # (19, 512)
                    v_negtive[ci, :] = 0.0
                    vi_vsub = torch.zeros(feat_seg_ci.shape[1]).cuda()  # (n)
                    vi_vsub = torch.cosine_similarity(feat_seg_ci.t().unsqueeze(1), v_negtive.unsqueeze(0), dim=2)  # (n,19)
                    vi_vsub = torch.exp(vi_vsub / self.temperature)
                    vi_vsub = torch.sum(vi_vsub, dim=1)
                    ci_cl_loss = vi_vplus / (vi_vplus + vi_vsub)
                    ci_cl_loss = ci_cl_loss.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)
                    ci_cl_loss = -torch.log(ci_cl_loss)
                    ci_cl_loss = torch.mean(ci_cl_loss)

                    rare_class_cnt += 1
                cl_loss += ci_cl_loss
            if rare_class_cnt == 0:
                pass
            else:
                loss.append(cl_loss / rare_class_cnt)

        if len(loss) != 0:
            return sum(loss) / len(loss)
        else:
            return 0

    def calc_proto(self, label, feat, q=None):
        # label: (B, 1, H, W) / feat: (B, C, H', W') / q: (B, 1, H, W)

        # Downsample
        label = downscale_label_ratio(label, label.shape[-1]//feat.shape[-1], 0.75, 19)
        
        if q != None:
            q = F.avg_pool2d(q, kernel_size=q.shape[-1]//feat.shape[-1])
            q_protos = torch.zeros(self.num_classes, 1).cuda()

        b, c, _, _ = feat.shape
        protos = torch.zeros(self.num_classes, c).cuda()

        for i in range(self.num_classes):
            mask = (label == i) # (B,1,H,W)
            sz = torch.sum(mask, dim=[-2,-1]) # (B, 1)
            cnt = torch.sum(sz > 0)
            if cnt == 0:
                continue
            feat_i = feat * mask # (B, C, H, W)
            proto = torch.sum(feat_i, dim=[-2, -1]) / sz # (B, C)
            proto = proto.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)

            if q != None:
                q_i = q * mask
                q_i = torch.sum(q_i, dim=[-2, -1]) / sz # (B, 1)
                q_i = q_i.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)
                
                q_i_soft = torch.softmax(q_i, dim=0)
                proto = torch.sum(proto * q_i_soft, dim=0)

                q_i = torch.sum(q_i, dim=0) / cnt # (1)
                q_protos[i] = q_i
            else:
                proto = torch.sum(proto, dim=0) / cnt # (C)
            
            protos[i] = proto

        if q != None:
            return protos, q_protos

        return protos

    def calc_proto_mix(self, label, feat, q=None):
        # label: (B, 1, H, W) / feat: (B, C, H', W') / q: (B, 1, H, W)

        # Downsample
        label = downscale_label_ratio(label, label.shape[-1]//feat.shape[-1], 0.75, 19)
        if q != None:
            q = F.avg_pool2d(q, kernel_size=q.shape[-1]//feat.shape[-1])

        b, c, _, _ = feat.shape
        protos = torch.rand(self.num_classes, c).cuda()

        feat = feat.permute(1, 0, 2, 3) # (C, B, H, W)

        for i in range(self.num_classes):
            mask = (label == i) # (B,1,H,W)
            sz = torch.sum(mask, dim=[-2,-1]) # (B, 1)
            cnt = torch.sum(sz > 0)
            if cnt == 0:
                continue
            feat_i = feat * mask # (B, C, H, W)
            if q != None:
                feat_i = feat_i * q
            proto = torch.sum(feat_i, dim=[-2, -1]) / sz # (B, C)
            proto = proto.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)
            proto = torch.sum(proto, dim=0) / cnt # (C)
            protos[i] = proto

        return protos
        
    def calc_cl_loss_feat(self, feat, label, q=None):
        # protos: (19, C) / feat: (B, C, H', W') / label: (B, 1, H, W) / q: (B, 1, H, W)

        assert self.protos_feat.dim() == 2
        assert self.protos_feat.shape[1] == feat.shape[1]
        b, c, _, _ = feat.shape

        # Downsample
        label = downscale_label_ratio(label, label.shape[-1]//feat.shape[-1], 0.75, 19)
        if q != None:
            q = F.avg_pool2d(q, kernel_size=q.shape[-1]//feat.shape[-1])
            q = q.squeeze(1) # (B,H,W)

        feat = feat.permute(1, 0, 2, 3) # (C, B, H, W)
        # feat = feat.view(c, -1) # (C, B*H*W)
        
        losses = []

        for i in range(self.num_classes):
            mask = (label == i) # (B,1,H,W)
            mask = mask.squeeze(1) # (B, H, W)

            v = feat[:, mask] # (C, N)
            v = v.permute(1, 0) # (N, C)
            n = v.shape[0] 
            if n == 0:
                continue
            # v_vplus = v.mm(protos[i].unsqueeze(1)) # (n,C)*(C,1)=(n,1)
            v_vplus = torch.cosine_similarity(v, self.protos_feat[i].unsqueeze(0), dim=1) # (n,C) (n,1) => (n)
            v_vplus = torch.exp(v_vplus / self.temperature)
            # v_vsub = v.mm(protos.permute(1, 0)) # (n, 19)
            v_vsub = torch.cosine_similarity(v.unsqueeze(1), self.protos_feat.unsqueeze(0), dim=2) # (n,1,C) (1,19,C) => (n,19)
            v_vsub =  torch.exp(v_vsub / self.temperature)
            v_vsub = torch.sum(v_vsub, dim=1) # (n)
            loss = v_vplus / v_vsub # (n)
            loss = -torch.log(loss)
                
            if q != None:
                q_i = q[mask] # (n)
                loss = loss * q_i

            loss = torch.mean(loss)
            losses.append(loss)

        if len(losses) == 0:
            return 0
        return sum(losses) / len(losses)

    def calc_cl_loss_out(self, feat, label, q=None):
        # protos: (19, C) / feat: (B, C, H', W') / label: (B, 1, H, W) / q: (B, 1, H, W)

        assert self.protos_out.dim() == 2
        assert self.protos_out.shape[1] == feat.shape[1]
        b, c, _, _ = feat.shape

        # Downsample
        label = downscale_label_ratio(label, label.shape[-1]//feat.shape[-1], 0.75, 19)
        if q != None:
            q = F.avg_pool2d(q, kernel_size=q.shape[-1]//feat.shape[-1])
            q = q.squeeze(1) # (B,H,W)

        feat = feat.permute(1, 0, 2, 3) # (C, B, H, W)
        # feat = feat.view(c, -1) # (C, B*H*W)
        
        losses = []

        for i in range(self.num_classes):
            mask = (label == i) # (B,1,H,W)
            mask = mask.squeeze(1) # (B, H, W)

            v = feat[:, mask] # (C, N)
            v = v.permute(1, 0) # (N, C)
            n = v.shape[0] 
            if n == 0:
                continue
            # v_vplus = v.mm(protos[i].unsqueeze(1)) # (n,C)*(C,1)=(n,1)
            v_vplus = torch.cosine_similarity(v, self.protos_out[i].unsqueeze(0), dim=1) # (n,C) (n,1) => (n)
            v_vplus = torch.exp(v_vplus / self.temperature)
            # v_vsub = v.mm(protos.permute(1, 0)) # (n, 19)
            v_vsub = torch.cosine_similarity(v.unsqueeze(1), self.protos_out.unsqueeze(0), dim=2) # (n,1,C) (1,19,C) => (n,19)
            v_vsub =  torch.exp(v_vsub / self.temperature)
            v_vsub = torch.sum(v_vsub, dim=1) # (n)
            loss = v_vplus / v_vsub # (n)
            loss = -torch.log(loss)
                
            if q != None:
                q_i = q[mask] # (n)
                loss = loss * q_i

            loss = torch.mean(loss)
            losses.append(loss)

        if len(losses) == 0:
            return 0
        return sum(losses) / len(losses)

    def train_step(self, data_batch, optimizer, **kwargs):
        # The iteration step during training.
        if self.local_iter == 0:
            if self.enable_px_d:
                self.px_d_model.train()
            if self.enable_img_d:
                self.img_d_model.train()

        # Simulated annealing
        # if self.enable_ctrst:
        #     if self.local_iter >= 1:
        #         self.adjust_temperature()

        # Discriminator
        if self.enable_px_d:
            self.px_d_optim.zero_grad()
            self.adjust_learning_rate_d(self.lr_px_d, self.px_d_optim)
        if self.enable_img_d:
            self.img_d_optim.zero_grad()
            self.adjust_learning_rate_d(self.lr_img_d , self.img_d_optim)
        # Segmentor
        optimizer.zero_grad()

        log_vars = self(**data_batch)

        if self.enable_px_d:
            self.px_d_optim.step()
        if self.enable_img_d:
            self.img_d_optim.step()
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

        # FFT and Normalize
        if self.enable_fft:
            if self.to_rgb:
                img = img.flip(dims=[1])
                target_img = target_img.flip(dims=[1])

            img_metas = self.init_norm_param(img_metas)
            target_img_metas = self.init_norm_param(target_img_metas)

            # for i in range(batch_size):
            #     save_image(img[i] / 255.0, "/root/autodl-tmp/DAFormer/demo/a_src_" + str(i) + ".png")
            #     save_image(target_img[i] / 255.0, "/root/autodl-tmp/DAFormer/demo/a_tgt_" + str(i) + ".png")

            if self.enable_src_in_tgt_b4_train:
                img = FDA_source_to_target(img, target_img, L=self.fft_beta).cuda()

            # for i in range(batch_size):
            #     save_image(img[i] / 255.0, "/root/autodl-tmp/DAFormer/demo/a_src_in_tgt_" + str(i) + ".png")

            # assert 1==0
            
            if self.enable_src_in_tgt:
                src_img_in_tgt = FDA_source_to_target(img, target_img, L=self.fft_beta).cuda()
            if self.enable_tgt_in_src:
                tgt_img_in_src = FDA_source_to_target(target_img, img, L=self.fft_beta).cuda()
            
            # for i in range(batch_size):
            #     save_image(img[i] / 255.0, "/root/autodl-tmp/DAFormer/demo/src_before_norm" + str(i) + "_.png")
            #     save_image(target_img[i] / 255.0, "/root/autodl-tmp/DAFormer/demo/tgt_before_norm" + str(i) + "_.png")
            #     save_image(tgt_img_in_src[i] / 255.0, "/root/autodl-tmp/DAFormer/demo/tgt_in_src_before_norm" + str(i) + "_.png")

            img = self.normalize(img, **self.norm_cfg)
            target_img = self.normalize(target_img, **self.norm_cfg)
            if self.enable_src_in_tgt:
                src_img_in_tgt = self.normalize(src_img_in_tgt, **self.norm_cfg)
            if self.enable_tgt_in_src:
                tgt_img_in_src = self.normalize(tgt_img_in_src, **self.norm_cfg)

            # for i in range(batch_size):
            #     save_image(img[i], "/root/autodl-tmp/DAFormer/demo/src_in_tgt_after_norm" + str(i) + "_.png")
            #     save_image(target_img[i], "/root/autodl-tmp/DAFormer/demo/tgt_after_norm" + str(i) + "_.png")
            #     save_image(tgt_img_in_src[i], "/root/autodl-tmp/DAFormer/demo/tgt_in_src_after_norm" + str(i) + "_.png")

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

            # Calculate CAM
            if self.enable_cls:
                # _, _, y_19 = self.cls_model(img)
                src_cam_19, src_feat_cls = self.cls_model.forward_cam(img)
                # src_cam_19 = F.upsample(src_cam_19_feat, size, mode='bilinear', align_corners=False)
                src_cls_gt, _ = src_seg_gt.view(b, c, -1).max(dim=2)
                src_cls_gt = (src_cls_gt > 0).float()   # (B, 19)
                src_cls_gt_one_hot = get_one_hot_cls(src_cls_gt, self.num_classes)  # (B,C,C)
                # src_cam_19 = src_cam_19 * src_cls_gt.unsqueeze(-1).unsqueeze(-1)
                # Ignore unseen class
                src_cam_19_masked = src_cam_19.clone()
                for i in range(batch_size):
                    classes = torch.unique(gt_semantic_seg[i])
                    classes = classes[classes!=255]
                    masks = torch.zeros(19).cuda()
                    masks[classes] = 1
                    src_cam_19_masked[i] = src_cam_19_masked[i] * masks.view(c, 1, 1)
                # Normalize
                for i in range(batch_size):
                    max_val, _ = src_cam_19_masked[i].view(c, -1).max(dim=1)
                    max_val = torch.where(max_val==0, 1.0, max_val.double())
                    src_cam_19_masked[i] = src_cam_19_masked[i] / max_val.view(c, 1, 1)                    

        # Train G

        # Don't accumulate grads in D
        if self.enable_px_d:
            for param in self.px_d_model.parameters():
                param.requires_grad = False
        if self.enable_img_d:
            for param in self.img_d_model.parameters():
                param.requires_grad = False

        # Train on source images
        clean_losses = self.get_model().forward_train(
            img, img_metas, gt_semantic_seg, return_feat=True, return_fusefeat=True, return_out=True)
        src_feat = clean_losses.pop('features')
        src_fusefeat = clean_losses.pop('fusefeatures')
        src_out = clean_losses.pop('out')

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

        # Adversarial
        if self.enable_px_d:
            px_adv_losses = self.px_d_model.forward(
                src_feat[-1], torch.cat((src_seg_gt, torch.zeros_like(src_seg_gt)), dim=1), 
                return_inv=True)

            px_adv_losses = add_prefix(px_adv_losses, 'adv.src')
            px_adv_loss, px_adv_log_vars = self._parse_losses(px_adv_losses)
            log_vars.update(px_adv_log_vars)

            px_adv_loss = self.px_adv_lambda * (px_adv_loss / 4)
            px_adv_loss.backward(retain_graph=True)

        if self.enable_img_d:
            img_adv_losses = self.img_d_model.forward(
                src_feat[-1], src_cam_19, src_cls_gt,
                torch.cat((src_cls_gt_one_hot, torch.zeros_like(src_cls_gt_one_hot)), dim=2),
                return_inv=True
            )
             
            img_adv_losses = add_prefix(img_adv_losses, 'adv.src')
            img_adv_loss, img_adv_log_vars = self._parse_losses(img_adv_losses)
            log_vars.update(img_adv_log_vars)

            img_adv_loss = self.img_adv_lambda * (img_adv_loss / 4)
            img_adv_loss.backward(retain_graph=True)

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
        ema_logits, tgt_fusefeat_te, tgt_out_te = self.get_ema_model().encode_decode(
            target_img, target_img_metas, return_fusefeat=True, return_out=True)

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
        if self.enable_rcm:
            mix_masks = get_class_masks(gt_semantic_seg, self.rare_class_mix)
        else:
            mix_masks = get_class_masks(gt_semantic_seg)
        src_lbl_retain, tgt_lbl_retain = [None] * batch_size, [None] * batch_size
        src_weight_retain, tgt_weight_retain = [None] * batch_size, [None] * batch_size

        for i in range(batch_size):
            strong_parameters['mix'] = mix_masks[i]
            mixed_img[i], mixed_lbl[i] = strong_transform(
                strong_parameters,
                data=torch.stack((img[i], target_img[i])),
                target=torch.stack((gt_semantic_seg[i][0], pseudo_label[i])))
            _, pseudo_weight[i] = strong_transform(
                strong_parameters,
                target=torch.stack((gt_pixel_weight[i], pseudo_weight[i])))
            if self.enable_adver:
                # Calculate retained label of source and target
                # (1,1,H,W)
                src_lbl_retain[i], tgt_lbl_retain[i] = lbl_retain(
                    strong_parameters['mix'],
                    torch.stack((gt_semantic_seg[i][0], pseudo_label[i]))
                )
                src_weight_retain, tgt_weight_retain = weight_retain(
                    strong_parameters['mix'],
                    torch.stack((gt_pixel_weight[i], pseudo_weight[i]))
                )
        mixed_img = torch.cat(mixed_img)
        mixed_lbl = torch.cat(mixed_lbl)
        if self.enable_adver:
            # (B,1,H,W)
            src_lbl_retain = torch.cat(src_lbl_retain)
            tgt_lbl_retain = torch.cat(tgt_lbl_retain)
            # (B,C,H,W)
            src_lbl_retain = get_one_hot(src_lbl_retain, self.num_classes)
            tgt_lbl_retain = get_one_hot(tgt_lbl_retain, self.num_classes)
            mix_lbl_onehot = get_one_hot(mixed_lbl, self.num_classes)
            # Segmentation retained label -> (B,C,H,W)
            src_lbl_retain = (src_lbl_retain * src_weight_retain).detach()
            tgt_lbl_retain = (tgt_lbl_retain * tgt_weight_retain).detach()
            mix_lbl_onehot = mix_lbl_onehot * pseudo_weight.unsqueeze(1)
            # Classification retained label -> (B,C,C)
            src_sum = src_lbl_retain.view(b,c,-1).sum(dim=2).float()
            tgt_sum = tgt_lbl_retain.view(b,c,-1).sum(dim=2).float()
            mix_sum = mix_lbl_onehot.view(b,c,-1).sum(dim=2).float()
            # (B,C)
            src_cls_retain = (src_sum / mix_sum).nan_to_num(nan=0.0, posinf=0.0, neginf=0.0).detach()
            tgt_cls_retain = (tgt_sum / mix_sum).nan_to_num(nan=0.0, posinf=0.0, neginf=0.0).detach()
            mix_cls_lbl = src_cls_retain + tgt_cls_retain
            # (B,C,C)
            src_cls_retain = get_one_hot_cls(src_cls_retain, src_cls_retain.shape[1]).detach()
            tgt_cls_retain = get_one_hot_cls(tgt_cls_retain, tgt_cls_retain.shape[1]).detach()

        with torch.no_grad():
            # y_19 -- multi-hot cls prediction (B, 19)
            # _, _, y_19 = self.cls_model(mixed_img)
            if self.enable_cls:
                mix_cam_19, _ = self.cls_model.forward_cam(mixed_img)
            # mask = (y_19 > self.cls_thred).float()
            # mix_cls_pred = y_19 * mask  # (B, 19)
            # mix_cam_19 = mix_cam_19 * mask.unsqueeze(-1).unsqueeze(-1)  # (B, 19, H, W)

        # Update protos
        mix_losses_te = self.get_ema_model().forward_train(mixed_img, img_metas, mixed_lbl, pseudo_weight, return_fusefeat=True, return_out=True)
        mix_fusefeat_te = mix_losses_te.pop('fusefeatures')
        mix_out_te = mix_losses_te.pop('out')

        clean_losses_te = self.get_ema_model().forward_train(
            img, img_metas, gt_semantic_seg, return_fusefeat=True, return_out=True)
        src_fusefeat_te = clean_losses_te.pop('fusefeatures')
        src_out_te = clean_losses_te.pop('out')

        # if self.enable_ctrst_feat:
        #     with torch.no_grad():
        #         protos_feat_s = self.calc_proto(gt_semantic_seg, src_fusefeat_te.clone())
        #         protos_feat_t, q_protos_feat_t = self.calc_proto(pseudo_label.unsqueeze(1), tgt_fusefeat_te.clone(), pseudo_weight.unsqueeze(1))
        #         mix_proto_alpha = self.mix_proto_alpha * q_protos_feat_t # (19, 1)
        #         protos_cur = (1 - mix_proto_alpha) * protos_feat_s + mix_proto_alpha * protos_feat_t
        #         proto_alpha = min(1 - 1 / (self.local_iter + 1), self.alpha)
        #         self.protos_feat = proto_alpha * self.protos_feat + (1 - proto_alpha) * protos_cur
        #         self.protos_feat = self.protos_feat.detach()

        # if self.enable_ctrst_out:
        #     with torch.no_grad():
        #         protos_out_s = self.calc_proto(gt_semantic_seg, src_out_te.clone())
        #         protos_out_t, q_protos_out_t = self.calc_proto(pseudo_label.unsqueeze(1), tgt_out_te.clone(), pseudo_weight.unsqueeze(1))
        #         mix_proto_alpha = self.mix_proto_alpha * q_protos_out_t # (19, 1)
        #         protos_cur = (1 - mix_proto_alpha) * protos_out_s + mix_proto_alpha * protos_out_t
        #         proto_alpha = min(1 - 1 / (self.local_iter + 1), self.alpha)
        #         self.protos_out = proto_alpha * self.protos_out + (1 - proto_alpha) * protos_cur
        #         self.protos_out = self.protos_out.detach()

        # # Category contrast for source
        # if self.enable_ctrst_feat:            
        #     src_cl_loss = self.calc_cl_loss_feat(src_fusefeat.clone(), gt_semantic_seg)
        #     log_vars['src.cl_loss_feat'] = 0.0
        #     if src_cl_loss > 0:
        #         log_vars['src.cl_loss_feat'] = src_cl_loss.item()
        #         src_cl_loss = self.ctrst_feat_lambda * src_cl_loss
        #         src_cl_loss.backward(retain_graph=True)

        # if self.enable_ctrst_out:
        #     src_cl_loss = self.calc_cl_loss_out(src_out.clone(), gt_semantic_seg)
        #     log_vars['src.cl_loss_out'] = 0.0
        #     if src_cl_loss > 0:
        #         log_vars['src.cl_loss_out'] = src_cl_loss.item()
        #         src_cl_loss = self.ctrst_out_lambda * src_cl_loss
        #         src_cl_loss.backward(retain_graph=True)

        if self.enable_ctrst_feat:
            with torch.no_grad():
                # protos_feat_s = self.calc_proto(gt_semantic_seg, src_fusefeat_te.clone())
                protos_feat_m, q_protos_feat_m = self.calc_proto(mixed_lbl, mix_fusefeat_te.clone(), pseudo_weight.unsqueeze(1))
                # Fuse source and mix protos
                # mix_alpha = self.mix_proto_alpha * q_protos_feat_m # (19, 1)
                # protos_feat = (1 - mix_alpha) * protos_feat_s + mix_alpha * protos_feat_m # (19, C)
                protos_feat = protos_feat_m # (19, C)
                # Exclude class not existing
                mask = torch.sum(protos_feat, dim=1)==0
                protos_feat[mask, :] = self.protos_feat[mask, :]
                # EMA update
                proto_alpha = min(1 - 1 / (self.local_iter + 1), self.alpha)
                proto_alpha = 1 - (1 - proto_alpha) * q_protos_feat_m
                self.protos_feat = proto_alpha * self.protos_feat + (1 - proto_alpha) * protos_feat
                self.protos_feat = self.protos_feat.detach()
        
        if self.enable_ctrst_out:
            with torch.no_grad():
                protos_out_s = self.calc_proto(gt_semantic_seg, src_out_te.clone())
                protos_out_m, q_protos_out_m = self.calc_proto(mixed_lbl, mix_out_te.clone(), pseudo_weight.unsqueeze(1))
                # Fuse source and mix protos
                # mix_alpha = self.mix_proto_alpha * q_protos_out_m # (19, 1)
                # protos_out = (1 - mix_alpha) * protos_out_s + mix_alpha * protos_out_m # (19, C)
                protos_out = protos_out_m # (19, C)
                # Exclude class not existing
                mask = torch.sum(protos_out, dim=1)==0
                protos_out[mask, :] = self.protos_out[mask, :]
                # EMA update
                proto_alpha = min(1 - 1 / (self.local_iter + 1), self.alpha)
                proto_alpha = 1 - (1 - proto_alpha) * q_protos_out_m
                self.protos_out = proto_alpha * self.protos_out + (1 - proto_alpha) * protos_out
                self.protos_out = self.protos_out.detach()

        # Train on mixed images
        mix_losses = self.get_model().forward_train(
            mixed_img, img_metas, mixed_lbl, pseudo_weight, return_feat=True, return_fusefeat=True, return_out=True)
        mix_feat = mix_losses.pop('features')
        mix_fusefeat = mix_losses.pop('fusefeatures')
        mix_out = mix_losses.pop('out')

        mix_losses = add_prefix(mix_losses, 'mix')
        mix_loss, mix_log_vars = self._parse_losses(mix_losses)
        log_vars.update(mix_log_vars)
        mix_loss.backward(retain_graph=True)

        # Category contrast for mix
        if self.enable_ctrst_feat:
            mix_cl_loss = self.calc_cl_loss_feat(mix_fusefeat.clone(), mixed_lbl, pseudo_weight.unsqueeze(1))
            log_vars['mix.cl_loss_feat'] = 0.0
            if mix_cl_loss > 0:
                log_vars['mix.cl_loss_feat'] = mix_cl_loss.item()
                mix_cl_loss = self.ctrst_feat_lambda * mix_cl_loss
                mix_cl_loss.backward(retain_graph=True)
        
        if self.enable_ctrst_out:
            mix_cl_loss = self.calc_cl_loss_out(mix_out.clone(), mixed_lbl, pseudo_weight.unsqueeze(1))
            log_vars['mix.cl_loss_out'] = 0.0
            if mix_cl_loss > 0:
                log_vars['mix.cl_loss_out'] = mix_cl_loss.item()
                mix_cl_loss = self.ctrst_out_lambda * mix_cl_loss
                mix_cl_loss.backward(retain_graph=True)

        if self.enable_px_d:
            px_adv_losses = self.px_d_model.forward(
                mix_feat[-1], torch.cat((src_lbl_retain, tgt_lbl_retain), dim=1), 
                return_inv=True)

            px_adv_losses = add_prefix(px_adv_losses, 'adv.mix')
            px_adv_loss, px_adv_log_vars = self._parse_losses(px_adv_losses)
            log_vars.update(px_adv_log_vars)

            px_adv_loss = self.px_adv_lambda * (px_adv_loss / 4)
            px_adv_loss.backward(retain_graph=True)

        if self.enable_img_d:
            img_adv_losses = self.img_d_model.forward(
                mix_feat[-1], mix_cam_19, mix_cls_lbl,
                torch.cat((src_cls_retain, tgt_cls_retain), dim=2), # (B,C,2*C)
                return_inv=True
            )

            img_adv_losses = add_prefix(img_adv_losses, 'adv.mix')
            img_adv_loss, img_adv_log_vars = self._parse_losses(img_adv_losses)
            log_vars.update(img_adv_log_vars)

            img_adv_loss = self.img_adv_lambda * (img_adv_loss / 4)
            img_adv_loss.backward()

        if self.enable_adver:
            # Train D

            # Bring back requires_grad
            if self.enable_px_d:
                for param in self.px_d_model.parameters():
                    param.requires_grad = True
            if self.enable_img_d:
                for param in self.img_d_model.parameters():
                    param.requires_grad = True        

            # Train on source images
            # Block gradients back to the segmentation network
            src_feat = src_feat[-1].detach()    

            if self.enable_px_d:
                px_losses = self.px_d_model.forward(
                    src_feat, torch.cat((src_seg_gt, torch.zeros_like(src_seg_gt)), dim=1)
                )

                px_losses = add_prefix(px_losses, 'src')
                px_loss, px_log_vars = self._parse_losses(px_losses)
                log_vars.update(px_log_vars)

                px_loss = px_loss / 2
                px_loss.backward(retain_graph=True)

            if self.enable_img_d:
                img_losses = self.img_d_model.forward(
                    src_feat, src_cam_19, src_cls_gt,
                    torch.cat((src_cls_gt_one_hot, torch.zeros_like(src_cls_gt_one_hot)), dim=2)
                )

                img_losses = add_prefix(img_losses, 'src')
                img_loss, img_log_vars = self._parse_losses(img_losses)
                log_vars.update(img_log_vars)

                img_loss = img_loss / 2
                img_loss.backward()

            # Train on mix images
            mix_feat = mix_feat[-1].detach()

            if self.enable_px_d:
                px_losses = self.px_d_model.forward(
                    mix_feat, torch.cat((src_lbl_retain, tgt_lbl_retain), dim=1)
                )

                px_losses = add_prefix(px_losses, 'mix')
                px_loss, px_log_vars = self._parse_losses(px_losses)
                log_vars.update(px_log_vars)

                px_loss = px_loss / 2
                px_loss.backward(retain_graph=True)

            if self.enable_img_d:
                img_losses = self.img_d_model.forward(
                    mix_feat, mix_cam_19, mix_cls_lbl,
                    torch.cat((src_cls_retain, tgt_cls_retain), dim=2)
                )

                img_losses = add_prefix(img_losses, 'mix')
                img_loss, img_log_vars = self._parse_losses(img_losses)
                log_vars.update(img_log_vars)

                img_loss = img_loss / 2
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
