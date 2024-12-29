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
import torch.nn as nn
import torch.nn.functional as F

from mmseg.core import add_prefix
from mmseg.models import UDA, build_segmentor
from mmseg.models.uda.uda_decorator import UDADecorator, get_module
from mmseg.models.utils.dacs_transforms import (denorm, get_class_masks,
                                                get_mean_std, strong_transform)
from mmseg.models.utils.visualization import subplotimg
from mmseg.utils.utils import downscale_label_ratio

from mmseg.models.uda.clsnet.network.resnet38_cls import ClsNet
from mmseg.models.uda.discriminator import FCDiscriminatorWoCls, PixelDiscriminator, ImageDiscriminator

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
	# print(label)
	label = torch.where(label==255, N, label)
	label = label.squeeze(1).view(-1)
	# print(label)
	ones = torch.sparse.torch.eye(N)
	ones = torch.cat((ones, torch.zeros(1, N)), dim=0).cuda()
	# print(ones)
	ones = ones.index_select(0, label)
	# print(ones)
	return ones.view(b, h, w, N).permute(0, 3, 1, 2)

@UDA.register_module()
class GAN(UDADecorator):

    def __init__(self, **cfg):
        super(GAN, self).__init__(**cfg)
        self.local_iter = 0
        self.max_iters = cfg['max_iters']
        self.fdist_lambda = cfg['imnet_feature_dist_lambda']
        self.fdist_classes = cfg['imnet_feature_dist_classes']
        self.fdist_scale_min_ratio = cfg['imnet_feature_dist_scale_min_ratio']
        self.enable_fdist = self.fdist_lambda > 0
        self.print_grad_magnitude = cfg['print_grad_magnitude']

        self.debug_fdist_mask = None
        self.debug_gt_rescale = None

        self.class_probs = {}

        if self.enable_fdist:
            self.imnet_model = build_segmentor(deepcopy(cfg['model']))
        else:
            self.imnet_model = None
        
        self.dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_classes = cfg['model']['decode_head']['num_classes']
        self.power = cfg['power']
        self.temperature = cfg['temperature']

        self.src_lbl = 0
        self.tgt_lbl = 1

        # Pixel level discriminator without class in output space
        self.enable_px_wo_cls_d = cfg['enable_px_wo_cls_d']
        if self.enable_px_wo_cls_d:
            self.px_wo_cls_d_model = FCDiscriminatorWoCls().to(self.dev)
            self.lr_px_wo_cls_d = cfg['lr_px_wo_cls_d']
            self.px_wo_cls_d_optim = optim.Adam(self.px_wo_cls_d_model.parameters(), lr=self.lr_px_wo_cls_d, betas=(0.9, 0.99))
            self.px_wo_cls_adv_lambda = cfg['px_wo_cls_adv_lambda']

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

    def get_imnet_model(self):
        return get_module(self.imnet_model)
    
    def adjust_learning_rate_d(self, base_lr, optimizer):
        lr = lr_poly(base_lr, self.local_iter, self.max_iters, self.power)
        optimizer.param_groups[0]['lr'] = lr
        if len(optimizer.param_groups) > 1:
            optimizer.param_groups[1]['lr'] = lr * 10

    def train_step(self, data_batch, optimizer, **kwargs):
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating is also defined in
        this method, such as GAN.

        Args:
            data (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``,
                ``num_samples``.
                ``loss`` is a tensor for back propagation, which can be a
                weighted sum of multiple losses.
                ``log_vars`` contains all the variables to be sent to the
                logger.
                ``num_samples`` indicates the batch size (when the model is
                DDP, it means the batch size on each GPU), which is used for
                averaging the logs.
        """

        if self.local_iter == 0:
            if self.enable_px_wo_cls_d:
                self.px_wo_cls_d_model.train()
            if self.enable_px_d:
                self.px_d_model.train()
            if self.enable_img_d:
                self.img_d_model.train()

        # Discriminator
        if self.enable_px_wo_cls_d:
            self.px_wo_cls_d_optim.zero_grad()
            self.adjust_learning_rate_d(self.lr_px_wo_cls_d, self.px_wo_cls_d_optim)
        if self.enable_px_d:
            self.px_d_optim.zero_grad()
            self.adjust_learning_rate_d(self.lr_px_d, self.px_d_optim)
        if self.enable_img_d:
            self.img_d_optim.zero_grad()
            self.adjust_learning_rate_d(self.lr_img_d , self.img_d_optim)

        # Segmentor
        optimizer.zero_grad()

        log_vars = self(**data_batch)

        optimizer.step()
        if self.enable_px_wo_cls_d:
            self.px_wo_cls_d_optim.step()
        if self.enable_px_d:
            self.px_d_optim.step()
        if self.enable_img_d:
            self.img_d_optim.step()

        log_vars.pop('loss', None)  # remove the unnecessary 'loss'
        outputs = dict(
            log_vars=log_vars, num_samples=len(data_batch['img_metas']))
        return outputs

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

        src_size = img.shape[-2:]
        tgt_size = target_img.shape[-2:]

        # one-hot seg label (B, 19, H, W)
        with torch.no_grad():
            src_seg_gt = get_one_hot(gt_semantic_seg, self.num_classes)

        # Cal CAM
        if self.enable_cls:
            with torch.no_grad():
                # y_19 -- multi-hot cls prediction (B, 19)
                _, _, y_19 = self.cls_model(target_img)
                tgt_cam_19, _ = self.cls_model.forward_cam(target_img)
                # tgt_cam_19 = F.upsample(tgt_cam_19_feat, size, mode='bilinear', align_corners=False)
                mask = (y_19 > self.cls_thred).float()
                tgt_cls_pred = y_19 * mask  # (B, 19)
                tgt_cam_19 = tgt_cam_19 * mask.unsqueeze(-1).unsqueeze(-1)  # (B, 19, H, W)

                b, c, _, _ = src_seg_gt.shape
                _, _, y_19 = self.cls_model(img)
                src_cam_19, _ = self.cls_model.forward_cam(img)
                # src_cam_19 = F.upsample(src_cam_19_feat, size, mode='bilinear', align_corners=False)
                src_cls_gt, _ = src_seg_gt.view(b, c, -1).max(dim=2)
                src_cls_gt = (src_cls_gt > 0).float()   # (B, 19)
                src_cam_19 = src_cam_19 * src_cls_gt.unsqueeze(-1).unsqueeze(-1)

        # Train G

        # Don't accumulate grads in D
        if self.enable_px_wo_cls_d:
            for param in self.px_wo_cls_d_model.parameters():
                param.requires_grad = False
        if self.enable_px_d:
            for param in self.px_d_model.parameters():
                param.requires_grad = False
        if self.enable_img_d:
            for param in self.img_d_model.parameters():
                param.requires_grad = False

        # Train on source images 
        clean_losses = self.get_model().forward_train_w_pred(
            img, img_metas, gt_semantic_seg, return_feat=True, return_pred=True)

        src_feat = clean_losses.pop('features')[-1]
        src_pred = clean_losses.pop('pred')
        # src_pred = torch.softmax(src_pred, dim=1)

        clean_losses = add_prefix(clean_losses, 'src')
        clean_loss, clean_log_vars = self._parse_losses(clean_losses)
        log_vars.update(clean_log_vars)
        # Segmentation loss
        clean_loss.backward(retain_graph=True)

        if self.print_grad_magnitude:
            params = self.get_model().backbone.parameters()
            seg_grads = [
                p.grad.detach().clone() for p in params if p.grad is not None
            ]
            grad_mag = calc_grad_magnitude(seg_grads)
            mmcv.print_log(f'Seg. Grad.: {grad_mag}', 'mmseg')

        # Adversarial
        if self.enable_px_wo_cls_d:
            px_wo_cls_adv_losses = self.px_wo_cls_d_model.forward_train(
                src_pred, self.tgt_lbl, return_inv=True)
    
            px_wo_cls_adv_losses = add_prefix(px_wo_cls_adv_losses, 'adv.src')
            px_wo_cls_adv_loss, px_wo_cls_adv_log_vars = self._parse_losses(px_wo_cls_adv_losses)
            log_vars.update(px_wo_cls_adv_log_vars)

            # 1/2 * 1/2(raw + inverse)
            px_wo_cls_adv_loss = self.px_wo_cls_adv_lambda * (px_wo_cls_adv_loss / 4)
            px_wo_cls_adv_loss.backward(retain_graph=True)

        if self.enable_px_d:
            px_adv_losses = self.px_d_model.forward(
                src_feat, torch.cat((src_seg_gt, torch.zeros_like(src_seg_gt)), dim=1), 
                return_inv=True)

            px_adv_losses = add_prefix(px_adv_losses, 'adv.src')
            px_adv_loss, px_adv_log_vars = self._parse_losses(px_adv_losses)
            log_vars.update(px_adv_log_vars)

            px_adv_loss = self.px_adv_lambda * (px_adv_loss / 4)
            px_adv_loss.backward(retain_graph=True)

        if self.enable_img_d:
            img_adv_losses = self.img_d_model.forward(
                src_feat, src_cam_19, src_cls_gt,
                torch.cat((src_cls_gt, torch.zeros_like(src_cls_gt)), dim=1),
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
            feat_loss.backward()
            log_vars.update(add_prefix(feat_log, 'src'))
            if self.print_grad_magnitude:
                params = self.get_model().backbone.parameters()
                fd_grads = [
                    p.grad.detach() for p in params if p.grad is not None
                ]
                fd_grads = [g2 - g1 for g1, g2 in zip(seg_grads, fd_grads)]
                grad_mag = calc_grad_magnitude(fd_grads)
                mmcv.print_log(f'Fdist Grad.: {grad_mag}', 'mmseg')

        # Train on target images
        tgt_pred, tgt_feat = self.get_model().encode_decode(
            target_img, target_img_metas, return_feat=True)
        tgt_feat = tgt_feat[-1]
        tgt_soft_lbl = F.softmax(tgt_pred.div(self.temperature), dim=1)
        tgt_soft_lbl[tgt_soft_lbl>0.9] = 0.9
        tgt_soft_lbl = tgt_soft_lbl.detach()

        # Adversarial
        if self.enable_px_wo_cls_d:
            px_wo_cls_adv_losses = self.px_wo_cls_d_model.forward_train(
                tgt_pred, self.src_lbl, return_inv=True)
    
            px_wo_cls_adv_losses = add_prefix(px_wo_cls_adv_losses, 'adv.tgt')
            px_wo_cls_adv_loss, px_wo_cls_adv_log_vars = self._parse_losses(px_wo_cls_adv_losses)
            log_vars.update(px_wo_cls_adv_log_vars)
     
            px_wo_cls_adv_loss = self.px_wo_cls_adv_lambda * (px_wo_cls_adv_loss / 4)
            px_wo_cls_adv_loss.backward(retain_graph=True)

        if self.enable_px_d:
            px_adv_losses = self.px_d_model.forward(
                tgt_feat, torch.cat((torch.zeros_like(tgt_soft_lbl), tgt_soft_lbl), dim=1), 
                return_inv=True)

            px_adv_losses = add_prefix(px_adv_losses, 'adv.tgt')
            px_adv_loss, px_adv_log_vars = self._parse_losses(px_adv_losses)
            log_vars.update(px_adv_log_vars)

            px_adv_loss = self.px_adv_lambda * (px_adv_loss / 4)
            px_adv_loss.backward(retain_graph=True)

        if self.enable_img_d:
            img_adv_losses = self.img_d_model.forward(
                tgt_feat, tgt_cam_19, tgt_cls_pred,
                torch.cat((torch.zeros_like(tgt_cls_pred), tgt_cls_pred), dim=1),
                return_inv=True
            )

            img_adv_losses = add_prefix(img_adv_losses, 'adv.tgt')
            img_adv_loss, img_adv_log_vars = self._parse_losses(img_adv_losses)
            log_vars.update(img_adv_log_vars)

            img_adv_loss = self.img_adv_lambda * (img_adv_loss / 4)
            img_adv_loss.backward()
        
        # Train D
        
        # Bring back requires_grad
        if self.enable_px_wo_cls_d:
            for param in self.px_wo_cls_d_model.parameters():
                param.requires_grad = True
        if self.enable_px_d:
            for param in self.px_d_model.parameters():
                param.requires_grad = True
        if self.enable_img_d:
            for param in self.img_d_model.parameters():
                param.requires_grad = True

        # Train on source images
        # Block gradients back to the segmentation network
        src_feat = src_feat.detach()    
        src_pred = src_pred.detach()

        if self.enable_px_wo_cls_d:
            px_wo_cls_losses = self.px_wo_cls_d_model.forward_train(
                src_pred, self.src_lbl)

            px_wo_cls_losses = add_prefix(px_wo_cls_losses, 'src')
            px_wo_cls_loss, px_wo_cls_log_vars = self._parse_losses(px_wo_cls_losses)
            log_vars.update(px_wo_cls_log_vars)

            px_wo_cls_loss = px_wo_cls_loss / 2
            px_wo_cls_loss.backward()

        if self.enable_px_d:
            px_losses = self.px_d_model.forward(
                src_feat, torch.cat((src_seg_gt, torch.zeros_like(src_seg_gt)), dim=1)
            )

            px_losses = add_prefix(px_losses, 'src')
            px_loss, px_log_vars = self._parse_losses(px_losses)
            log_vars.update(px_log_vars)

            px_loss = px_loss / 2
            px_loss.backward()

        if self.enable_img_d:
            img_losses = self.img_d_model.forward(
                src_feat, src_cam_19, src_cls_gt,
                torch.cat((src_cls_gt, torch.zeros_like(src_cls_gt)), dim=1)
            )

            img_losses = add_prefix(img_losses, 'src')
            img_loss, img_log_vars = self._parse_losses(img_losses)
            log_vars.update(img_log_vars)

            img_loss = img_loss / 2
            img_loss.backward()

        # Train on target images
        tgt_feat = tgt_feat.detach()
        tgt_pred = tgt_pred.detach()

        if self.enable_px_wo_cls_d:
            px_wo_cls_losses = self.px_wo_cls_d_model.forward_train(
                tgt_pred, self.tgt_lbl)

            px_wo_cls_losses = add_prefix(px_wo_cls_losses, 'tgt')
            px_wo_cls_loss, px_wo_cls_log_vars = self._parse_losses(px_wo_cls_losses)
            log_vars.update(px_wo_cls_log_vars)

            px_wo_cls_loss = px_wo_cls_loss / 2
            px_wo_cls_loss.backward()

        if self.enable_px_d:
            px_losses = self.px_d_model.forward(
                tgt_feat, torch.cat((torch.zeros_like(tgt_soft_lbl), tgt_soft_lbl), dim=1)
            )

            px_losses = add_prefix(px_losses, 'tgt')
            px_loss, px_log_vars = self._parse_losses(px_losses)
            log_vars.update(px_log_vars)

            px_loss = px_loss / 2
            px_loss.backward()

        if self.enable_img_d:
            img_losses = self.img_d_model.forward(
                tgt_feat, tgt_cam_19, tgt_cls_pred,
                torch.cat((torch.zeros_like(tgt_cls_pred), tgt_cls_pred), dim=1)
            )

            img_losses = add_prefix(img_losses, 'tgt')
            img_loss, img_log_vars = self._parse_losses(img_losses)
            log_vars.update(img_log_vars)

            img_loss = img_loss / 2
            img_loss.backward()

        self.local_iter += 1

        return log_vars
