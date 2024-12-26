import copy
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.core import add_prefix
from mmseg.ops import resize
from .. import builder
from ..builder import SEGMENTORS
from .base import BaseSegmentor, SemiBaseSegmentor
from ..losses import accuracy
from torch.distributions.uniform import Uniform
from torchvision.transforms import Resize
#导入分类网络
from .clsnet.network.resnet38_cls import ClsNet
import torchsnooper
@SEGMENTORS.register_module()
class ConstractiveLearning(SemiBaseSegmentor):
    def __init__(
        self,
        backbone, #using resnet18
        decode_head, #using pspNet...
        train_cfg=None,
        test_cfg=None,
        pretrained=None,
        weight_re_labeled=1.0,
        weight_re_unlabeled=1.0,
        weight_re_strong=1.0,
        temperature=1.0,
    ):
        super(ConstractiveLearning,self).__init__()
        #载入backbone
        self.backbone = builder.build_backbone(backbone)
        #初始化decode_head
        self._init_decode_head(decode_head)
        #载入训练配置
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        #初始化权重
        self.init_weights(pretrained=pretrained)
        self.is_enable_cl=True
        #初始化分类网络
        self.clsnet=ClsNet()
        #载入分类网络权重
        print('The weight of ClsNet has load!')
        self.clsnet.load_state_dict(torch.load('/home/b502/workspace_zhangbo/IFR-main/IFR/checkpoint/ep50.pth'))
        self.clsnet.eval()
        #稀疏类的train id
        self.rare_class_id=[4,5,6,7,11,12,13,14,15,16,17,18]
        self.temperature=0.5
        #添加历史的负样本
        self.class_wise_negative_samples=dict()
    #初始化权重
    def init_weights(self, pretrained=None):
        super(ConstractiveLearning, self).init_weights(pretrained)
        #nn.init.constant_(self.class_wise_merge.weight, 1)
        self.backbone.init_weights(pretrained=pretrained)
        self.decode_head.init_weights()
    def encode_decode(self, img, img_metas=None):
        x = self.backbone(img)
        out = self.decode_head(x)
        out = resize(input=out, size=img.shape[2:], mode='bilinear', align_corners=self.align_corners)
        return out
    def _init_decode_head(self, decode_head):
        """Initialize ``decode_head``"""
        self.decode_head = builder.build_head(decode_head)
        self.align_corners = self.decode_head.align_corners
        self.num_classes = self.decode_head.num_classes
    def forward(self, return_loss=True, img_metas=None, **kwargs):
        if return_loss:
            return self.forward_train(**kwargs)
        else:
            return self.forward_test(img_metas=img_metas, **kwargs)
    def downscale_label_ratio(self,gt,
                            scale_factor,
                            min_ratio,
                            n_classes,
                            ignore_index=255):
        assert scale_factor > 1
        bs, orig_c, orig_h, orig_w = gt.shape
        assert orig_c == 1
        trg_h, trg_w = orig_h // scale_factor, orig_w // scale_factor
        ignore_substitute = n_classes

        out = gt.clone()  # otw. next line would modify original gt
        out[out == ignore_index] = ignore_substitute
        out = F.one_hot(
            out.squeeze(1), num_classes=n_classes + 1).permute(0, 3, 1, 2)
        assert list(out.shape) == [bs, n_classes + 1, orig_h, orig_w], out.shape
        out = F.avg_pool2d(out.float(), kernel_size=scale_factor)
        gt_ratio, out = torch.max(out, dim=1, keepdim=True)
        out[out == ignore_substitute] = ignore_index
        out[gt_ratio < min_ratio] = ignore_index
        assert list(out.shape) == [bs, 1, trg_h, trg_w], out.shape
        return out
    #fea_from_segNet和fea_from_clsNet尺寸不一致，但这不影响，
    #@torchsnooper.snoop()
    def calculate_cl_loss(self,fea_from_segNet,fea_from_clsNet,cam_19,gt):
        #这个要求来自分割网络提取的特征图和来自分类网络提取的特征图的通道数是一致的
        #辅助计时工具
        # overall_start = torch.cuda.Event(enable_timing=True)
        # overall_end = torch.cuda.Event(enable_timing=True)
        # step_start = torch.cuda.Event(enable_timing=True)
        # step_end = torch.cuda.Event(enable_timing=True)
        # overall_start.record()

        assert fea_from_clsNet.shape[1]==fea_from_segNet.shape[1]
        gt=gt.clone()
        gt=gt.unsqueeze(1)
        #生成正样本
        Vsamples=torch.zeros((2,19,fea_from_clsNet.shape[1]))   # (2,19,512)
        for i in range(fea_from_clsNet.shape[0]):
            for c in range(cam_19.shape[1]):
                if torch.max(cam_19[i,c,:,:])!=0:
                    # (512)
                    Vsamples[i,c,:]=torch.sum(cam_19[i,c,:,:]*fea_from_clsNet[i,:,:,:],dim=[1,2])/torch.sum(cam_19[i,c,:,:])
        #计算对比损失
        ##在此之前需要下采样gt gt的维度应该是3 B*H*W  
        ##注意，关于GroudTruth的下采样需要重写
        gt=self.downscale_label_ratio(gt,gt.shape[-1]//fea_from_segNet.shape[-1],0.75,19)
        gt=gt.squeeze()
        batch_loss=[]
        #groudTruth=F.interpolate(gt.contiguous(),size=fea_from_segNet.shape[2:],mode='bilinear',align_corners=self.align_corners)
        ##遍历BatchSize维度
        for i in range(fea_from_segNet.shape[0]):
            classes=torch.unique(gt[i])
            rare_class_cnt=0
            cl_loss=0.0
            for ci in classes:
                ci_cl_loss=0.0
                #如果类别是稀疏类
                if ci in self.rare_class_id:
                    fea_from_segNet_ci=fea_from_segNet[i]
                    mask_ci=(gt[i]==ci)
                    fea_from_segNet_ci=fea_from_segNet_ci[:,mask_ci]    # (512,n)
                    if fea_from_segNet_ci.shape[1]<=1:
                        continue
                    #512*420
                    ##计算vi*v+
                    vi_vplus=torch.zeros(fea_from_segNet_ci.shape[1])
                    vi_vplus=torch.cosine_similarity(Vsamples[i,ci,:].cuda().unsqueeze(0),fea_from_segNet_ci.t(),dim=1)
                    vi_vplus=torch.exp(vi_vplus/self.temperature)
                    # for k in range(fea_from_segNet_ci.shape[1]):
                    #     vi_vplus[k]=torch.cosine_similarity(Vsamples[i,ci,:].cuda(), fea_from_segNet_ci[:,k], dim=0)
                    # vi_vplus=torch.exp(vi_vplus.squeeze())
                    ##计算vi*v- 这一步的耗时很高，打算采用pytorch的广播机制
                    #step_start.record()
                    v_negtive=Vsamples[i].clone()
                    v_negtive[ci,:]=0.0
                    v_negtive=v_negtive.cuda()
                    vi_vsub=torch.zeros(fea_from_segNet_ci.shape[1]).cuda()
                    for k in range(fea_from_segNet_ci.shape[1]):
                        kk=torch.cosine_similarity(fea_from_segNet_ci[:,k].unsqueeze(0),v_negtive,dim=1)
                        kk=torch.exp(kk/self.temperature)
                        vi_vsub[k]=torch.sum(kk)
                    # for k in range(fea_from_segNet_ci.shape[1]):
                    #     for kk in range(v_negtive.shape[0]):
                    #         vi_vsub[k]=vi_vsub[k]+torch.exp(torch.cosine_similarity(v_negtive[kk,:],fea_from_segNet_ci[:,k],dim=0))
                    # step_end.record()
                    # torch.cuda.synchronize()
                    # print('calculate vi_vsub time consume:',step_start.elapsed_time(step_end))
                    ##计算对比损失
                    tt=vi_vplus+vi_vsub
                    temp_cl=vi_vplus/tt
                    # temp_cl=torch.zeros(fea_from_segNet_ci.shape[1]).cuda()
                    # for k in range(fea_from_segNet_ci.shape[1]):
                    #     temp_cl[k]=vi_vplus[k]/(vi_vplus[k]+vi_vsub[k])
                    temp_cl=-torch.log(temp_cl)
                    ci_cl_loss=torch.mean(temp_cl)
                    rare_class_cnt=rare_class_cnt+1
                cl_loss=cl_loss+ci_cl_loss
            if rare_class_cnt==0:
                pass
                #batch_loss.append(0)
            else:
                batch_loss.append(cl_loss/rare_class_cnt)
        # overall_end.record()
        # torch.cuda.synchronize()
        # print('cl_loss time consume:',overall_start.elapsed_time(overall_end))
        if len(batch_loss)!=0:
            all_loss=0.0
            for b in batch_loss:
                all_loss=all_loss+b
            return all_loss/len(batch_loss)
        else:
            return 0
    #@torchsnooper.snoop()
    def forward_train(self, img_v0_0, img_v0_1, img_v0_1_s, img_v1_0, img_v1_1, img_v1_1_s, gt,cls_img_v0_0,iter):
        #首先获取已标注的图片的shape
        n, c, h, w = img_v0_1.shape
        #Ground Truth 压缩一个维度 压缩的是通道维度
        gt = gt.squeeze(1)
        #新建loss
        losses = dict()
        # supervised loss
        feats_v0_0 = self.backbone(img_v0_0)
        #backbone传入的特征输入到decoder...
        feats_v0_0 = self.decode_head(feats_v0_0, return_feat=True)
        logits_v0_0 = self.decode_head.cls_seg(feats_v0_0)
        #双线性插值恢复到原图大小...
        logits_v0_0 = F.interpolate(logits_v0_0, size=gt.shape[1:], mode='bilinear', align_corners=self.align_corners)
        loss = dict()
        #直接计算标签和预测图的交叉熵loss
        loss['loss_seg'] = self.decode_head.loss_decode(logits_v0_0, gt, ignore_index=255)
        #计算标签和预测图的准确度...
        loss['acc_seg'] = accuracy(logits_v0_0, gt)
        #向losses字典中添加元素
        losses.update(add_prefix(loss, 'decode'))
        ### constractive loss
        ###首先需要分类网络生成类别先验
        ###分类网络推理
        with torch.no_grad():
        #推理分类结果
            #x_19,fea,y_19=self.clsnet(cls_img_v0_0)
            #推理之后生成类别激活图
            cam_19,fea_conv4=self.clsnet.forward_cam(cls_img_v0_0)
        #隔离掉没有出现过的类
        for i in range(gt.shape[0]):
            batch_classes = torch.unique(gt[i].clone().detach())
            label19=torch.zeros(19).cuda()
            for j in range(19):
                if j in batch_classes:
                    label19[j]=1
            cam_19[i]=cam_19[i]*label19.view(19, 1, 1)
        #生成mask
        ##在生成mask之前首先需要resize
        #cam_19=F.interpolate(cam_19,feats_v0_0.shape[2:],mode='bilinear',align_corners=self.align_corners)
        ##归一化
        for i in range(gt.shape[0]):
            for j in range(cam_19.shape[1]):
                if torch.max(cam_19[i,j,:,:])!=0:
                    cam_19[i,j,:,:]=cam_19[i,j,:,:]/torch.max(cam_19[i,j,:,:])
        ##到这一步cam_19就是一个归一化的score map
        #计算对比损失
        cl_loss_supervised=self.calculate_cl_loss(feats_v0_0.clone(),fea_conv4,cam_19,gt)

        if cl_loss_supervised!=0:
            loss['cl_loss_supervised'] = cl_loss_supervised * 0.5
        
        #将对比损失用于无标注的数据
        # feats_v0_1 = self.backbone(img_v0_1)
        # #backbone传入的特征输入到decoder...
        # feats_v0_1 = self.decode_head(feats_v0_1, return_feat=True)
        # logits_v0_1 = self.decode_head.cls_seg(feats_v0_1)
        # #双线性插值恢复到原图大小...
        # logits_v0_1 = F.interpolate(logits_v0_1, size=gt.shape[1:], mode='bilinear', align_corners=self.align_corners)
        # pseudo_label = torch.argmax(logits_v0_1, dim=1)
        # ###分类网络推理
        # with torch.no_grad():
        # #推理分类结果
        #     #x_19,fea,y_19=self.clsnet(img_v0_1)
        #     #推理之后生成类别激活图
        #     cam_19,fea_conv4=self.clsnet.forward_cam(img_v0_1)
        # # #隔离掉没有出现过的类
        # for i in range(gt.shape[0]):
        #     batch_classes = torch.unique(pseudo_label[i].clone())
        #     label19=torch.zeros(19).cuda()
        #     for j in range(19):
        #         if j in batch_classes:
        #             label19[j]=1
        #     cam_19[i]=cam_19[i]*label19.view(19, 1, 1)
        # # ##归一化
        # for i in range(gt.shape[0]):
        #     for c in range(cam_19.shape[1]):
        #         if torch.max(cam_19[i,j,:,:])!=0:
        #             cam_19[i,j,:,:]=cam_19[i,j,:,:]/torch.max(cam_19[i,j,:,:])
        # cl_loss_unsupervised=self.calculate_cl_loss(feats_v0_1.clone(),fea_conv4,cam_19,pseudo_label)
        # if cl_loss_unsupervised!=0:
        #      loss['cl_loss_unsupervised'] = cl_loss_unsupervised * 0.05
        losses.update(add_prefix(loss, 'decode'))
        return losses
    # TODO refactor
    def slide_inference(self, img, img_meta, rescale):
        """Inference by sliding-window with overlap.

        If h_crop > h_img or w_crop > w_img, the small patch will be used to
        decode without padding.
        """

        h_stride, w_stride = self.test_cfg.stride
        h_crop, w_crop = self.test_cfg.crop_size
        batch_size, _, h_img, w_img = img.size()
        num_classes = self.num_classes
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = img.new_zeros((batch_size, num_classes, h_img, w_img))
        count_mat = img.new_zeros((batch_size, 1, h_img, w_img))
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = img[:, :, y1:y2, x1:x2]
                crop_seg_logit = self.encode_decode(crop_img, img_meta)
                preds += F.pad(crop_seg_logit, (int(x1), int(preds.shape[3] - x2), int(y1), int(preds.shape[2] - y2)))

                count_mat[:, :, y1:y2, x1:x2] += 1
        assert (count_mat == 0).sum() == 0
        if torch.onnx.is_in_onnx_export():
            # cast count_mat to constant while exporting to ONNX
            count_mat = torch.from_numpy(count_mat.cpu().detach().numpy()).to(device=img.device)
        preds = preds / count_mat
        if rescale:
            preds = resize(
                preds,
                size=img_meta[0]['ori_shape'][:2],
                mode='bilinear',
                align_corners=self.align_corners,
                warning=False,
            )
        return preds

    def whole_inference(self, img, img_meta, rescale):
        """Inference with full image."""

        seg_logit = self.encode_decode(img, img_meta)
        if rescale:
            # support dynamic shape for onnx
            if torch.onnx.is_in_onnx_export():
                size = img.shape[2:]
            else:
                size = img_meta[0]['ori_shape'][:2]
            seg_logit = resize(seg_logit, size=size, mode='bilinear', align_corners=self.align_corners, warning=False)

        return seg_logit

    def inference(self, img, img_meta, rescale):
        """Inference with slide/whole style.

        Args:
            img (Tensor): The input image of shape (N, 3, H, W).
            img_meta (dict): Image info dict where each dict has: 'img_shape',
                'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            rescale (bool): Whether rescale back to original shape.

        Returns:
            Tensor: The output segmentation map.
        """

        assert self.test_cfg.mode in ['slide', 'whole']
        ori_shape = img_meta[0]['ori_shape']
        assert all(_['ori_shape'] == ori_shape for _ in img_meta)
        if self.test_cfg.mode == 'slide':
            seg_logit = self.slide_inference(img, img_meta, rescale)
        else:
            seg_logit = self.whole_inference(img, img_meta, rescale)
        output = F.softmax(seg_logit, dim=1)
        flip = img_meta[0]['flip']
        if flip:
            flip_direction = img_meta[0]['flip_direction']
            assert flip_direction in ['horizontal', 'vertical']
            if flip_direction == 'horizontal':
                output = output.flip(dims=(3,))
            elif flip_direction == 'vertical':
                output = output.flip(dims=(2,))

        return output

    def simple_test(self, img, img_meta, rescale=True):
        """Simple test with single image."""
        seg_logit = self.inference(img, img_meta, rescale)
        seg_pred = seg_logit.argmax(dim=1)
        if torch.onnx.is_in_onnx_export():
            # our inference backend only support 4D output
            seg_pred = seg_pred.unsqueeze(0)
            return seg_pred
        seg_pred = seg_pred.cpu().numpy()
        # unravel batch dim
        seg_pred = list(seg_pred)
        return seg_pred

    def aug_test(self, imgs, img_metas, rescale=True):
        """Test with augmentations.

        Only rescale=True is supported.
        """
        # aug_test rescale all imgs back to ori_shape for now
        assert rescale
        # to save memory, we get augmented seg logit inplace
        seg_logit = self.inference(imgs[0], img_metas[0], rescale)
        for i in range(1, len(imgs)):
            cur_seg_logit = self.inference(imgs[i], img_metas[i], rescale)
            seg_logit += cur_seg_logit
        seg_logit /= len(imgs)
        seg_pred = seg_logit.argmax(dim=1)
        seg_pred = seg_pred.cpu().numpy()
        # unravel batch dim
        seg_pred = list(seg_pred)
        return seg_pred
