import torch
import torch.nn as nn
import torch.nn.functional as F

def soft_label_cross_entropy(pred, soft_label, pixel_weights=None):

    loss = -soft_label.float()*F.log_softmax(pred, dim=1)
    if pixel_weights is None:
        return torch.mean(torch.sum(loss, dim=1))
    return torch.mean(pixel_weights*torch.sum(loss, dim=1))

class FCDiscriminatorWoCls(nn.Module):

    def __init__(self, num_classes=19, ndf=64):
        super(FCDiscriminatorWoCls, self).__init__()
        self.conv1 = nn.Conv2d(num_classes, ndf, kernel_size=4, stride=2, padding=1) # 4*4*19*64
        self.conv2 = nn.Conv2d(ndf, ndf*2, kernel_size=4, stride=2, padding=1) # 4*4*64*128
        self.conv3 = nn.Conv2d(ndf*2, ndf*4, kernel_size=4, stride=2, padding=1) # 4*4*128*256
        self.conv4 = nn.Conv2d(ndf*4, ndf*8, kernel_size=4, stride=2, padding=1) # 4*4*256*512
        self.classifier = nn.Conv2d(ndf*8, 1, kernel_size=4, stride=2, padding=1) # 4*4*512*1
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        # Vanilla
        # self.bce_loss = torch.nn.BCEWithLogitsLoss()
        # LS
        self.bce_loss = torch.nn.MSELoss()

    def forward(self, x):
        x = self.conv1(x)
        x = self.leaky_relu(x)
        x = self.conv2(x)
        x = self.leaky_relu(x)
        x = self.conv3(x)
        x = self.leaky_relu(x)
        x = self.conv4(x)
        x = self.leaky_relu(x)
        x = self.classifier(x)
        return x
    
    def forward_train(self, inputs, gt_dis, return_inv=False):
        pred = self.forward(inputs)
        
        dev = pred.device
        losses = dict()
        loss = self.bce_loss(pred, torch.FloatTensor(pred.data.size()).fill_(gt_dis).to(device=dev))
        losses['loss_px_wo_cls_dis'] = loss
        if return_inv:
            loss_inv = self.bce_loss(pred, torch.FloatTensor(pred.data.size()).fill_(1-gt_dis).to(device=dev))
            losses['loss_px_wo_cls_dis_inv'] = loss_inv
        return losses
    
class PixelDiscriminator(nn.Module):

    def __init__(self, input_nc=256, ndf=512, num_classes=19):
        super(PixelDiscriminator, self).__init__()
        self.D = nn.Sequential(
            nn.Conv2d(input_nc, ndf, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(ndf, ndf//2, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
		)
        self.cls1 = nn.Conv2d(ndf//2, num_classes, kernel_size=3, stride=1, padding=1)
        self.cls2 = nn.Conv2d(ndf//2, num_classes, kernel_size=3, stride=1, padding=1)

    def forward(self, x, gt, weights=None, return_inv=False):
        b,_,h,w = gt.shape
        size = gt.shape[-2:]
        losses = dict()
        out = self.D(x)
        src_out = self.cls1(out)
        tgt_out = self.cls2(out)
        if size is not None:
            src_out = F.interpolate(src_out, size=size, mode='bilinear', align_corners=True)
            tgt_out = F.interpolate(tgt_out, size=size, mode='bilinear', align_corners=True)
        out = torch.cat((src_out, tgt_out), dim=1)
        loss = soft_label_cross_entropy(out, gt, weights)
        losses['loss_px_dis'] = loss
        acc = (out.argmax(dim=1)==gt.argmax(dim=1)).sum().float()
        acc = (acc / (b*h*w)) * 100.0
        losses['acc_px_dis'] = acc
        if return_inv:
            out = torch.cat((tgt_out, src_out), dim=1)
            loss_inv = soft_label_cross_entropy(out, gt, weights)
            losses['loss_px_dis_inv'] = loss_inv
        return losses
    
class ImageDiscriminator(nn.Module):

    def __init__(self, input_nc=256, num_classes=19):
        super(ImageDiscriminator, self).__init__()
        self.D = nn.Sequential(
            nn.Linear(input_nc, 2048),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(2048,512),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        self.cls1 = nn.Linear(512, num_classes)
        self.cls2 = nn.Linear(512, num_classes)
        self.num_classes = num_classes
        
    def forward(self, x, cam, gt, gt_2k, weights=None, return_inv=False):
        # x -- (b, 256, h, w)
        # cam -- (b, 19, h, w)
        assert cam.size(1) == self.num_classes, 'cam.size(1) != num_class'
        batch_size = x.shape[0]
        size = x.shape[-2:]
        
        cam = F.interpolate(cam, size=size, mode='bilinear', align_corners=True).detach()
        cams_feature = cam.unsqueeze(2)*x.unsqueeze(1) # bs*19*256*h*w
        cams_feature = cams_feature.view(cams_feature.size(0),cams_feature.size(1),cams_feature.size(2),-1) 
        cams_feature = torch.mean(cams_feature,-1) # b*19*256*1
        cams_feature = cams_feature.reshape(batch_size, self.num_classes, -1) # b*19*256
        mask = gt > 0
        feature_list = [cams_feature[i][mask[i]] for i in range(batch_size)] # b*n*256
        out = [self.D(y) for y in feature_list]
        src_out = [self.cls1(y) for y in out]
        tgt_out = [self.cls2(y) for y in out]   # b*n*19
        # labels = [torch.nonzero(gt_2k[i]).squeeze(1) for i in range(gt_2k.shape[0])]
        labels = [gt_2k[i][mask[i]] for i in range(batch_size)]  # b*n*38
        weight_ls = [weights[i][mask[i]] for i in range(batch_size)] # b*n


        losses = dict()
        loss = 0
        loss_inv = 0
        acc = 0
        cnt = 0

        # n*19
        for src_logit, tgt_logit, label, weight in zip(src_out, tgt_out, labels, weight_ls):
            if label.sum() == 0:
                continue
            cnt += label.sum()
            out = torch.cat((src_logit, tgt_logit), dim=1)
            loss += soft_label_cross_entropy(out, label, weight)

            acc += (out.argmax(dim=1)==label.argmax(dim=1)).sum().float()

            if return_inv:
                out = torch.cat((tgt_logit, src_logit), dim=-1)
                loss_inv += soft_label_cross_entropy(out, label, weight)

        losses['loss_img_dis'] = loss / batch_size
        losses['acc_img_dis'] = (acc / cnt) * 100.0
        if return_inv:
            losses['loss_img_dis_inv'] = loss_inv
        return losses