import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
import numpy as np

from .resnet38d import Net


class ClsNet(Net):
    def __init__(self):
        super().__init__()
        self.dropout7 = torch.nn.Dropout2d(0.5)
        # class 19
        self.fc = nn.Conv2d(4096, 19, 1, bias=False)
        torch.nn.init.xavier_uniform_(self.fc.weight)
        self.not_training = [self.conv1a, self.b2, self.b2_1, self.b2_2]
        self.from_scratch_layers = [self.fc]
    def forward(self, x):
        #执行父类的forward方法，意味着输入x,经过backbone之后得到输出
        x,conv4 = super().forward(x)
        #dropout
        x = self.dropout7(x)
        #池化层 这一步应该是全局平均池化;池化的结果是得到了一个B*C*1的一维向量！
        x = F.avg_pool2d(x, kernel_size=(x.size(2), x.size(3)), padding=0)
        #池化之后的特征
        feature = x
        #reshape tensor; -1表示该维度取决于其他维度！
        #这一步应该是reshape成两个维度 B*C
        feature = feature.view(feature.size(0), -1)
        
        # class 19
        x = self.fc(x)
        x = x.view(x.size(0), -1)
        y = torch.sigmoid(x)
        # x -- B*19 的原始分类结果
        # y -- sigmoid后的分类结果
        # feature -- B*C的GAP后的特征
        return x, feature, y
    def multi_label(self, x):
        x = torch.sigmoid(x)
        tmp = x.cpu()
        tmp = tmp.data.numpy()
        _, cls = np.where(tmp>0.5)

        return cls, tmp


    def forward_cam(self, x):
        x,conv4 = super().forward(x)
        x = F.conv2d(x, self.fc.weight)
        x = F.relu(x)

        return x,conv4

    def get_parameter_groups(self):
        groups = ([], [], [], [])

        for m in self.modules():

            if isinstance(m, nn.Conv2d):

                if m.weight.requires_grad:
                    if m in self.from_scratch_layers:
                        groups[2].append(m.weight)
                    else:
                        groups[0].append(m.weight)

                if m.bias is not None and m.bias.requires_grad:

                    if m in self.from_scratch_layers:
                        groups[3].append(m.bias)
                    else:
                        groups[1].append(m.bias)

        return groups
 