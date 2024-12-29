###思路，由于类别数的改变，只能保留backbone的预训练参数，在分割数据集上重新训练
import os
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import json
import imutils as imtools
import numpy as np
from network.resnet38_cls import ClsNet 
import torch.optim as optim
from tqdm import tqdm
import torch.nn.functional as F
def compute_acc(pred_labels, gt_labels):
    pred_correct_count = 0
    pred_correct_list = []
    for pred_label in pred_labels:
        if pred_label in gt_labels:
            pred_correct_count += 1
    union = len(gt_labels) + len(pred_labels) - pred_correct_count
    acc = round(pred_correct_count/union, 4)
    return acc
class AverageMeter:
    def __init__(self, *keys):
        self.__data = dict()
        for k in keys:
            self.__data[k] = [0.0, 0]

    def add(self, dict):
        for k, v in dict.items():
            self.__data[k][0] += v
            self.__data[k][1] += 1

    def get(self, *keys):
        if len(keys) == 1:
            return self.__data[keys[0]][0] / self.__data[keys[0]][1]
        else:
            v_list = [self.__data[k][0] / self.__data[k][1] for k in keys]
            return tuple(v_list)

    def pop(self, key=None):
        if key is None:
            for k in self.__data.keys():
                self.__data[k] = [0.0, 0]
        else:
            v = self.get(key)
            self.__data[key] = [0.0, 0]
            return v
#自建优化器
class PolyOptimizer(torch.optim.SGD):

    def __init__(self, params, lr, weight_decay, max_step, momentum=0.9):
        super().__init__(params, lr, weight_decay)

        self.global_step = 0
        self.max_step = max_step
        self.momentum = momentum

        self.__initial_lr = [group['lr'] for group in self.param_groups]


    def step(self, closure=None):

        if self.global_step < self.max_step:
            lr_mult = (1 - self.global_step / self.max_step) ** self.momentum

            for i in range(len(self.param_groups)):
                self.param_groups[i]['lr'] = self.__initial_lr[i] * lr_mult

        super().step(closure)

        self.global_step += 1
#该数据集主要起喂给分类网络训练数据的作用
class cityscape_cls_dataset(Dataset):
    def __init__(self,cls_label_json_path):
        self.cls_label_json_path=cls_label_json_path
        #加载json
        with open(self.cls_label_json_path) as f:
            self.data=json.load(f)
        self.file_names=[]
        for filname,label in self.data.items():
            self.file_names.append(filname)
        self.crop_size=224
        #json加载完成
    def __getitem__(self,index):
        #载入标注
        label19=torch.from_numpy(np.array(self.data[self.file_names[index]])).float()
        #载入图片
        img=Image.open(self.file_names[index]).convert("RGB")
        #载入完成之后数据增强
        img = imtools.ResizeLong(img, 256, 512)
        img = imtools.Flip(img)
        img = imtools.ColorJitter(img)
        img = np.array(img)
        img = imtools.NNormalize(img)
        img = imtools.Crop(img, self.crop_size)
        img = img.transpose(2,0,1)
        img = torch.from_numpy(img)
        return img,label19
    def __len__(self):
        return len(self.file_names)
# parser = argparse.ArgumentParser()
# parser.add_argument("--network", default="network.resnet38_cls", type=str)
# parser.add_argument("--num_workers", default=8, type=int)
# parser.add_argument("--_root", required=True, type=str, help="the path to the dataset folder")
# parser.add_argument("--th", default=0.15, type=float, help="the threshold for the response map")
# parser.add_argument("--save_path", required=True, default=None, type=str, help="the path to save the CAM")
# args = parser.parse_args()
#about paramters
batch_size=16
num_workers=8
max_epoches=60
wt_dec=5e-4
lr=0.01
weights='/root/autodl-tmp/DAFormer/pretrained/res38_cls.pth'
#step1: build the model
model=ClsNet()
#step2: build the dataset
# training_dataset=cityscape_cls_dataset('/root/autodl-tmp/DAFormer/mmseg/models/segmentors/clsnet/data/cls_label.json')
# gta
training_dataset=cityscape_cls_dataset('/root/autodl-tmp/DAFormer/data/gta/cls_label.json')
train_data_loader = DataLoader(training_dataset,
                                    batch_size=batch_size,
                                    shuffle=True,
                                    num_workers=num_workers,
                                    pin_memory=True,
                                    drop_last=True)
max_step = (len(training_dataset) // batch_size) * max_epoches
#step3: build the optimizer 
param_groups = model.get_parameter_groups()
optimizer = PolyOptimizer([
    {'params': param_groups[0], 'lr': lr, 'weight_decay': wt_dec},
    {'params': param_groups[1], 'lr': 2*lr, 'weight_decay': 0},
    {'params': param_groups[2], 'lr': 10*lr, 'weight_decay': wt_dec},
    {'params': param_groups[3], 'lr': 20*lr, 'weight_decay': 0}
], lr=lr, weight_decay=wt_dec, max_step=max_step)
#step4: load pretrained para to model
model_dict = model.state_dict()
weights_dict = torch.load(weights)
#取出参数相同的部分
weights_dict = {k: v for k, v in weights_dict.items() if k in model_dict}
model_dict.update(weights_dict)#用预训练模型参数更新new_model中的部分参数
model.load_state_dict(model_dict)
model=model.cuda()
model.train()
avg_meter = AverageMeter('loss')
step=0
save_folder='/root/autodl-tmp/DAFormer/pretrained'
for ep in range(max_epoches):
        ep_count = 0
        ep_acc_vote = 0

        cls19_ep_EM = 0
        cls19_ep_acc = 0
        for iter, (data, label_19) in tqdm(enumerate(train_data_loader)):
            img = data.cuda()
            label_19=label_19.cuda()
            #喂入模型,前向传播
            x_19,fea,y_19=model(img)
            #计算准确率
            cls19_prob = y_19.cpu().data.numpy()
            cls19_gt = label_19.cpu().data.numpy()
            for num, one in enumerate(cls19_prob):
                ep_count += 1
                pass_cls = np.where(one > 0.5)[0]
                true_cls_19 = np.where(cls19_gt[num] == 1)[0]

                if np.array_equal(pass_cls, true_cls_19) == True:  # exact match
                    cls19_ep_EM += 1

                acc = compute_acc(pass_cls, true_cls_19)
                cls19_ep_acc += acc
            
            avg_cls19_ep_EM = round(cls19_ep_EM/ep_count, 4)
            avg_cls19_ep_acc = round(cls19_ep_acc/ep_count, 4)

            cls_19_loss = F.multilabel_soft_margin_loss(x_19, label_19)

            loss = cls_19_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            torch.cuda.empty_cache()
            print(' 19 Classification Loss ', cls_19_loss.item(),step,avg_cls19_ep_acc)
            step += 1
        #10个epoch保存一次模型
        if ep % 10 == 0:
            torch.save(model.state_dict(), '{}/'.format(save_folder) + 'ClsEp{}.pth'.format(ep))
            print('Loss: {} achieves the lowest one => Epoch {} weights are saved!'.format(loss, ep))