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
import imageio
import cv2
from torchvision.utils import save_image
from torchvision.transforms.functional import normalize, resize, to_pil_image

import numpy as np
from matplotlib import cm
from PIL import Image

CLASSES = ('road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
               'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky',
               'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle',
               'bicycle')

def overlay_mask(img: Image.Image, mask: Image.Image, colormap: str = "jet", alpha: float = 0.7) -> Image.Image:
    """Overlay a colormapped mask on a background image
    >>> from PIL import Image
    >>> import matplotlib.pyplot as plt
    >>> from torchcam.utils import overlay_mask
    >>> img = ...
    >>> cam = ...
    >>> overlay = overlay_mask(img, cam)
    Args:
        img: background image
        mask: mask to be overlayed in grayscale
        colormap: colormap to be applied on the mask
        alpha: transparency of the background image
    Returns:
        overlayed image
    Raises:
        TypeError: when the arguments have invalid types
        ValueError: when the alpha argument has an incorrect value
    """

    if not isinstance(img, Image.Image) or not isinstance(mask, Image.Image):
        raise TypeError("img and mask arguments need to be PIL.Image")

    if not isinstance(alpha, float) or alpha < 0 or alpha >= 1:
        raise ValueError("alpha argument is expected to be of type float between 0 and 1")

    cmap = cm.get_cmap(colormap)
    # Resize mask and apply colormap
    overlay = mask.resize(img.size, resample=Image.BICUBIC)
    overlay = (255 * cmap(np.asarray(overlay) ** 2)[:, :, :3]).astype(np.uint8)
    # Overlay the image with the mask
    overlayed_img = Image.fromarray((alpha * np.asarray(img) + (1 - alpha) * overlay).astype(np.uint8))

    return overlayed_img

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
        # img = imtools.ResizeLong(img, 256, 512)
        # img = imtools.Flip(img)
        # img = imtools.ColorJitter(img)
        # img = np.array(img)
        # img = imtools.NNormalize(img)
        # img = imtools.Crop(img, self.crop_size)
        # img = img.transpose(2,0,1)
        img = np.asarray(img)
        img = model.normalize(img)
        img = imtools.HWC_to_CHW(img)
        return img,self.file_names[index],label19
    def __len__(self):
        return len(self.file_names)
#一些参数
weights='/root/autodl-tmp/DAFormer/pretrained/ep50.pth'
#建立模型
model=ClsNet()
#建立数据集
val_dataset=cityscape_cls_dataset('/root/autodl-tmp/DAFormer/mmseg/models/uda/clsnet/data/gta_val_label.json')
val_loader = DataLoader(val_dataset,batch_size=1,shuffle=False, num_workers=1)
#载入预训练模型
weights_dict = torch.load(weights)
model.load_state_dict(weights_dict) 
model=model.cuda()
model.eval()
class_wise_acc=np.zeros(19)
class_wise_all_cnt=np.zeros(19)
it = 0
for iter, (data,img_path ,label_19) in tqdm(enumerate(val_loader)):
    img = data.cuda()
    label_19=label_19.cuda()
    ori_img = Image.open(img_path[0])
    ori_img.save("/root/autodl-tmp/DAFormer/demo/cam_gta_" + str(it) + ".png")

    with torch.no_grad():
        #喂入模型,前向传播
        x_19,fea,y_19=model(img)
        #推理类别激活图
        cam_19,fea_conv4=model.forward_cam(img)
        cam_19 = cam_19.detach().cpu().numpy() * label_19.clone().view(19, 1, 1).cpu().numpy()
        # print(img.shape)
        # print(cam_19.shape)
        # assert 1==0
        #可视化特征图
        for index in range(19):#通过遍历的方式，将19个通道的tensor拿出
            result = overlay_mask(ori_img, to_pil_image(cam_19[0][index], mode='F'), alpha=0.5)
            result.save("/root/autodl-tmp/DAFormer/demo/cam_gta_" + str(it) + "_" + str(index) + "_" + CLASSES[index] + ".png")
            # imageio.imsave( '/root/autodl-tmp/DAFormer/demo/'+str(index) + "_cam.png", cam_19[index])
        #计算准确率
        cls19_prob = y_19.cpu().data.numpy()
        cls19_gt = label_19.cpu().data.numpy()
        cls19_prob=cls19_prob[0]
        cls19_gt=cls19_gt[0]
        cls19_prob[np.where(cls19_prob>=0.5)]=1
        cls19_prob[np.where(cls19_prob<0.5)]=0
        for i in range(19):
            if cls19_prob[i]==cls19_gt[i] and cls19_prob[i]==1:
                class_wise_acc[i]+=1
            if cls19_gt[i]==1:
                class_wise_all_cnt[i]+=1
    it = it + 1
for i in range(19):
    if class_wise_all_cnt[i]!=0:
        class_wise_acc[i]=class_wise_acc[i]/class_wise_all_cnt[i]
print(class_wise_acc)
print(class_wise_all_cnt)