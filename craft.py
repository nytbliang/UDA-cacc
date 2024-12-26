from PIL import Image
import matplotlib.pyplot as plt
import mmseg.models.uda.clsnet.imutils as imtools
import numpy as np
import torch

# cam = "/root/autodl-tmp/DAFormer/demo/"

# img = "/root/autodl-tmp/DAFormer/data/cityscapes/leftImg8bit/val/frankfurt/frankfurt_000000_001016_leftImg8bit.png"
# img1 = Image.open(img).convert('RGB')
# img1 = imtools.ResizeLong(img1, 256, 512)
# img1 = np.array(img1)
# img1 = imtools.Crop(img1, 224)
# img1 = Image.fromarray(np.uint8(img1))

# for i in range(19):
#     img2 = Image.open(cam+str(i)+"_cam.png")

#     img1 = img1.convert("RGBA")
#     img2 = img2.convert("RGBA")

#     out = Image.blend(img1, img2, 0.8)
#     out.save(cam + "blend_cam_" + str(i) + ".png")
#     print(str(i) + " saved" )

# img = '/root/autodl-tmp/DAFormer/work_dirs/local-basic/221103_2124_cs2acdcnight_uda_warm_fdthings_rcs_croppl_a999_daformer_mitb5_s0_4723c/preds1/GOPR0355_frame_000138_rgb_anon.png'
img = '/root/autodl-tmp/DAFormer/labelTrainIds/GOPR0355_frame_000138_rgb_anon.png'
img = Image.open(img)

img_np = np.array(img)
img_np = np.unique(img_np)
print(img_np)