from PIL import Image
import numpy as np
import os
from pathlib import Path

CLASSES = ('road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
           'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky',
           'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle',
           'bicycle')

PALETTE = [[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
           [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
           [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
           [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100],
           [0, 80, 100], [0, 0, 230], [119, 11, 32]]

def color2gray(img_rgb, rgb_mapping):
    gray_img = np.zeros(shape=(img_rgb.shape[0], img_rgb.shape[1]), dtype=np.uint8)
    for map_idx, rgb in enumerate(rgb_mapping):
        idx = np.where((img_rgb[..., 0]==rgb[0]) & (img_rgb[..., 1]==rgb[1]) & (img_rgb[..., 2]==rgb[2]))
        gray_img[idx] = map_idx
    return gray_img

def main():
    img_paths = list(Path('./preds').rglob('*.png'))
    img_paths = [str(p) for p in img_paths]

    out_dir = './labelTrainIds'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    for i in range(len(img_paths)):
        print(f'convert {i}')
        img = Image.open(img_paths[i])
        img = np.array(img)
        img_train_id = color2gray(img, PALETTE)
        name_train_id = img_paths[i][6:-12] + 'labelTrainIds.png'
        img_out = Image.fromarray(img_train_id)
        print(f'{out_dir}/{name_train_id}')
        img_out.save(f'{out_dir}/{name_train_id}')


if __name__ == '__main__':
    main()

