import numpy as np
from PIL import Image
import torch
import torchvision
from imgaug import augmenters as iaa
import sys
import os

if len(sys.argv) > 2:
    folder_path = sys.argv[1]
    folder_path = folder_path.strip('/')
    ext = sys.argv[2]
else:
    folder_path = "."
    ext = "jpg"




snp_aug = iaa.SaltAndPepper(0.1,per_channel=True)
gauss_aug = iaa.AdditiveGaussianNoise(scale=0.3*255,per_channel=True)
brightness_aug = iaa.WithBrightnessChannels(iaa.Add((-50,50)))
hue_aug = iaa.WithHueAndSaturation(
        iaa.WithChannels(0, iaa.Add((0, 10)))
        )

items = os.listdir(folder_path)
items = [f for f in items if f.split('.')[1] == ext]

def create_augs(filename):
    img_path = folder_path + '/' + filename
    img =  np.asarray(Image.open(img_path).convert('RGB'))
    combin = [[0],[1],[2],[0,1],[0,2],[1,2],[0,1,2]]
    augs = [snp_aug,brightness_aug,hue_aug]
    i = 0
    for c in combin:
        res = np.copy(img) 
        for idx in c:
           res = augs[idx](image=res)
        name, _ = filename.split('.')
        Image.fromarray(res).save(name+str(i)+'.'+ext)
        i+=1

for f in items:
    create_augs(f)









