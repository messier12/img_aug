import numpy as np
from PIL import Image
import torch
import torchvision
from imgaug import augmenters as iaa
import sys

if len(sys.argv) > 1:
    img_path = sys.argv[1]
else:
    img_path = "test.png"


im =  np.asarray(Image.open(img_path).convert('RGB'))

snp_aug = iaa.SaltAndPepper(0.1,per_channel=True)
gauss_aug = iaa.AdditiveGaussianNoise(scale=0.3*255,per_channel=True)
brightness_aug = iaa.WithBrightnessChannels(iaa.Add((-50,50)))
hue_aug = iaa.WithHueAndSaturation(
        iaa.WithChannels(0, iaa.Add((0, 10)))
        )


combin = [[0],[1],[2],[0,1],[0,2],[1,2],[0,1,2]]
augs = [snp_aug,brightness_aug,hue_aug]
i = 0
for c in combin:
    res = np.copy(im) 
    for idx in c:
       res = augs[idx](image=res)
    name, ext = img_path.split('.')
    Image.fromarray(res).save(name+str(i)+'.'+ext)
    i+=1









