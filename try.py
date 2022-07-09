import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torchvision
from torchvision import transforms
import skimage
from skimage import morphology
from skimage.morphology import disk,square,diamond
from skimage import measure
from skimage.measure import label
from skimage.color import label2rgb
from skimage.io import imread
import os
from skimage import io
"""
data = np.load('/shared/home/v_zixin_tang/dataset/data256/dicTrain.npy')
print(data.shape)
dir = "/shared/home/v_zixin_tang/dataset/HK2_DIC/data256/dicTrain"
if not os.path.exists(dir):
    os.makedirs(dir)
for i in range (data.shape[0]):
    B = data[i, : ,:]
    Image.fromarray(B).save(dir + "/" + str(i)+ ".tif")
"""
path1=r'/shared/home/v_zixin_tang/dataset/HK2_DIC/data256/dicTrain/2.tif'
img=Image.open(path1)
#img=Image.fromarray(np.uint8(img))
img = np.asarray(img)
#mask=Image.open(path2)
plt.imshow(img)
plt.show()