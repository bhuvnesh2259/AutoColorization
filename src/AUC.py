import glob
import numpy as np
from PIL import Image
import tensorflow as tf
import math
import matplotlib.image as mpimg
import glob
import skimage.color as sk

def AUC(img1, img2):
    diff = np.square(img1-img2)
    auc = np.zeros(256)
    for i in range(256):
        auc[i] += diff[diff<=i/512].size
    return auc, np.sum(auc)/(img1.size*256)