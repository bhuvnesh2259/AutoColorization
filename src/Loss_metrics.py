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

def NN_ab(y):
    # y is [N, H, W, 3]
    NN_ab_x = np.round((y[:,:,:,1]+0.6)*19/1.2)
    NN_ab_y = np.round((y[:,:,:,2]+0.6)*19/1.2)
    NN_ab = NN_ab_x*20+NN_ab_y
    return NN_ab.astype(int)

def NN_ab1(y):
    # y is [N, H, W, 3]
    NN_ab_x = np.round((y[:,:,:,1]+0.6)*255/1.2)
    NN_ab_y = np.round((y[:,:,:,2]+0.6)*255/1.2)
    return NN_ab_x.astype(int), NN_ab_y.astype(int) 

def assign_prob1(NN_u, NN_v, y):
    # NN_u is [N, H, W]
    # NN_v is [N, H, W]
    # y is [N, H, W, 3]
    prob_dist1 = np.zeros((y.shape[0]*y.shape[1]*y.shape[2], 256))
    prob_dist2 = np.zeros((y.shape[0]*y.shape[1]*y.shape[2], 256))
    NN_u = np.reshape(NN_u, [NN_u.size,])
    NN_v = np.reshape(NN_v, [NN_v.size,])
    prob_dist1[range(y.shape[0]*y.shape[1]*y.shape[2]),NN_u] = 1
    prob_dist2[range(y.shape[0]*y.shape[1]*y.shape[2]),NN_v] = 1
    return np.reshape(prob_dist1,[y.shape[0], y.shape[1], y.shape[2], 256]), np.reshape(prob_dist2,[y.shape[0], y.shape[1], y.shape[2], 256])

def assign_prob(NN, y):
    # NN is [N, H, W]
    # y is [N, H, W, 3]
    prob_dist = np.zeros((y.shape[0]*y.shape[1]*y.shape[2], 400))
    NN = np.reshape(NN, [NN.size,])
    #prob_dist[range(y.shape[0]*y.shape[1]*y.shape[2]),NN] = 1
    
    prob_dist[range(y.shape[0]*y.shape[1]*y.shape[2]),NN] = 0.12
    prob_dist[range(NN[NN<399].size),NN[NN<399]+1] = 0.11
    prob_dist[range(NN[NN>0].size),NN[NN>0]+1] = 0.11
    prob_dist[range(NN[NN<380].size),NN[NN<380]+1] = 0.11
    prob_dist[range(NN[NN>20].size),NN[NN>20]+1] = 0.11
    prob_dist[range(NN[NN<379].size),NN[NN<379]+1] = 0.11
    prob_dist[range(NN[NN>21].size),NN[NN>21]+1] = 0.11
    prob_dist[range(NN[NN<381].size),NN[NN<381]+1] = 0.11
    prob_dist[range(NN[NN>19].size),NN[NN>19]+1] = 0.11    
    return np.reshape(prob_dist,[y.shape[0], y.shape[1], y.shape[2], 400])

def Prob_dist1(y):
    # Returns ab prob distribution for given training batch
    # y is [N, H, W,3] dim
    NN_u, NN_v = NN_ab1(y)
    p1, p2 = assign_prob1(NN_u, NN_v, y)
    return p1, p2

def Prob_dist(y):
    # Returns ab prob distribution for given training batch
    # y is [N, H, W,3] dim
    # x is [N, H, W, 400] dim
    NN = NN_ab(y)
    p = assign_prob(NN, y)
    return p

def assign_bin(y):
    # Returns the ab bin value for a given training batch
    # y is [N, H, W, 3] dim
    # NN is [N, H, W] dim
    NN = NN_ab(y)
    return NN

def YUV2rgb(UV, y):
    # UV is (N, H, W, 400)
    # returns RGB values corresponding to this YUV input
    inv_mat  = np.array([[1,0,1.13983],[1,-0.39465,-0.58060],[1,2.03211,0]])
    UV_arg_max = np.argmax(UV, axis=3)
    size_list = UV_arg_max.shape
    UV_t = np.exp(UV)/np.sum(np.exp(UV),axis=3)[...,np.newaxis]
    #UV_t = UV + 1e-6
    UV_a = np.zeros_like(UV_arg_max, dtype=np.float64)
    UV_b = np.zeros_like(UV_arg_max, dtype=np.float64)
    norm_a = np.zeros_like(UV_arg_max, dtype=np.float64)
    norm_b = np.zeros_like(UV_arg_max, dtype=np.float64)
    # code for taking mean
    #for i in range(20):
    #    for j in range(20):
    #        UV_a += (-0.6+i*1.2/19+1.2/40)*UV_t[:,:,:,i*20+j]
    #        UV_b += (-0.6+j*1.2/19+1.2/40)*UV_t[:,:,:,i*20+j]
    # Code for taking mode
    for i in range(20):
        for j in range(20):
              UV_a += (-0.6+i*1.2/19)*np.exp(np.log(UV_t[:,:,:,i*20+j])/0.32)
              UV_b += (-0.6+j*1.2/19)*np.exp(np.log(UV_t[:,:,:,i*20+j])/0.32)
              norm_a += np.exp(np.log(UV_t[:,:,:,i*20+j])/0.32)
    UV_a = UV_a/norm_a
    UV_b = UV_b/norm_a
    #UV_arg_row_number = UV_arg_max//20
    #UV_arg_column_number = UV_arg_max%20
    #UV_a = -0.6 + UV_arg_row_number*1.2/19 + 1.2/40
    #UV_b = -0.6 + UV_arg_column_number*1.2/19 + 1.2/40
    # Code for soft annealing
    YUV_output = np.concatenate((y[...],UV_a[..., np.newaxis],UV_b[..., np.newaxis]),axis = 3)
    RGB_output = YUV_output.dot(inv_mat)
    RGB_output[RGB_output>1] = 1
    RGB_output[RGB_output<0] = 0
    return RGB_output

def YUV2rgb2(U, V, y):
    # U is (N, H, W, 256)
    # V is (N, H, W, 256)
    # returns RGB values corresponding to this YUV input
    inv_mat  = np.array([[1,0,1.13983],[1,-0.39465,-0.58060],[1,2.03211,0]])
    U_arg_max = np.argmax(U,axis = 3)
    V_arg_max = np.argmax(V,axis = 3)
    UV_val_a = -0.6 + U_arg_max*1.2/255
    UV_val_b = -0.6 + V_arg_max*1.2/255
    YUV_output = np.concatenate((y[...],UV_val_a[...,np.newaxis],UV_val_b[...,np.newaxis]),axis = 3)
    RGB_output = YUV_output.dot(inv_mat)
    #RGB_output[RGB_output>1] = 1
    #RGB_output[RGB_output<0] = 0
    return RGB_output