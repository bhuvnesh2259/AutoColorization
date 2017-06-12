import tensorflow as tf
import numpy as np
import math
import matplotlib.image as mpimg
import glob
from data_utils import load_data, augment_data,normalise_train,normalise_test,rgb2gray
import matplotlib.pyplot as plt
import tensorflow.contrib.slim as slim



''' a very baseline garbage model
'''
def baselinish(X,Y,is_training):
    
    W_conv = tf.get_variable("Wconv",shape = [7,7,1,32],initializer=tf.contrib.layers.xavier_initializer())
    b_conv = tf.get_variable("bconv",shape = [32])
    a1 = tf.nn.conv2d(X, W_conv, strides=[1,1,1,1], padding='SAME') + b_conv
    a1 = tf.nn.relu(a1)
    a1 = tf.contrib.layers.batch_norm(a1,center = True, scale = True, is_training = is_training,scope = 'bn1')
    #a1 = tf.nn.max_pool(a1,ksize = [1,2,2,1],strides = [1,2,2,1],padding = 'VALID')

    W_conv1 = tf.get_variable("Wconv1",shape = [3,3,32,32],initializer=tf.contrib.layers.xavier_initializer())
    b_conv1 = tf.get_variable("bconv1",shape = [32])
    a1 = tf.nn.conv2d(a1, W_conv1, strides=[1,1,1,1], padding='SAME') + b_conv1
    a1 = tf.nn.sigmoid(a1)
    a1 = tf.contrib.layers.batch_norm(a1,center = True, scale = True, is_training = is_training,scope = 'bn3')
    
    W_conv2 = tf.get_variable("Wconv2",shape = [3,3,32,64],initializer=tf.contrib.layers.xavier_initializer())
    b_conv2 = tf.get_variable("bconv2",shape = [64])
    a1 = tf.nn.conv2d(a1, W_conv2, strides=[1,1,1,1], padding='SAME') + b_conv2
    a1 = tf.nn.relu(a1)
    a1 = tf.contrib.layers.batch_norm(a1,center = True, scale = True, is_training = is_training,scope = 'bn2')
    
    W_conv3 = tf.get_variable("Wconv3",shape = [1,1,64,32],initializer=tf.contrib.layers.xavier_initializer())
    b_conv3 = tf.get_variable("bconv3",shape = [32])
    a1 = tf.nn.conv2d(a1, W_conv3, strides=[1,1,1,1], padding='SAME') + b_conv3
    a1 = tf.nn.softmax(a1,dim = 3)
    W_conv4 = tf.get_variable("Wconv4",shape = [1,1,32,3],initializer=tf.contrib.layers.xavier_initializer())
    b_conv4 = tf.get_variable("bconv4",shape = [3])
    a1 = tf.nn.conv2d(a1, W_conv4, strides=[1,1,1,1], padding='SAME') + b_conv4
    


    return a1
    
    #a1 = tf.nn.relu(a1)
    #a1 = tf.contrib.layers.batch_norm(a1,center = True, scale = True, is_training = is_training,scope = 'bn3')



def baselinish2(X,is_training):

    Y = tf.image.convert_image_dtype(X,tf.float32)
    #inputt = tf.image.rgb_to_hsv(Y)
    #inpp = inputt[:,:,:,0:2]
    #inp = inputt[:,:,:,2:3]

    conv_mat = tf.constant(np.array([[0.299,0.587,0.114],[-0.14713,-0.2888,0.436],[0.615,-0.514999,-0.10001]]),dtype = tf.float32)
    inv_conv_mat = tf.constant(np.array([[1,0,1.13983],[1,-0.39465,-0.58060],[1,2.03211,0]]),dtype = tf.float32)

    Y = tf.reshape(Y,[-1,3])

    inputt = tf.matmul(Y,conv_mat)
    inputt = tf.reshape(inputt,[-1,64,64,3])
    Y = tf.reshape(Y,[-1,64,64,3])
    inpp = inputt[:,:,:,1:3]
    inp = inputt[:,:,:,0:1]

    W_conv = tf.get_variable("Wconv",shape = [7,7,1,32],initializer=tf.contrib.layers.xavier_initializer())
    b_conv = tf.get_variable("bconv",shape = [32])
    a1 = tf.nn.conv2d(inp, W_conv, strides=[1,1,1,1], padding='SAME') + b_conv
    a1 = tf.nn.relu(a1)
    a1 = tf.contrib.layers.batch_norm(a1,center = True, scale = True, is_training = is_training,scope = 'bn1')
    #a1 = tf.nn.max_pool(a1,ksize = [1,2,2,1],strides = [1,2,2,1],padding = 'VALID')

    W_conv1 = tf.get_variable("Wconv1",shape = [3,3,32,32],initializer=tf.contrib.layers.xavier_initializer())
    b_conv1 = tf.get_variable("bconv1",shape = [32])
    a1 = tf.nn.conv2d(a1, W_conv1, strides=[1,1,1,1], padding='SAME') + b_conv1
    a1 = tf.nn.sigmoid(a1)
    a1 = tf.contrib.layers.batch_norm(a1,center = True, scale = True, is_training = is_training,scope = 'bn3')
    
    W_conv2 = tf.get_variable("Wconv2",shape = [3,3,32,64],initializer=tf.contrib.layers.xavier_initializer())
    b_conv2 = tf.get_variable("bconv2",shape = [64])
    a1 = tf.nn.conv2d(a1, W_conv2, strides=[1,1,1,1], padding='SAME') + b_conv2
    a1 = tf.nn.relu(a1)
    a1 = tf.contrib.layers.batch_norm(a1,center = True, scale = True, is_training = is_training,scope = 'bn2')
    
    W_conv3 = tf.get_variable("Wconv3",shape = [1,1,64,32],initializer=tf.contrib.layers.xavier_initializer())
    b_conv3 = tf.get_variable("bconv3",shape = [32])
    a1 = tf.nn.conv2d(a1, W_conv3, strides=[1,1,1,1], padding='SAME') + b_conv3
    #a1 = tf.nn.relu(a1)
    W_conv4 = tf.get_variable("Wconv4",shape = [1,1,32,2],initializer=tf.contrib.layers.xavier_initializer())
    b_conv4 = tf.get_variable("bconv4",shape = [2])
    a1 = tf.nn.conv2d(a1, W_conv4, strides=[1,1,1,1], padding='SAME') + b_conv4

    a3 = tf.concat((inp,a1),axis = 3)
    #a2 = tf.image.hsv_to_rgb(a3)
    a3 = tf.reshape(a3,[-1,3])
    a2 = tf.matmul(a3,inv_conv_mat)
    a2 = tf.reshape(a2,[-1,64,64,3])

    
    return a1,a2,inpp,Y

def baselinish3(X,is_training):

    Y = tf.image.convert_image_dtype(X,tf.float32)
    #inputt = tf.image.rgb_to_hsv(Y)
    #inpp = inputt[:,:,:,0:2]
    #inp = inputt[:,:,:,2:3]

    conv_mat = tf.constant(np.array([[0.299,0.587,0.114],[-0.14713,-0.2888,0.436],[0.615,-0.514999,-0.10001]]),dtype = tf.float32)
    inv_conv_mat = tf.constant(np.array([[1,0,1.13983],[1,-0.39465,-0.58060],[1,2.03211,0]]),dtype = tf.float32)

    Y = tf.reshape(Y,[-1,3])

    inputt = tf.matmul(Y,conv_mat)
    inputt = tf.reshape(inputt,[-1,32,32,3])
    Y = tf.reshape(Y,[-1,32,32,3])
    inpp = inputt[:,:,:,1:3]
    inp = inputt[:,:,:,0:1]

    W_conv = tf.get_variable("Wconv",shape = [7,7,1,32],initializer=tf.contrib.layers.xavier_initializer())
    b_conv = tf.get_variable("bconv",shape = [32])
    a1 = tf.nn.conv2d(inp, W_conv, strides=[1,1,1,1], padding='SAME') + b_conv
    a1 = tf.nn.relu(a1)
    a1 = tf.contrib.layers.batch_norm(a1,center = True, scale = True, is_training = is_training,scope = 'bn1')
    #a1 = tf.nn.max_pool(a1,ksize = [1,2,2,1],strides = [1,2,2,1],padding = 'VALID')

    W_conv1 = tf.get_variable("Wconv1",shape = [3,3,32,32],initializer=tf.contrib.layers.xavier_initializer())
    b_conv1 = tf.get_variable("bconv1",shape = [32])
    a1 = tf.nn.conv2d(a1, W_conv1, strides=[1,1,1,1], padding='SAME') + b_conv1
    a1 = tf.nn.sigmoid(a1)
    a1 = tf.contrib.layers.batch_norm(a1,center = True, scale = True, is_training = is_training,scope = 'bn3')
    
    W_conv2 = tf.get_variable("Wconv2",shape = [3,3,32,64],initializer=tf.contrib.layers.xavier_initializer())
    b_conv2 = tf.get_variable("bconv2",shape = [64])
    a1 = tf.nn.conv2d(a1, W_conv2, strides=[1,1,1,1], padding='SAME') + b_conv2
    a1 = tf.nn.relu(a1)
    a1 = tf.contrib.layers.batch_norm(a1,center = True, scale = True, is_training = is_training,scope = 'bn2')
    
    W_conv3 = tf.get_variable("Wconv3",shape = [1,1,64,32],initializer=tf.contrib.layers.xavier_initializer())
    b_conv3 = tf.get_variable("bconv3",shape = [32])
    a1 = tf.nn.conv2d(a1, W_conv3, strides=[1,1,1,1], padding='SAME') + b_conv3
    #a1 = tf.nn.relu(a1)
    W_conv4 = tf.get_variable("Wconv4",shape = [1,1,32,2],initializer=tf.contrib.layers.xavier_initializer())
    b_conv4 = tf.get_variable("bconv4",shape = [2])
    a1 = tf.nn.conv2d(a1, W_conv4, strides=[1,1,1,1], padding='SAME') + b_conv4

    a3 = tf.concat((inp,a1),axis = 3)
    #a2 = tf.image.hsv_to_rgb(a3)
    a3 = tf.reshape(a3,[-1,3])
    a2 = tf.matmul(a3,inv_conv_mat)
    a2 = tf.reshape(a2,[-1,32,32,3])

    
    return a1,a2,inpp,Y

def lrelu(x, leak=0., name='lrelu'):
    return tf.maximum(leak*x, x)


def complex_pokemon_model(gray_image,train=True):
   conv1 = lrelu(slim.convolution(gray_image, 32, 3, stride=1, scope='conv1', normalizer_fn=slim.batch_norm, activation_fn=tf.identity))
   conv2 = lrelu(slim.convolution(conv1, 32, 3, stride=1, scope='conv2', normalizer_fn=slim.batch_norm, activation_fn=tf.identity))
   conv3 = lrelu(slim.convolution(conv2, 64, 3, stride=1, scope='conv3', normalizer_fn=slim.batch_norm, activation_fn=tf.identity))
   conv4 = lrelu(slim.convolution(conv3, 64, 3, stride=1, scope='conv4', normalizer_fn=slim.batch_norm, activation_fn=tf.identity))
   conv5 = lrelu(slim.convolution(conv4, 128, 3, stride=1, scope='conv5', normalizer_fn=slim.batch_norm, activation_fn=tf.identity))
   conv6 = lrelu(slim.convolution(conv5, 128, 3, stride=1, scope='conv6', normalizer_fn=slim.batch_norm, activation_fn=tf.identity))
   conv7 = lrelu(slim.convolution(conv6, 256, 3, stride=1, scope='conv7', normalizer_fn=slim.batch_norm, activation_fn=tf.identity))
   conv8 = lrelu(slim.convolution(conv7, 256, 3, stride=1, scope='conv8', normalizer_fn=slim.batch_norm, activation_fn=tf.identity))
   conv9 = lrelu(slim.convolution(conv8, 128, 3, stride=1, scope='conv9', normalizer_fn=slim.batch_norm, activation_fn=tf.identity))
   conv10 = lrelu(slim.convolution(conv9, 128, 3, stride=1, scope='conv10', normalizer_fn=slim.batch_norm, activation_fn=tf.identity))
   conv11 = lrelu(slim.convolution(conv10, 64, 1, stride=1, scope='conv11', normalizer_fn=slim.batch_norm, activation_fn=tf.identity))
   conv12 = lrelu(slim.convolution(conv11, 64, 1, stride=1, scope='conv12', normalizer_fn=slim.batch_norm, activation_fn=tf.identity))
   conv13 = lrelu(slim.convolution(conv12, 32, 1, stride=1, scope='conv13', normalizer_fn=slim.batch_norm, activation_fn=tf.identity))
   conv14 = lrelu(slim.convolution(conv13, 32, 1, stride=1, scope='conv14', normalizer_fn=slim.batch_norm, activation_fn=tf.identity))
   conv15 = lrelu(slim.convolution(conv14, 16, 1, stride=1, scope='conv15', normalizer_fn=slim.batch_norm, activation_fn=tf.identity))
   conv16 = lrelu(slim.convolution(conv15, 16, 1, stride=1, scope='conv16', normalizer_fn=slim.batch_norm, activation_fn=tf.identity))
   conv17 = lrelu(slim.convolution(conv16, 8, 1, stride=1, scope='conv17', normalizer_fn=slim.batch_norm, activation_fn=tf.identity))
   if train: conv17 = tf.nn.dropout(conv17, 0.8)
   conv18 = lrelu(slim.convolution(conv17, 3, 1, stride=1, scope='conv18', normalizer_fn=slim.batch_norm, activation_fn=tf.identity))
   if train: conv18 = tf.nn.dropout(conv18, 0.8)
   
   tf.add_to_collection('vars', conv1)
   tf.add_to_collection('vars', conv2)
   tf.add_to_collection('vars', conv3)
   tf.add_to_collection('vars', conv4)
   tf.add_to_collection('vars', conv5)
   tf.add_to_collection('vars', conv6)
   tf.add_to_collection('vars', conv7)
   tf.add_to_collection('vars', conv8)
   tf.add_to_collection('vars', conv9)
   tf.add_to_collection('vars', conv10)
   tf.add_to_collection('vars', conv11)
   tf.add_to_collection('vars', conv12)
   tf.add_to_collection('vars', conv13)
   tf.add_to_collection('vars', conv14)
   tf.add_to_collection('vars', conv15)
   tf.add_to_collection('vars', conv16)
   tf.add_to_collection('vars', conv17)
   tf.add_to_collection('vars', conv18)
   
   return conv18


def complex_pokemon_model2(X,train=True):
    
    
  Y = tf.image.convert_image_dtype(X,tf.float32)

  conv_mat = tf.constant(np.array([[0.299,0.587,0.114],[-0.14713,-0.2888,0.436],[0.615,-0.514999,-0.10001]]),dtype = tf.float32)
  inv_conv_mat = tf.constant(np.array([[1,0,1.13983],[1,-0.39465,-0.58060],[1,2.03211,0]]),dtype = tf.float32)

  Y = tf.reshape(Y,[-1,3])

  inputt = tf.matmul(Y,conv_mat)
  inputt = tf.reshape(inputt,[-1,32,32,3])
  Y = tf.reshape(Y,[-1,32,32,3])
  inpp = inputt[:,:,:,1:3]
  inp = inputt[:,:,:,0:1]

  conv1 = lrelu(slim.convolution(inp, 32, 3, stride=1, scope='conv1', normalizer_fn=slim.batch_norm, normalizer_params = {'is_training':train},activation_fn=tf.identity))
  conv2 = lrelu(slim.convolution(conv1, 32, 3, stride=1, scope='conv2', normalizer_fn=slim.batch_norm,normalizer_params = {'is_training':train}, activation_fn=tf.identity))
  conv3 = lrelu(slim.convolution(conv2, 64, 3, stride=1, scope='conv3', normalizer_fn=slim.batch_norm,normalizer_params = {'is_training':train}, activation_fn=tf.identity))
  conv4 = lrelu(slim.convolution(conv3, 64, 3, stride=1, scope='conv4', normalizer_fn=slim.batch_norm,normalizer_params = {'is_training':train}, activation_fn=tf.identity))
  conv5 = lrelu(slim.convolution(conv4, 128, 3, stride=1, scope='conv5', normalizer_fn=slim.batch_norm,normalizer_params = {'is_training':train}, activation_fn=tf.identity))
  conv6 = lrelu(slim.convolution(conv5, 128, 3, stride=1, scope='conv6', normalizer_fn=slim.batch_norm,normalizer_params = {'is_training':train}, activation_fn=tf.identity))
  conv7 = lrelu(slim.convolution(conv6, 256, 3, stride=1, scope='conv7', normalizer_fn=slim.batch_norm,normalizer_params = {'is_training':train}, activation_fn=tf.identity))
  conv8 = lrelu(slim.convolution(conv7, 256, 3, stride=1, scope='conv8', normalizer_fn=slim.batch_norm,normalizer_params = {'is_training':train}, activation_fn=tf.identity))
  conv9 = lrelu(slim.convolution(conv8, 128, 3, stride=1, scope='conv9', normalizer_fn=slim.batch_norm,normalizer_params = {'is_training':train}, activation_fn=tf.identity))
  conv10 = lrelu(slim.convolution(conv9, 128, 3, stride=1, scope='conv10', normalizer_fn=slim.batch_norm,normalizer_params = {'is_training':train}, activation_fn=tf.identity))
  conv11 = lrelu(slim.convolution(conv10, 64, 1, stride=1, scope='conv11', normalizer_fn=slim.batch_norm, activation_fn=tf.identity))
  conv12 = lrelu(slim.convolution(conv11, 64, 1, stride=1, scope='conv12', normalizer_fn=slim.batch_norm,normalizer_params = {'is_training':train}, activation_fn=tf.identity))
  conv13 = lrelu(slim.convolution(conv12, 32, 1, stride=1, scope='conv13', normalizer_fn=slim.batch_norm,normalizer_params = {'is_training':train}, activation_fn=tf.identity))
  conv14 = lrelu(slim.convolution(conv13, 32, 1, stride=1, scope='conv14', normalizer_fn=slim.batch_norm,normalizer_params = {'is_training':train}, activation_fn=tf.identity))
  conv15 = lrelu(slim.convolution(conv14, 16, 1, stride=1, scope='conv15', normalizer_fn=slim.batch_norm,normalizer_params = {'is_training':train}, activation_fn=tf.identity))
  conv16 = lrelu(slim.convolution(conv15, 16, 1, stride=1, scope='conv16', normalizer_fn=slim.batch_norm,normalizer_params = {'is_training':train}, activation_fn=tf.identity))
  conv17 = lrelu(slim.convolution(conv16, 8, 1, stride=1, scope='conv17', normalizer_fn=slim.batch_norm,normalizer_params = {'is_training':train}, activation_fn=tf.identity))
  #if train: conv17 = tf.nn.dropout(conv17, 0.8)
  conv18 = (slim.convolution(conv17, 2, 1, stride=1, scope='conv18', activation_fn=tf.identity))
  #if train: conv18 = tf.nn.dropout(conv18, 0.8)

  a3 = tf.concat((inp,conv18),axis = 3)
  a3 = tf.reshape(a3,[-1,3])
  a2 = tf.matmul(a3,inv_conv_mat)
  a2 = tf.reshape(a2,[-1,32,32,3])

    
  return conv18,a2,inpp,Y
   
  



def complex_pokemon_model3(X,train=True):

    
  #X_input = tf.image.convert_image_dtype(X,tf.float32)
  X_input = X

  conv_mat = tf.constant(np.array([[0.299,0.587,0.114],[-0.14713,-0.2888,0.436],[0.615,-0.514999,-0.10001]]),dtype = tf.float32)
  inv_conv_mat = tf.constant(np.array([[1,0,1.13983],[1,-0.39465,-0.58060],[1,2.03211,0]]),dtype = tf.float32)

  X_input = tf.reshape(X_input,[-1,3])

  YUV = tf.matmul(X_input,conv_mat)
  YUV = tf.reshape(YUV,[-1,32,32,3])
  X_input = tf.reshape(X_input,[-1,32,32,3])
  UV = YUV[:,:,:,1:3]
  YY = YUV[:,:,:,0:1]

  conv1 = lrelu(slim.convolution(YY, 32, 3, stride=1, scope='conv1', normalizer_fn=slim.batch_norm, normalizer_params = {'is_training':train},activation_fn=tf.identity))
  conv2 = lrelu(slim.convolution(conv1, 32, 3, stride=1, scope='conv2', normalizer_fn=slim.batch_norm,normalizer_params = {'is_training':train}, activation_fn=tf.identity))
  conv3 = lrelu(slim.convolution(conv2, 64, 3, stride=1, scope='conv3', normalizer_fn=slim.batch_norm,normalizer_params = {'is_training':train}, activation_fn=tf.identity))
  conv4 = lrelu(slim.convolution(conv3, 64, 3, stride=1, scope='conv4', normalizer_fn=slim.batch_norm,normalizer_params = {'is_training':train}, activation_fn=tf.identity))
  conv5 = lrelu(slim.convolution(conv4, 128, 3, stride=1, scope='conv5', normalizer_fn=slim.batch_norm,normalizer_params = {'is_training':train}, activation_fn=tf.identity))
  conv6 = lrelu(slim.convolution(conv5, 128, 3, stride=1, scope='conv6', normalizer_fn=slim.batch_norm,normalizer_params = {'is_training':train}, activation_fn=tf.identity))
  conv7 = lrelu(slim.convolution(conv6, 256, 3, stride=1, scope='conv7', normalizer_fn=slim.batch_norm,normalizer_params = {'is_training':train}, activation_fn=tf.identity))
  conv8 = lrelu(slim.convolution(conv7, 256, 3, stride=1, scope='conv8', normalizer_fn=slim.batch_norm,normalizer_params = {'is_training':train}, activation_fn=tf.identity))
  conv9 = lrelu(slim.convolution(conv8, 128, 3, stride=1, scope='conv9', normalizer_fn=slim.batch_norm,normalizer_params = {'is_training':train}, activation_fn=tf.identity))
  conv10 = lrelu(slim.convolution(conv9, 128, 3, stride=1, scope='conv10', normalizer_fn=slim.batch_norm,normalizer_params = {'is_training':train}, activation_fn=tf.identity))
  conv11 = lrelu(slim.convolution(conv10, 128, 1, stride=1, scope='conv11', normalizer_fn=slim.batch_norm, activation_fn=tf.identity))
  conv12 = lrelu(slim.convolution(conv11, 64, 1, stride=1, scope='conv12', normalizer_fn=slim.batch_norm, activation_fn=tf.identity))
  conv13 = lrelu(slim.convolution(conv12, 64, 1, stride=1, scope='conv13', normalizer_fn=slim.batch_norm, activation_fn=tf.identity))
  #if train: conv17 = tf.nn.dropout(conv17, 0.8)
  conv18 = (slim.convolution(conv13, 400, 1, stride=1, scope='conv18', activation_fn=tf.nn.relu))
  #if train: conv18 = tf.nn.dropout(conv18, 0.8)


  #YUV_out = tf.concat((YY,conv18),axis = 3)
    #a2 = tf.image.hsv_to_rgb(a3)
 # a3 = tf.reshape(a3,[-1,3])
  #YUV_out = a3;
  #a2 = tf.matmul(a3,inv_conv_mat)
  #a2 = tf.reshape(a2,[-1,64,64,3])
  #RGB_out = a2;
  #a3 = tf.concat((conv18,inp),axis = 3)
  #a2 = tf.image.hsv_to_rgb(a3)

   # UV_out , RGB_out, UV_in, Y_in
  return conv18,YY, UV
   
  
def complex_pokemon_model4(X,train=True):

    
  #X_input = tf.image.convert_image_dtype(X,tf.float32)
  X_input = X

  conv_mat = tf.constant(np.array([[0.299,0.587,0.114],[-0.14713,-0.2888,0.436],[0.615,-0.514999,-0.10001]]),dtype = tf.float32)
  inv_conv_mat = tf.constant(np.array([[1,0,1.13983],[1,-0.39465,-0.58060],[1,2.03211,0]]),dtype = tf.float32)

  X_input = tf.reshape(X_input,[-1,3])

  YUV = tf.matmul(X_input,conv_mat)
  YUV = tf.reshape(YUV,[-1,32,32,3])
  X_input = tf.reshape(X_input,[-1,32,32,3])
  UV = YUV[:,:,:,1:3]
  YY = YUV[:,:,:,0:1]

  conv1 = lrelu(slim.convolution(YY, 32, 3, stride=1, scope='conv1', normalizer_fn=slim.batch_norm, normalizer_params = {'is_training':train},activation_fn=tf.identity))
  conv2 = lrelu(slim.convolution(conv1, 32, 3, stride=1, scope='conv2', normalizer_fn=slim.batch_norm,normalizer_params = {'is_training':train}, activation_fn=tf.identity))
  conv3 = lrelu(slim.convolution(conv2, 64, 3, stride=1, scope='conv3', normalizer_fn=slim.batch_norm,normalizer_params = {'is_training':train}, activation_fn=tf.identity))
  conv4 = lrelu(slim.convolution(conv3, 64, 3, stride=1, scope='conv4', normalizer_fn=slim.batch_norm,normalizer_params = {'is_training':train}, activation_fn=tf.identity))
  conv5 = lrelu(slim.convolution(conv4, 128, 3, stride=1, scope='conv5', normalizer_fn=slim.batch_norm,normalizer_params = {'is_training':train}, activation_fn=tf.identity))
  conv6 = lrelu(slim.convolution(conv5, 128, 3, stride=1, scope='conv6', normalizer_fn=slim.batch_norm,normalizer_params = {'is_training':train}, activation_fn=tf.identity))
  conv7 = lrelu(slim.convolution(conv6, 256, 3, stride=1, scope='conv7', normalizer_fn=slim.batch_norm,normalizer_params = {'is_training':train}, activation_fn=tf.identity))
  conv8 = lrelu(slim.convolution(conv7, 256, 3, stride=1, scope='conv8', normalizer_fn=slim.batch_norm,normalizer_params = {'is_training':train}, activation_fn=tf.identity))
  conv9 = lrelu(slim.convolution(conv8, 128, 3, stride=1, scope='conv9', normalizer_fn=slim.batch_norm,normalizer_params = {'is_training':train}, activation_fn=tf.identity))
  conv10 = lrelu(slim.convolution(conv9, 128, 3, stride=1, scope='conv10', normalizer_fn=slim.batch_norm,normalizer_params = {'is_training':train}, activation_fn=tf.identity))
  conv11 = lrelu(slim.convolution(conv10, 64, 1, stride=1, scope='conv11', normalizer_fn=slim.batch_norm, activation_fn=tf.identity))
  conv12 = lrelu(slim.convolution(conv11, 64, 1, stride=1, scope='conv12', normalizer_fn=slim.batch_norm,normalizer_params = {'is_training':train}, activation_fn=tf.identity))
  conv13 = lrelu(slim.convolution(conv12, 32, 1, stride=1, scope='conv13', normalizer_fn=slim.batch_norm,normalizer_params = {'is_training':train}, activation_fn=tf.identity))
  conv14 = lrelu(slim.convolution(conv13, 32, 1, stride=1, scope='conv14', normalizer_fn=slim.batch_norm,normalizer_params = {'is_training':train}, activation_fn=tf.identity))
  conv15 = lrelu(slim.convolution(conv14, 16, 1, stride=1, scope='conv15', normalizer_fn=slim.batch_norm,normalizer_params = {'is_training':train}, activation_fn=tf.identity))
  conv16 = lrelu(slim.convolution(conv15, 16, 1, stride=1, scope='conv16', normalizer_fn=slim.batch_norm,normalizer_params = {'is_training':train}, activation_fn=tf.identity))
  conv17 = lrelu(slim.convolution(conv16, 8, 1, stride=1, scope='conv17', normalizer_fn=slim.batch_norm,normalizer_params = {'is_training':train}, activation_fn=tf.identity))
  #if train: conv17 = tf.nn.dropout(conv17, 0.8)
  conv18_a = (slim.convolution(conv17, 256, 1, stride=1, scope='conv18_a', activation_fn=tf.tanh))
  conv18_b = (slim.convolution(conv17, 256, 1, stride=1, scope='conv18_b', activation_fn=tf.tanh))
  #if train: conv18 = tf.nn.dropout(conv18, 0.8)


  #YUV_out = tf.concat((YY,conv18),axis = 3)
    #a2 = tf.image.hsv_to_rgb(a3)
 # a3 = tf.reshape(a3,[-1,3])
  #YUV_out = a3;
  #a2 = tf.matmul(a3,inv_conv_mat)
  #a2 = tf.reshape(a2,[-1,64,64,3])
  #RGB_out = a2;
  #a3 = tf.concat((conv18,inp),axis = 3)
  #a2 = tf.image.hsv_to_rgb(a3)

   # UV_out , RGB_out, UV_in, Y_in
  return conv18_a,conv18_b,YY, UV



def simpler_model(gray_image,train = True):
  #bn = lambda x: slim.batch_norm(x, 'is_training' =train)
  conv1 = lrelu(slim.convolution(gray_image, 32, 3, stride=1, scope='conv1', normalizer_fn=slim.batch_norm, normalizer_params = {'is_training':train}, activation_fn=tf.identity))
  conv2 = lrelu(slim.convolution(conv1, 32, 3, stride=1, scope='conv2', normalizer_fn=slim.batch_norm, normalizer_params = {'is_training':train},activation_fn=tf.identity))
  conv3 = lrelu(slim.convolution(conv2, 64, 3, stride=1, scope='conv3', normalizer_fn=slim.batch_norm, normalizer_params = {'is_training':train},activation_fn=tf.identity))
  conv5 = lrelu(slim.convolution(conv3, 128, 3, stride=1, scope='conv5', normalizer_fn=slim.batch_norm, normalizer_params = {'is_training':train},activation_fn=tf.identity))
  conv12 = lrelu(slim.convolution(conv5, 64, 1, stride=1, scope='conv12', normalizer_fn=slim.batch_norm, normalizer_params = {'is_training':train},activation_fn=tf.identity))
  conv13 = lrelu(slim.convolution(conv12, 32, 1, stride=1, scope='conv13', normalizer_fn=slim.batch_norm, normalizer_params = {'is_training':train},activation_fn=tf.identity))
  conv14 = lrelu(slim.convolution(conv13, 32, 1, stride=1, scope='conv14', normalizer_fn=slim.batch_norm,normalizer_params = {'is_training':train}, activation_fn=tf.identity))
  conv15 = lrelu(slim.convolution(conv14, 16, 1, stride=1, scope='conv15', normalizer_fn=slim.batch_norm,normalizer_params = {'is_training':train}, activation_fn=tf.identity))
  conv16 = lrelu(slim.convolution(conv15, 16, 1, stride=1, scope='conv16', normalizer_fn=slim.batch_norm,normalizer_params = {'is_training':train}, activation_fn=tf.identity))
  conv17 = lrelu(slim.convolution(conv16, 8, 1, stride=1, scope='conv17', normalizer_fn=slim.batch_norm, normalizer_params = {'is_training':train},activation_fn=tf.identity))
  #if train: conv17 = tf.nn.dropout(conv17, 0.8)
  conv18 = (slim.convolution(conv17, 2, 1, stride=1, scope='conv18', activation_fn=tf.identity))
  #if train: conv18 = tf.nn.dropout(conv18, 0.8)
  
  return conv18

    