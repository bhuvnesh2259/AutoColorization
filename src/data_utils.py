import glob
import numpy as np
from PIL import Image
import tensorflow as tf
import math
import matplotlib.image as mpimg
import glob
import skimage.color as sk

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

#i = 1

#for filename in glob.iglob('/Users/kushaagragoyal/Desktop/CS231n/Project/tiny-imagenet-200/train/**/**/*.JPEG',recursive=True):
 #   print(filename)
  #  img = Image.open(filename).convert('LA')
   # img.save('grayscale/train/' + str(i) + '.png')
  #  i += 1
    
### Load Data Function ... 
#### Augment data function ...
### Normalise data function
### Visualise Data

def rgb2gray(rgb):
    a = np.dot(rgb[...,:3], [0.299, 0.587, 0.114])
    return a[:,:,:,np.newaxis]
def load_labels(file_name='../data/tiny-imagenet-200/words.txt', label_map={}):
    file1 = open(file_name)
    file2 = open('../data/tiny-imagenet-200/wnids.txt')
    subset = file2.readlines()
    tiny = []
    for val in subset:
        tiny.append(val.strip())
    lines = file1.readlines()
    #print("# of lines: ", len(lines))
    label_map = {}
    i=0
    for line in lines:
        abc = line.strip().split('\t')
        if abc[0] in tiny: 
            label_map[abc[0]] = abc[1]
    return label_map        
    
    
def load_data():
	input_image = []
	file_label_map = load_labels()
	labels = []    
	i = 0
	for filename in glob.iglob('../data/tiny-imagenet-200/train/**/*.JPEG',recursive=True):
		#print("file_name", filename)
		file_name = filename.split("/")[-1].split("_")[0]
		labels.append(file_label_map[file_name])       
		img = mpimg.imread(filename)
		if(img.shape == (64,64,3)):
			input_image.append(list(mpimg.imread(filename)))
		if(i%10000 == 0):
			print(i)
		if(i >600):
			break

		i+=1

	input_image = np.asarray(input_image)
	gray_data = rgb2gray(input_image)
	rand_indices = np.arange(input_image.shape[0])
	np.random.shuffle(rand_indices)
	lenn = int(4.0*input_image.shape[0]/5.0)
	X_train = gray_data[rand_indices[0:lenn]]
	X_test = gray_data[rand_indices[lenn:]]
	Y_train = input_image[rand_indices[0:lenn]]
	Y_test = input_image[rand_indices[lenn:]]
	labels_train = labels[0:lenn] 
	labels_test = labels[lenn:]    
	print("Shape of X_train, X_test, Y_train, Y_test:", len(labels_train), len(labels_test))
	return X_train, X_test, Y_train, Y_test, labels_train, labels_test

def augment_image(image):
	image = tf.image.random_flip_left_right(img)
	image = tf.image.random_brightness(img,max_delta=63)
	image = tf.image.random_contrast(img,lower=0.2, upper=1.8)
	return image

def augment_data(data):
	return tf.map_fn(augment_image,data)

def normalise_train(image_data):
	mean_image = np.mean(image_data,axis = 0,dtype = 'float32')
	std_image = np.sqrt(np.var(image_data,axis = 0,dtype = 'float32'))
	image_data = (image_data - mean_image)/std_image
	return image_data,mean_image,std_image

def normalise_test(image,mean_image,std_image):
	image = (image - mean_image)/std_image
	return image



def rgb2lab(rgb):
	return sk.rgb2lab(rgb)

def lab2rgb(lab):
	return sk.lab2rgb(lab)


'''
var  = tf.trainable_variables()
var = [v.name for v in tf.trainable_variables() ]
print (var)
numpy_val = sess.run([var[0]])
weights = np.asarray(numpy_val[0])
print (weights)

print ("Observe that weights are close to [0.299,0.587,0.114] , hence it learns rgb to gray conversion")
'''



