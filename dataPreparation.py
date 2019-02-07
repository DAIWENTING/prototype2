# -*- coding: utf-8 -*-
"""
Created on Tue Dec 26 10:58:55 2017

@author: wenting
"""
import numpy as np
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from sklearn.decomposition import PCA

import os
from keras.models import Model
#from keras.layers import Input, Dense
#from keras.utils import np_utils
import scipy.io as sio
#from scipy.misc import imread, imresize
import random
import pandas as pd

    
base_model = VGG16(weights='imagenet', include_top=True)
base_model.summary()
model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc2').output)

# change the path based on the setting of different cases
defFoldName = "./data/prototype2_data/SolderBridge"
defimgs = os.listdir(defFoldName)
defimgNum = len(defimgs)
features = np.zeros((defimgNum,4096),dtype = float,order = 'c')

# change the path based on the setting of different cases
norFoldName = "./data/prototype2_data/Normal_Type-2"
norimgs = os.listdir(norFoldName)
norimgNum = len(norimgs)
features2 = np.zeros((norimgNum,4096),dtype = float,order = 'c')

imgs_all = []


# extract features of defective samples
for i in range(defimgNum):
    img_path =  defFoldName + "/" + defimgs[i]
    imgs_all.append(img_path)
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features[i,:]= model.predict(x)
    #if i%50 == 0:
        #print(i)

# extract features of normal samples
for i in range(norimgNum):
    img_path2 = norFoldName + "/" + norimgs[i]
    imgs_all.append(img_path2)
    img2 = image.load_img(img_path2, target_size=(224, 224))
    x = image.img_to_array(img2)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features2[i,:]= model.predict(x)
    #if i%50 == 0:
        #print(i)

f = np.vstack((features,features2)) 

# create labels
label = np.ones((len(f),1),dtype = float)
label[:len(features)] = -1

# PCA dimension reduction 
pca = PCA(n_components=500)
pca.fit(f)
features_500 = pca.transform(f)



a = list(range(len(f)))
random.shuffle(a)

train_idx = a[:round(0.65 * len(f))]
test_idx = a[round(0.65 * len(f)):]

train_data = features_500[train_idx]
train_label = label[train_idx]

test_data = features_500[test_idx]
test_label = label[test_idx]

train_path = []
test_path = []

for i in range(len(train_idx)):
    train_path.append(imgs_all[train_idx[i]])
    
for i in range(len(test_idx)):
    test_path.append(imgs_all[test_idx[i]])
    
dic = {'train_data':train_data}
sio.savemat('train_data.mat',dic)

dic = {'train_label':train_label}
sio.savemat('train_label.mat',dic)

dic = {'test_data':test_data}
sio.savemat('test_data.mat',dic)

dic = {'test_label':test_label}
sio.savemat('test_label.mat',dic)

trainframe = pd.DataFrame({'ImagePath':train_path})
trainframe.to_csv("train_path.csv")

testframe = pd.DataFrame({'ImagePath':test_path})
testframe.to_csv("test_path.csv")
