import os
import cv2
import copy
import shutil
import pickle
from PIL import Image
import numpy as np
import pandas as pd
from tqdm import tqdm
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model, load_model, Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.layers import Layer, Lambda, Dropout,Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D

""" The Targeted DeepFool attack """

def deepfool_targeted(image, model, num_classes=10, target = 4, overshoot=0.02, max_iter=100, shape=(224, 224, 3),count=0):
    image_array = np.array(image)
    image_norm = tf.cast(image_array/255.-0.5, tf.float32)
    image_norm = np.reshape(image_norm, shape)  # 28*28*1
    image_norm = image_norm[tf.newaxis, ...]  # 1*28*28*1

    f_image = model(image_norm).numpy().flatten()
    I = (np.array(f_image)).flatten().argsort()[::-1]
    I = I[0:num_classes]
    label = I[0]
    target_ind = np.where(I==target)[0][0]

    # print(label, "label")
    # print(I[target_ind],"target")

    input_shape = np.shape(image_norm)
    pert_image = copy.deepcopy(image_norm)
    w = np.zeros(input_shape)
    r_tot = np.zeros(input_shape)

    loop_i = 0
    x = tf.Variable(pert_image)
    fs = model(x)
    k_i = label

    #print(fs)

    def loss_func(logits, I, k):
        return logits[0, I[k]]
    cos_sim = 0
    prev_pert = 0
    r_tot = 0
    while k_i != target and loop_i < max_iter:

        one_hot_label_0 = tf.one_hot(label, num_classes)
        with tf.GradientTape() as tape:
            tape.watch(x)
            fs = model(x)
            loss_value = loss_func(fs, I, 0)
            #print(loss_value)
        grad_orig = tape.gradient(loss_value, x)

        with tf.GradientTape() as tape:
            tape.watch(x)
            fs = model(x)
            loss_value1 = loss_func(fs, I, target_ind)
            #print(loss_value1)
        cur_grad = tape.gradient(loss_value1, x)
        v1 = np.array(cur_grad).flatten()
        v2 = np.array(grad_orig).flatten()
        cos_sim = np.dot(v1,v1)/(np.linalg.norm(v1)*np.linalg.norm(v2))
        w = cur_grad - grad_orig

        f = (fs[0, I[target_ind]] - fs[0, I[0]]).numpy()

        pert = abs(f) #/ np.linalg.norm(w)
        

        r_i = (pert + 1e-4) * w / np.linalg.norm(w)         #  encountering divide by zero (Resolved)
        r_tot = np.float32(np.nan_to_num((r_tot + r_i),nan=0.7)) # nan issue  previously 0.9
        pert_image = image_norm + (1 + overshoot) * r_tot

        x = tf.Variable(pert_image)
        fs = model(x)
        k_i = np.argmax(np.array(fs).flatten())
        prev_pert = pert
        loop_i += 1

        # add perturbation file to the end of the code

        # print("Perturbation ",np.nan_to_num(pert,nan=100))  # need revision for nan (Resolved)
        # print("current index = ",k_i)

    r_tot = (1 + overshoot) * r_tot

    if k_i!=target and count<15:
        print("====ROUND",str(count+1),"====")
        cv2.imwrite('adversarial1.jpg',np.array((pert_image[0]+0.5)*255))
        round2 = cv2.imread('adversarial1.jpg',0)
        r_tot, loop_i, label, k_i,pert_image,cos_sim = deepfool_targeted(round2,mnist_model,target = target,num_classes=10,max_iter = 100,shape=(28,28,1),overshoot = 0.04,count=count+1)

    return r_tot, loop_i, label, k_i,pert_image,cos_sim



""" Generate adversarial images"""
def adv_error(X_test,Y_test,model,num_classes):
    adv = 0
    nat = 0

    for t in tqdm(range(len(X_test))):
        for target in tqdm(range(0,10)):
            if target == Y_test[t]:
                nat+=1
            else:
                _,_,_,k,pert,cos_sim = deepfool_targeted(X_test[t],model,target = target,num_classes=num_classes,max_iter=50,shape=(28,28,1),overshoot = 0.08,count=0)
                
                lab = np.argmax(model.predict(pert))
                cv2.imwrite('/content/1000_advs1/'+str(Y_test[t])+'_'+str(target)+'_'+str(t)+'.jpg',np.array((pert[0]+0.5)*255))
                if lab != target:
                    adv+=1
                if lab != Y_test[t]:
                    nat+=1
        print("\n=======> Generated "+str((t+1)*10)+" examples !\n")

    #nat+=len(X_test)
    return adv,nat

(X_train,Y_train),(X_test,Y_test) = tf.keras.datasets.mnist.load_data()
print("Loading MNIST Model....")
mnist_model = tf.keras.models.load_model('model/mnist_cnw.model')
print("Model loaded sucessfully !")
try:
		os.mkdir("/content/1000_advs1/")
except:
		shutil.rmtree("/content/1000_advs1/")
		os.mkdir("/content/1000_advs1/")


print("Generating adversarial examples.... (This may take a while)")
adv, nat = adv_error(X_test,Y_test,mnist_model,10)
print("Adversarial success rate: ",adv/len(X_test)*100)



