#!/usr/bin/env python
# coding: utf-8

#   # Name:Saad,Mohammad
# 
#   # Student no :300267006                                                  
#  
#  #  Assignment (2)
# 
#   # ELG7186[EI] Learning-Based Computer Vision 

# In[1]:


import numpy as np
import os
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import skimage.io as skio
from skimage.util import crop
from skimage.color import rgb2gray
from skimage.transform import resize
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix,ConfusionMatrixDisplay
from imblearn.under_sampling import RandomUnderSampler 
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from skimage import filters, feature
from skimage.feature import match_template
#if you don't have the imblearn pkg installed uncomment the following command
#!pip install -U imbalanced-learn


# ## 1.2 Image Preprocessing 
# 
# 
# 

# **A function with the name (load_labe_resize) is written to load data and put each on the suitable category,besides that resizing was made to all images to be on the size (32,32)** 

# In[2]:


image_directory_training="./textures/training"
image_directory_testing="./textures/testing"
files_training=os.listdir(image_directory_training)
files_training=os.listdir(image_directory_training)
def img_process(path):
    img_read=skio.imread(path) # to read each img
    img_resized= resize(img_read,(32,32),preserve_range=True,anti_aliasing=True,order=0)#.reshape(-1,1) #to resize the images to 40,60 and the make them 1D
    return img_resized 
def load_label_resize(image_directory):
    files=os.listdir(image_directory)
    x_canvas=[]
    x_cushion=[]
    x_linsseed=[]
    x_sand=[]
    x_seat=[]
    x_stone=[]
    for i in files : #to open each file and read images
        path=os.path.join(image_directory,i)
        if i == "canvas1":
            for img in os.listdir(path):
                x_canvas.append(img_process(os.path.join(path,img)))
        elif i == "cushion1":
            for img in os.listdir(path):
                x_cushion.append(img_process(os.path.join(path,img)))
        
        elif i == "linsseeds1":
            for img in os.listdir(path): 
                x_linsseed.append(img_process(os.path.join(path,img)))
        
        elif i == "sand1":
            for img in os.listdir(path):
                x_sand.append(img_process(os.path.join(path,img)))
            
            
        elif i == "seat2":
            for img in os.listdir(path):
                x_seat.append(img_process(os.path.join(path,img)))
                
        elif i == "stone1":
            for img in os.listdir(path):
                x_stone.append(img_process(os.path.join(path,img)))
                
    return np.array(x_canvas),np.array(x_cushion),np.array(x_linsseed),np.array(x_sand),np.array(x_seat),np.array(x_stone)
  


# Reading the **trainig data** and **testing data** using the custom-made function described above

# In[3]:


x_canvas,x_cushion,x_linsseed,x_sand,x_seat,x_stone=load_label_resize(image_directory_training)#training
x_canvas_t,x_cushion_t,x_linsseed_t,x_sand_t,x_seat_t,x_stone_t=load_label_resize(image_directory_testing)#testing


# **A function was designed to pair images to form our data set<br>
# the function combine images to construct different pairs with size (:,32,32,2)**

# In[4]:



def pairs_combination(x1,x2):
    array=np.empty([32,32,2])#some sort of place holder
    t=[]
    if (x1.sum()).sum()==(x2.sum()).sum():#if the two arguments were similar
        for i in range(0,x1.shape[0]):
            for j in range (i+1,x1.shape[0]):
                array[:,:,0]=x1[i]
                array[:,:,1]=x1[j]
                t.append(array.copy())
               
    else:#if the classes were different
        for i in range(0,x1.shape[0]):
            for j in range(0,x2.shape[0]):
                array[:,:,0]=x1[i]
                array[:,:,1]=x2[j]
                t.append(array.copy())
               
        
     
    
    positive=np.array(t) #output shape (:,32,32,2)
    return positive


# Combining the positive class exampels for training and testing :

# In[5]:


#training
canvas=pairs_combination(x_canvas,x_canvas)
cushion=pairs_combination(x_cushion,x_cushion)
linsseed=pairs_combination(x_linsseed,x_linsseed)
sand=pairs_combination(x_sand,x_sand)
seat=pairs_combination(x_seat,x_seat)
stone=pairs_combination(x_stone,x_stone)

#testing
canvas_t=pairs_combination(x_canvas_t,x_canvas_t)
cushion_t=pairs_combination(x_cushion_t,x_cushion_t)
linsseed_t=pairs_combination(x_linsseed_t,x_linsseed_t)
sand_t=pairs_combination(x_sand_t,x_sand_t)
seat_t=pairs_combination(x_seat_t,x_seat_t)
stone_t=pairs_combination(x_stone_t,x_stone_t)


# Merging all the exampels :

# In[6]:


positive_pairs_training=np.vstack((canvas,cushion,linsseed,sand,seat,stone))
positive_pairs_testing=np.vstack((canvas_t,cushion_t,linsseed_t,sand_t,seat_t,stone_t))


# Combining the negative class exampels for training and testing :

# In[7]:


#training
canvas_cushion=pairs_combination(x_canvas,x_cushion)
canvas_linsseed=pairs_combination(x_canvas,x_linsseed)
canvas_sand=pairs_combination(x_canvas,x_sand)
canvas_seat=pairs_combination(x_canvas,x_seat)
canvas_stone=pairs_combination(x_canvas,x_stone)
cushion_linsseed=pairs_combination(x_cushion,x_linsseed)
cushion_sand=pairs_combination(x_cushion,x_sand)
cushion_seat=pairs_combination(x_cushion,x_seat)
cushion_stone=pairs_combination(x_cushion,x_stone)
linsseed_sand=pairs_combination(x_linsseed,x_sand)
linsseed_seat=pairs_combination(x_linsseed,x_seat)
linsseed_stone=pairs_combination(x_linsseed,x_stone)
sand_seat=pairs_combination(x_sand,x_seat)
sand_stone=pairs_combination(x_sand,x_stone)
seat_stone=pairs_combination(x_seat,x_stone)

#testing
canvas_cushion_t=pairs_combination(x_canvas_t,x_cushion_t)
canvas_linsseed_t=pairs_combination(x_canvas_t,x_linsseed_t)
canvas_sand_t=pairs_combination(x_canvas_t,x_sand_t)
canvas_seat_t=pairs_combination(x_canvas_t,x_seat_t)
canvas_stone_t=pairs_combination(x_canvas_t,x_stone_t)
cushion_linsseed_t=pairs_combination(x_cushion_t,x_linsseed_t)
cushion_sand_t=pairs_combination(x_cushion_t,x_sand_t)
cushion_seat_t=pairs_combination(x_cushion_t,x_seat_t)
cushion_stone_t=pairs_combination(x_cushion_t,x_stone_t)
linsseed_sand_t=pairs_combination(x_linsseed_t,x_sand_t)
linsseed_seat_t=pairs_combination(x_linsseed_t,x_seat_t)
linsseed_stone_t=pairs_combination(x_linsseed_t,x_stone_t)
sand_seat_t=pairs_combination(x_sand_t,x_seat_t)
sand_stone_t=pairs_combination(x_sand_t,x_stone_t)
seat_stone_t=pairs_combination(x_seat_t,x_stone_t)


# Merging all the exampels :

# In[8]:



negative_pairs_training=np.vstack((canvas_cushion,canvas_linsseed,canvas_sand,canvas_seat,canvas_stone,cushion_linsseed,cushion_sand,cushion_seat,cushion_stone,linsseed_sand,linsseed_seat,linsseed_stone,sand_seat,sand_stone,seat_stone))
negative_pairs_testing=np.vstack((canvas_cushion_t,canvas_linsseed_t,canvas_sand_t,canvas_seat_t,canvas_stone_t,cushion_linsseed_t,cushion_sand_t,cushion_seat_t,cushion_stone_t,linsseed_sand_t,linsseed_seat_t,linsseed_stone_t,sand_seat_t,sand_stone_t,seat_stone_t))


# In[9]:


print('the number of the positive examples used for training is {}'.format(positive_pairs_training.shape[0]))
print('the number of the negative examples used for training is {}'.format(negative_pairs_training.shape[0]))
print('the total number of training pairs will equal {}'.format(positive_pairs_training.shape[0]+negative_pairs_training.shape[0]))
print('---------------------------------------------------------------------')
print('the number of the positive examples used for testing is {}'.format(positive_pairs_testing.shape[0]))
print('the number of the negative examples used for testing is {}'.format(negative_pairs_testing.shape[0]))
print('the total number of testing pairs will equal {}'.format(positive_pairs_testing.shape[0]+negative_pairs_testing.shape[0]))


# As noticed from above that the total number of pairs used for training was calculated successfuly as expected considering the relation **N*(N-1)/2** >>> (180* 179)/2=16110<br>
# <br>
# **The data is imbalanced in a very observable way which we can deal with using sub-sampling of negative class**

# ## 1.3 Image Matching 
# 

# A function named (matchingImages) was created to match images using cross-correlation (cc), convolution (conv) or sum of squared differences (ssd).<br>
# the function recieves image pair as input arguments as well as the method ,and a bool value to normalize or not.<br>
# the function will return the score value<br>

# In[10]:


def  matchingImages (imageA, imageB, method='cc', normalize=False) :
    if normalize==True:
        imageA=(imageA-np.mean(imageA))
        imageB=(imageB-np.mean(imageB))
        std_A=((np.square((imageA-np.mean(imageA)))).sum()).sum()
        std_B=((np.square((imageB-np.mean(imageB)))).sum()).sum()
    if method == 'cc':
        if normalize==True:
            cc=((np.multiply(imageA,imageB)).sum()).sum()/np.sqrt(std_A*std_B)
        else:
            cc=((np.multiply(imageA,imageB)).sum()).sum()
        return cc
        
    elif method == 'conv':
        imageB=np.flip(np.flip(imageB,1),0)
        if normalize==True:
            conv=((np.multiply(imageA,imageB)).sum()).sum() #using zero mean matrices
        else:
            conv=((np.multiply(imageA,imageB)).sum()).sum()
        
        return conv
        
        
        
    elif method == 'ssd':
        if normalize==True:
            ssd=(np.square((imageA/np.std(imageA))-(imageB/np.std(imageB))).sum()).sum()#normalized by subtracting mean and dividing by std
        else:
            ssd=(np.square(imageA-imageB).sum()).sum()
        return ssd
    
    


# -The normalization used in "ssd" method is made by subtracting the mean and then dividing by std of the values to ensure the new values are between [ 1 and -1 ]. <br>
# it may help in finding better threshold because the individual diffrences in brightness(pixel intensities) will be all rescaled and the the method can be more generalized for different inputs.
# 

# The training data is splitted to training and validation [0.8-0.2] respectively<br>
# the splitting used stratifing to ensure an equale ratio of positive and negative examples<br>

# In[11]:


X=np.vstack((positive_pairs_training,negative_pairs_training))
y_train_1=np.ones_like(positive_pairs_training[:,0,0,0])
y_train_2=np.zeros_like(negative_pairs_training[:,0,0,0])
y=np.hstack((y_train_1,y_train_2))
X_train, X_valid, y_train, y_valid = train_test_split(X, y, stratify=y, test_size=0.2, random_state=1)


# To find a suitable value for threshold , a function was designed to run over all the exmaples and the scores were collected in a vector to calculate the mean and standard deviation 

# In[12]:


def find_threshold(data,method,normalize):
    y=[]
    for i in range (data.shape[0]):
        y.append(matchingImages(data[i,:,:,0],data[i,:,:,1],method,normalize=normalize))
    y=np.array(y)
    mean=np.mean(y)
    std=np.std(y)
    
    return mean , std
         


# The positive examples were seperated from negative ones to be used in finding the threshold value

# In[13]:


positive=np.where(y_train==1)
Train_positive=X_train[positive].copy()
negative=np.where(y_train==0)
Train_negative=X_train[negative].copy()


# The upcoming cell are used for estimating resonable values for the threshold <br>
# the used method is built on kowing the mean and std for positive exampels and negative exampels scores and then setting the threshold accordingly<br>
# **The scores are so similar and the methods can't distinguish between classes clearly as noticed in the mean and std values below** 

# The tuning process was experimented on 'ssd' method both normalized and not in the cell below :

# In[14]:


mean_positive_ssd_false ,std_positive_ssd_false =find_threshold(Train_positive,'ssd',False)
mean_negative_ssd_false ,std_negative_ssd_false =find_threshold(Train_negative,'ssd',False)
print('for ssd without normalization the mean score for positive exampels is {} with std {} and the mean for negative is {} with std {}'.format(mean_positive_ssd_false ,std_positive_ssd_false,mean_negative_ssd_false ,std_negative_ssd_false))
threshold_ssd_false=mean_negative_ssd_false-std_negative_ssd_false
print('for ssd without normalization the threshold for being poisitve is to be less than {}'.format(threshold_ssd_false))
mean_positive_ssd_true ,std_positive_ssd_true =find_threshold(Train_positive,'ssd',True)
mean_negative_ssd_true ,std_negative_ssd_true =find_threshold(Train_negative,'ssd',True)
print('for ssd with normalization the mean score for positive exampels is {} with std {} and the mean for negative is {} with std {}'.format(mean_positive_ssd_true ,std_positive_ssd_true,mean_negative_ssd_true ,std_negative_ssd_true))
threshold_ssd_true=mean_negative_ssd_true-std_negative_ssd_true
print('for ssd with normalization the threshold for being poisitve is to be less than {}'.format(threshold_ssd_true))


# The tuning process was experimented on 'cc' method both normalized and not in the cell below :

# In[15]:


mean_positive_cc_false ,std_positive_cc_false =find_threshold(Train_positive,'cc',False)
mean_negative_cc_false ,std_negative_cc_false =find_threshold(Train_negative,'cc',False)
print('for cc without normalization the mean score for positive exampels is {} with std {} and the mean for negative is {} with std {}'.format(mean_positive_cc_false ,std_positive_cc_false,mean_negative_cc_false ,std_negative_cc_false))
threshold_cc_false=mean_negative_cc_false+std_negative_cc_false
print('for cc without normalization the threshold for being poisitve is to be more than {}'.format(threshold_cc_false))
mean_positive_cc_true ,std_positive_cc_true =find_threshold(Train_positive,'cc',True)
mean_negative_cc_true ,std_negative_cc_true =find_threshold(Train_negative,'cc',True)
print('for cc with normalization the mean score for positive exampels is {} with std {} and the mean for negative is {} with std {}'.format(mean_positive_cc_true ,std_positive_cc_true,mean_negative_cc_true ,std_negative_cc_true))
threshold_cc_true=mean_negative_cc_true+std_negative_cc_true
print('for cc with normalization the threshold for being poisitve is to be more than {}'.format(threshold_cc_true))


# The tuning process was experimented on 'conv' method both normalized and not in the cell below :

# In[16]:


mean_positive_conv_false ,std_positive_conv_false =find_threshold(Train_positive,'conv',False)
mean_negative_conv_false ,std_negative_conv_false =find_threshold(Train_negative,'conv',False)
print('for conv without normalization the mean score for positive exampels is {} with std {} and the mean for negative is {} with std {}'.format(mean_positive_conv_false ,std_positive_conv_false,mean_negative_conv_false ,std_negative_conv_false))
threshold_conv_false=mean_negative_conv_false-std_negative_conv_false
print('for conv without normalization the threshold for being poisitve is to be less than {}'.format(threshold_conv_false))
mean_positive_conv_true ,std_positive_conv_true =find_threshold(Train_positive,'conv',True)
mean_negative_conv_true ,std_negative_conv_true =find_threshold(Train_negative,'conv',True)
print('for conv with normalization the mean score for positive exampels is {} with std {} and the mean for negative is {} with std {}'.format(mean_positive_conv_true ,std_positive_conv_true,mean_negative_conv_true ,std_negative_conv_true))
threshold_conv_true=mean_negative_conv_true-std_negative_conv_true
print('for conv with normalization the threshold for being poisitve is to be less than {}'.format(threshold_conv_true))


# **The function designed below uses the score values from the function(matchingImages) and then classifies based on the above tuned thresholds**

# In[17]:


def classify_score(score,method,normalize):
    global threshold_ssd_false,threshold_ssd_true,threshold_cc_false,threshold_cc_true,threshold_conv_false,threshold_conv_true
    if method == 'ssd':
        if normalize==True:
            if score <= threshold_ssd_true:
                return True
            else:
                return False
        else:
             if score <= threshold_ssd_false:
                return True
             else:
                return False
    elif method =='cc':
        if normalize==True:
            if score >= threshold_cc_true:
                return True
            else:
                return False
        else:
             if score >= threshold_cc_false:
                return True
             else:
                return False
    
    elif method =='conv':
        if normalize==True:
            if score <= threshold_conv_true:
                return True
            else:
                return False
        else:
             if score <= threshold_conv_false:
                return True
             else:
                return False
    
        
            
    


# **Another function was designed to predict and evaluate the model**<br>
# this function was called 6-times(using the validation set ) one for each classifier

# In[18]:


def classifier_evaluate(data_x,data_y,method,normalize):
    y_predict=[]
    for i in range(data_x.shape[0]):
        score=matchingImages (data_x[i,:,:,1],data_x[i,:,:,0] , method=method, normalize=normalize)
        y_predict.append(classify_score(score,method,normalize))
    y_predict=np.array(y_predict)
    y_predict.shape
    print(classification_report(data_y, y_predict, target_names=['not_match','match']))
    cm=confusion_matrix(data_y,y_predict)
    ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=['not-matching','matching']).plot()
    return y_predict


# **ssd not normalized:**

# In[19]:


y_predict_ssd_false=classifier_evaluate(X_valid,y_valid,'ssd',False)
    


# **ssd normalized:**

# In[20]:


y_predict_ssd_true=classifier_evaluate(X_valid,y_valid,'ssd',True)


# **cross convolution not normalized:**

# In[21]:


y_predict_cc_false=classifier_evaluate(X_valid,y_valid,'cc',False)


# **cross correlation normalized:<br>**
# one can notice that normalized cc performs better in matching texture detection 

# In[22]:


y_predict_cc_true=classifier_evaluate(X_valid,y_valid,'cc',True)


# **convolution not normalized :**

# In[23]:


y_predict_conv_false=classifier_evaluate(X_valid,y_valid,'conv',False)


# **convolution normalized :**

# In[24]:


y_predict_conv_true=classifier_evaluate(X_valid,y_valid,'conv',True)


# **After evaluating each classifier we can notice that they all have almost the same accuracy and performance over the data validation set**

# The training data is highly imbalanced ,therefore we need to subsampel the majority class which is the not matching class in our case

# In[25]:


data_before=np.bincount((y_train==1.0))
print('the number of positive exampels in the training set is {} and the negative exampels are {}'.format(data_before[1],data_before[0]))


# In[26]:


X_train_flat=X_train.reshape(-1,2048)
X_valid_flat=X_valid.reshape(-1,2048)


# In the above cell the data was flattened in order to be ready for training the MLP classfier<br>

# In[27]:


rus = RandomUnderSampler(sampling_strategy=0.6,random_state=42)
X_res, y_res = rus.fit_resample(X_train_flat, y_train)
data_after=np.bincount((y_res==1.0))
print('the number of positive exampels in the new training set is {} and the negative exampels are {}'.format(data_after[1],data_after[0]))


# **Using Imblearn library ready made function for under sampeling the data to have a ratio of 0.6 between positive to negative exampels**

# The MLP model was trained with the under-sampeled data and the results on the validation set are shown below<br>
# It looks like that the best model that we can have will be a model which predicts only one class(the majority one) <br>

# In[28]:


nn_clf = MLPClassifier(max_iter=100,solver='sgd',random_state=1,verbose='true')
nn_clf.fit(X_res,y_res)

y_pred = nn_clf.predict(X_valid_flat)
print(classification_report(y_valid, y_pred, target_names=['not_match','match']))
cm=confusion_matrix(y_valid,y_pred)
ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=['not-matching','matching']).plot()


# **As noticed from the figure above that there is no prediction in matching class , not even a wrong one**

# ## 1.5 Classification Comparison
# At first we will combine the testing data in a way similar to the training data:

# In[29]:


X_test=np.vstack((positive_pairs_testing,negative_pairs_testing))
y_test_1=np.ones_like(positive_pairs_testing[:,0,0,0])
y_test_2=np.zeros_like(negative_pairs_testing[:,0,0,0])
y_test=np.hstack((y_test_1,y_test_2))
X_test_flat=X_test.reshape(-1,2048)


# In[30]:


print('the number of sampels in testing data is {}'.format(X_test.shape[0]))


# At first we will use the testing data with the MLP model:

# In[31]:


y_pred = nn_clf.predict(X_test_flat)
print(classification_report(y_test, y_pred, target_names=['not_match','match']))
cm=confusion_matrix(y_test,y_pred)
ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=['not-matching','matching']).plot()


# As mentioned before , the model is predicting only one class,the not-matching class<br>
# **Then We will test the classifiers in  [ 1.3 ]  to compare the performance:**

# In[32]:


y_predict_ssd_false=classifier_evaluate(X_test,y_test,'ssd',False)


# In[33]:


y_predict_ssd_true=classifier_evaluate(X_test,y_test,'ssd',True)


# In[34]:


y_predict_cc_false=classifier_evaluate(X_test,y_test,'cc',False)


# In[35]:


y_predict_cc_true=classifier_evaluate(X_test,y_test,'cc',True)


# In[36]:


y_predict_conv_false=classifier_evaluate(X_test,y_test,'conv',False)


# In[37]:


y_predict_conv_true=classifier_evaluate(X_test,y_test,'conv',True)


# By looking at the previous data regarding models performance,one can notice that:<br>
# -The **MLP** model has the highest accuracy among all but it has zero performance on matching exampels and it got that high number of accuracy because of data imbalance.<br>
# *it can be said that MLP cannot be generalized and it's actualy similar to a function that return always False regardless the data*<br>
# -The **ssd without normalization** can predict 90 correct matching exampel from 270 with precision 30% ,however the total accuracy 78%<br>
# *it's better and more generalized than MLP and it detect  a small portion from the positive exampels correctly*<br>
# -The **ssd witht normalization** can predict 98 correct matching exampel from 270 with precision 31% ,however the total accuracy is still the same as without normalization 78% , because it's performance on negative exampels was slightly worse than without normalization<br>
# *it can be considered similar to ssd without normalization as there is no huge difference except for the sligh better performance on positive classes *<br>
# -The **cc without normalization** managed to predict only 71 one positive exampel which is considered low ,on the other hand it performed better on the negative examples with high precision and recall.The overall accuracy of the model is 80% and this number can't be trusted because of the imbalance in data.<br>
# *it can't be thought of as an improvement from the previous models and it can be also considerd slightly worse*<br>
# -The **cc with normalization** got the same scores exactly like **ssd witht normalization**<br>
# -The **conv without normalization** has the lowest scores(before MLP) on positive examples and highest scores on negative ones(after MLP)-recall 91%- with total accuarcy 80%<br>
# -The **conv with normalization** is similar or slightly less in performance than **cc without normalization**<br>
# <br>
# 
# From those results , it can be concluded that **MLP** and **convolution without normalization** are the worst and can't be generalized<br>
# The best ones (among others) are **cc with normalization** and **ssd witht normalization** <br> 
# 
# 

# ## 1.6 Feature Engineering
# In this section two methods used and tested seprately to process the images before training the **MLP**<br>
# <br>
# -The **first one** is built on the idea of template matching by using a slice from one image and applying cross-correlation with the other image in the same pair, the return will depend on the size of the slice (window or kernel) taken from the first image in the pair <br>
# *A function was built with the name "featuer_process_cc" to extract 25 value as the result of applying this process on image pairs*<br>
# Those values are our new features to train the model

# In[38]:


def featuer_process_cc(data):
    result=[]
    for i in range(data.shape[0]):
        A=np.copy(data[i,2:30,2:30,1])# size of window
        B= match_template(data[i,:,:,0], A)# normalized cross correlation
        result.append(B)
    result=np.array(result)
    return result
    
        
    


# The **second** method  calculates the **histogram** over 32 bin in order to have 32 feature to train the model with

# In[39]:


def featuer_process_hist(data):
    result=[]
    for i in range(data.shape[0]):
        B=np.histogram(data[i,:],bins=32)
        result.append(B[0])
    result=np.array(result)
    return result


# The first method was applied first on all the dataset we have (Training, validation ,Testing)

# In[40]:



New_train=featuer_process_cc(X_train)
New_train_flat=New_train.reshape(-1,25)


# In[41]:


New_valid=featuer_process_cc(X_valid)

New_valid_flat=New_valid.reshape(-1,25)


# In[42]:


New_test=featuer_process_cc(X_test)
New_test_flat=New_test.reshape(-1,25)


# In[43]:


rus2 = RandomUnderSampler(sampling_strategy=0.6,random_state=42)
X_res_new, y_res_new = rus2.fit_resample(New_train_flat, y_train)
data_after=np.bincount((y_res_new==1.0))
print('the number of positive exampels in the new training set is {} and the negative exampels are {}'.format(data_after[1],data_after[0]))


# Now the **MLP** will be trained with the new **training** data(after featuer extraction and under-sampling)

# In[44]:


nn_clf2 = MLPClassifier(max_iter=200,solver='sgd',random_state=1,verbose='true')
nn_clf2.fit(X_res_new,y_res_new)
y_pred2 = nn_clf2.predict(New_valid_flat)
print(classification_report(y_valid, y_pred2, target_names=['not_match','match']))
cm2=confusion_matrix(y_valid,y_pred2)
ConfusionMatrixDisplay(confusion_matrix=cm2,display_labels=['not-matching','matching']).plot()


# The results with the **cc method** show large improvement compared to **MLP with original data** and they also show slight improvement compared to conventional normalized cross correlation used in  **1.3** (which gave the best results back then)<br>
# 
# The model now can predict positive classes(which MLP couldn't do before) with higher precision and accuracy than the methods used in **1.3** 
# 

# Now the **second method(histogram method)** will applied on all the dataset (Training, validation ,Testing)

# In[45]:


New_train_hist=featuer_process_hist(X_train_flat)
New_valid_hist=featuer_process_hist(X_valid_flat)
New_test_hist=featuer_process_hist(X_test_flat)


# In[46]:


rus3 = RandomUnderSampler(sampling_strategy=0.6,random_state=42)
X_res_hist, y_res_hist= rus3.fit_resample(New_train_hist, y_train)
data_after=np.bincount((y_res_hist==1.0))
print('the number of positive exampels in the new training set is {} and the negative exampels are {}'.format(data_after[1],data_after[0]))


# Now the **MLP** will be trained with the new **training** data(after featuer extraction and under-sampling)

# In[47]:


nn_clf3 = MLPClassifier(max_iter=200,solver='sgd',random_state=1,verbose='true')
nn_clf3.fit(X_res_hist,y_res_hist)
y_pred3 = nn_clf3.predict(New_valid_hist)
print(classification_report(y_valid, y_pred3, target_names=['not_match','match']))
cm3=confusion_matrix(y_valid,y_pred3)
ConfusionMatrixDisplay(confusion_matrix=cm3,display_labels=['not-matching','matching']).plot()


# As observed from the validation data ,the metrics reveal that this method has helped the MLP to do better than the first method.<br>
# The model now can detect large number of positive exampels with high recall and better precision than any method used in this notebook ,the model with this method is more genralized and can be considered as the best so far<br>

# ## 1.7 Discussion on Feature Engineering

# In this section we will examine the MLP model (the one which was trained using histogram data) using the test data to ensure that the method of histogram really helped in improving the model<br>

# The method depends on extracting the histogram of image pairs and then feed it to the network as featuers<br>
# <br>
# If the images are similar the histogram content of them will be also similar as they share approximately the same pixels intensity and distribution<br>
# and the opposite happens with not-matching images as they represent different texture which means different intensities and distributions.<br>
# 
# <br>
# This can make featuers more understandable by the model,and it can now learn the difference in a more obvious way ,so we can expect that the model will be able to detect positive exampels as it failed to do so in 1.4
# 
# 

# In[48]:


y_pred4 = nn_clf3.predict(New_test_hist)
print(classification_report(y_test, y_pred4, target_names=['not_match','match']))
cm3=confusion_matrix(y_test,y_pred4)
ConfusionMatrixDisplay(confusion_matrix=cm3,display_labels=['not-matching','matching']).plot()


# The results are still  higher than other models tested  in **1.5**
# Which proves that the histogram method is succsseful and it managed to boost the abilities of the MLP to learn how to distinguish between matched and not matched images
