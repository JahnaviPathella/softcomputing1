#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np
import pandas as pd
import seaborn as sns
sns.set_style('darkgrid')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras import regularizers

# Ignore Warnings
import warnings
warnings.filterwarnings("ignore")


# In[2]:


IMAGE_SIZE = [128,128]
train_data_dir = r'C:\Users\G VENKATA RAMANA\Downloads\archive (2)\Training'
test_data_dir = r'C:\Users\G VENKATA RAMANA\Downloads\archive (2)\Testing'

#train data preprocessing
filepaths = []
labels = []

folds = os.listdir(train_data_dir)
for fold in folds:
    foldpath = os.path.join(train_data_dir, fold)
    filelist = os.listdir(foldpath)
    for file in filelist:
        fpath = os.path.join(foldpath, file)
        
        filepaths.append(fpath)
        labels.append(fold)
Fseries = pd.Series(filepaths, name= 'filepaths')
Lseries = pd.Series(labels, name='labels')
traindata = pd.concat([Fseries, Lseries], axis= 1)
traindata


# In[3]:


#test data preprocessing
filepaths = []
labels = []
folds = os.listdir(test_data_dir)

for fold in folds:
    foldpath = os.path.join(test_data_dir,fold)
    filelist = os.listdir(foldpath) 
    for file in filelist:
        fpath = os.path.join(foldpath,file)
        filepaths.append (fpath)
        labels.append(fold)
Fseries = pd.Series(filepaths,name = 'filepaths')
Lseries = pd.Series(labels, name = 'labels')
testdata = pd.concat([Fseries,Lseries],axis='columns')
testdata


# In[4]:


#Splitting the testdata to validation and test data
valid_df,test_df = train_test_split(testdata,train_size=0.75,shuffle=True,random_state=123)


# In[5]:


#data augmentation
batch_size=32
channels = 3 
img_shape=(IMAGE_SIZE[0],IMAGE_SIZE[1],channels)  #128*128*3
tr_gen = ImageDataGenerator()
ts_gen = ImageDataGenerator()
train_gen = tr_gen.flow_from_dataframe(traindata, 
                                       x_col= 'filepaths', 
                                       y_col= 'labels',
                                       target_size= IMAGE_SIZE, 
                                       class_mode= 'categorical', 
                                       color_mode= 'rgb', 
                                       shuffle= True,
                                       batch_size= batch_size)
test_gen = tr_gen.flow_from_dataframe(test_df, 
                                      x_col= 'filepaths', 
                                      y_col= 'labels',
                                      target_size= IMAGE_SIZE, 
                                       class_mode= 'categorical', 
                                      color_mode= 'rgb', 
                                      shuffle= False ,
                                      batch_size= batch_size)
valid_gen = tr_gen.flow_from_dataframe(valid_df,
                                        x_col= 'filepaths',
                                        y_col= 'labels',
                                        target_size= IMAGE_SIZE,
                                        class_mode= 'categorical',
                                        color_mode= 'rgb',
                                        shuffle= True,
                                        batch_size= batch_size)


# In[6]:


#_______________________________Implementing transfer learning_______________________________

IMAGE_SIZE = (128, 128)
channels = 3
img_shape = (IMAGE_SIZE[0], IMAGE_SIZE[1], channels)
class_count = len(list(train_gen.class_indices.keys()))
base_model = MobileNetV2(include_top= False, weights= "imagenet", input_shape= img_shape, pooling= 'max')
# Freeze all layers except the last 10 for fine-tuning
for layer in base_model.layers[:-10]:
  layer.trainable = False
  
model = Sequential([
    base_model,
    BatchNormalization(axis= -1, momentum= 0.99, epsilon= 0.001),
    Dense(256, kernel_regularizer= regularizers.l2( 0.016),
          activity_regularizer= regularizers.l1(0.006),
          bias_regularizer= regularizers.l1(0.006),
          activation= 'relu'),
    Dropout(rate= 0.45, seed= 123),
    Dense(class_count, activation= 'softmax')
])

model.compile(Adamax(learning_rate= 0.001), loss= 'categorical_crossentropy', metrics= ['accuracy'])

model.summary()
Model: "sequential_1"
# Ignore Warnings
import warnings
warnings.filterwarnings("ignore")



# In[7]:


#_______________________________Training the model_______________________________

history = model.fit(x= train_gen, 
                    epochs= 20, 
                    verbose= 1, 
                    validation_data= valid_gen, 
                    shuffle= False)

warnings.filterwarnings("ignore")


# In[8]:


#_______________________________Model Evaluating_______________________________

# Define needed variables
tr_acc = history.history['accuracy']
tr_loss = history.history['loss']
val_acc = history.history['val_accuracy']
val_loss = history.history['val_loss']
index_loss = np.argmin(val_loss)
val_lowest = val_loss[index_loss]
index_acc = np.argmax(val_acc)
acc_highest = val_acc[index_acc]
Epochs = [i+1 for i in range(len(tr_acc))]
loss_label = f'best epoch= {str(index_loss + 1)}'
acc_label = f'best epoch= {str(index_acc + 1)}'
plt.figure(figsize= (20, 8))
plt.style.use('fivethirtyeight')

plt.subplot(1, 2, 1)
plt.plot(Epochs, tr_loss, 'r', label= 'Training loss')
plt.plot(Epochs, val_loss, 'g', label= 'Validation loss')
plt.scatter(index_loss + 1, val_lowest, s= 150, c= 'blue', label= loss_label)
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(Epochs, tr_acc, 'r', label= 'Training Accuracy')
plt.plot(Epochs, val_acc, 'g', label= 'Validation Accuracy')
plt.scatter(index_acc + 1 , acc_highest, s= 150, c= 'blue', label= acc_label)
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout
plt.show()


# In[9]:


#confusion matrix
train_score = model.evaluate(train_gen ,  verbose = 1)
valid_score = model.evaluate(valid_gen ,  verbose = 1)
test_score = model.evaluate(test_gen ,  verbose = 1)


preds = model.predict_generator(test_gen)
y_pred = np.argmax(preds , axis = 1)
g_dict = test_gen.class_indices
classes = list(g_dict.keys())
cm = confusion_matrix(test_gen.classes, y_pred)
cm 


import itertools
plt.figure(figsize= (10, 10))
plt.imshow(cm, interpolation= 'nearest', cmap= plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation= 45)
plt.yticks(tick_marks, classes)


thresh = cm.max() / 2.
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, cm[i, j], horizontalalignment= 'center', color= 'white' if cm[i, j] > thresh else 'black')

plt.tight_layout()
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()


# In[10]:


train_score = model.evaluate(train_gen, verbose=1)
valid_score = model.evaluate(valid_gen, verbose=1)
test_score = model.evaluate(test_gen, verbose=1)

# Use 'predict' instead of 'predict_generator'
preds = model.predict(test_gen)
y_pred = np.argmax(preds, axis=1)
g_dict = test_gen.class_indices  # No change here

# Confusion Matrix
classes = list(g_dict.keys())
cm = confusion_matrix(test_gen.classes, y_pred)

# Plot Confusion Matrix
import itertools
plt.figure(figsize=(10, 10))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)

# Labeling the matrix
thresh = cm.max() / 2.
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, cm[i, j], horizontalalignment='center', 
             color='white' if cm[i, j] > thresh else 'black')

plt.tight_layout()
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()


# In[11]:


# Classification Report
from sklearn.metrics import classification_report
print("Classification Report:")
print(classification_report(test_gen.classes, y_pred, target_names=classes))


# In[ ]:




