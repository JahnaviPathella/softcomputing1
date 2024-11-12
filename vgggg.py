#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16, ResNet50
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense, Dropout

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# from spektral.layers import GCNConv, GlobalAvgPool
# from spektral.data import Graph
# from spektral.utils import sp_matrix_to_sp_tensor
# from spektral.data import Dataset, Loader
# import scipy.sparse as sp


# In[3]:


import warnings
warnings.filterwarnings('ignore')


# In[4]:


print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


# In[5]:


base_dir = '/kaggle/input/brain-tumor-mri-dataset'


# In[6]:


train_dir = os.path.join(base_dir, 'C:\Users\G VENKATA RAMANA\Downloads\archive (2)\Training')
test_dir = os.path.join(base_dir, 'C:\Users\G VENKATA RAMANA\Downloads\archive (2)\Testing')


# In[7]:


train_dir = os.path.join(base_dir, r'C:\Users\G VENKATA RAMANA\Downloads\archive (2)\Training')
test_dir = os.path.join(base_dir, r'C:\Users\G VENKATA RAMANA\Downloads\archive (2)\Testing')


# In[8]:


def create_dataframe(directory):
    filepaths = []
    labels = []
    folds = os.listdir(directory)
    for fold in folds:
        foldpath = os.path.join(directory, fold)
        if os.path.isdir(foldpath):
            filelist = os.listdir(foldpath)
            for fpath in filelist:
                fullpath = os.path.join(foldpath, fpath)
                labels.append(fold)
                filepaths.append(fullpath)
    return pd.DataFrame({'filepaths': filepaths, 'labels': labels})


# In[9]:


# Create DataFrames
train_df = create_dataframe(train_dir)

train_df


# In[10]:


test_df = create_dataframe(test_dir)

test_df


# In[11]:


# Split test_df into validation and test sets
val_df, test_df = train_test_split(test_df, test_size=0.5, random_state=43)

val_df


# In[12]:


batch_size = 32  # Increased batch size to utilize P100 GPU memory
image_size = (224, 224)


# In[13]:


train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=20,
                                   width_shift_range=0.1,
                                   height_shift_range=0.1,
                                   shear_range=0.1,
                                   zoom_range=0.1,
                                   horizontal_flip=True,
                                   fill_mode='nearest')


# In[14]:


valid_datagen = ImageDataGenerator(rescale=1./255)


# In[15]:


train_generator = train_datagen.flow_from_dataframe(
    train_df,
    x_col='filepaths',
    y_col='labels',
    target_size=image_size,
    class_mode='categorical',
    batch_size=batch_size,
    shuffle=True
)


# In[16]:


valid_generator = valid_datagen.flow_from_dataframe(
    val_df,
    x_col='filepaths',
    y_col='labels',
    target_size=image_size,
    class_mode='categorical',
    batch_size=batch_size,
    shuffle=False
)


# In[17]:


test_generator = valid_datagen.flow_from_dataframe(
    test_df,
    x_col='filepaths',
    y_col='labels',
    target_size=image_size,
    class_mode='categorical',
    batch_size=batch_size,
    shuffle=False
)


# In[18]:


# Get class indices and labels
g_dict = train_generator.class_indices
classes = list(g_dict.keys())


# In[19]:


# Get a batch of images and labels
images, labels = next(train_generator)


# In[20]:


# Plot images
plt.figure(figsize=(16, 16))
for i in range(16):
    plt.subplot(4, 4, i + 1)
    image = images[i]
    plt.imshow(image)
    index = np.argmax(labels[i])
    class_name = classes[index]
    plt.title(class_name, color='blue', fontsize=12)
    plt.axis('off')
plt.tight_layout()
plt.show()


# In[21]:


class_count = len(train_generator.class_indices)
print("Number of Classes:", class_count)


# In[22]:


# Load the VGG16 model without the top layers
base_model_vgg = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model_vgg.trainable = False  # Freeze the base model


# In[23]:


from tensorflow.keras import layers, models


# In[24]:


# Build the model
vgg_model = models.Sequential([
    base_model_vgg,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(64, activation='relu'),
    layers.Dense(class_count, activation='softmax')
])


# In[25]:


vgg_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
vgg_model.summary()


# In[25]:


epochs = 20

history_vgg16 = vgg_model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    validation_data=valid_generator,
    validation_steps=valid_generator.samples // batch_size,
    epochs=epochs
)


# In[26]:


# Evaluate on test data
loss_vgg16, accuracy_vgg16 = vgg_model.evaluate(test_generator)
print('VGG16 Test Loss:', loss_vgg16)
print('VGG16 Test Accuracy:', accuracy_vgg16)


# In[1]:


# Load the VGG16 model without the top layers
base_model_vgg = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model_vgg.trainable = False  # Freeze the base model


# In[ ]:




