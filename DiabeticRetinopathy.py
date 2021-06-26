from tensorflow.keras.layers import Dense, Flatten,Dropout,BatchNormalization
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.python.keras.models import Sequential
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.applications import ResNet152
from tensorflow.keras.applications import ResNet101
from tensorflow.keras.applications import ResNet101V2
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
%matplotlib inline
import numpy as np

path=\"C:/Users/busra/veri_manipulasyonu1/TRAIN_IMAGE\"

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
     path
     labels = 'inferred'
     label_mode='int'
     seed=123
     color_mode ='rgb'
     image_size=(224, 224)
     batch_size=64
     shuffle=True
     )
for image, label in train_ds.take(1):
    print(label)
