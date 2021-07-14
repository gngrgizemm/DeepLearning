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

path="C:/Users/busra/veri_manipulasyonu1/TRAIN_IMAGE"

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    path,
    labels = 'inferred',
    label_mode='int',
    seed=123,
    color_mode ='rgb',
    image_size=(224, 224),
    batch_size=64, 
    shuffle=True)

train_ds

for image, label in train_ds.take(1):
    print(label)
train_ds.class_names

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    path,
    labels = 'inferred',
    label_mode='int',
    seed=123,
    color_mode ='rgb',
    validation_split=0.2,
    subset="training",
    image_size=(224, 224),
    batch_size=64)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    path,
    labels = 'inferred',
    label_mode='int',
    seed=123,
    color_mode ='rgb',
    validation_split=0.2,
    subset="validation",
    image_size=(224, 224),
    batch_size=64)

train_ds

val_ds

class_names = train_ds.class_names
plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i].numpy().argmax()])
        plt.axis("off")

model = Sequential()
model.add(ResNet101V2 (include_top=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    classifier_activation="softmax"))
model.add(Flatten())
model.add(Dense(512, activation='softmax'))
model.add(Dense(5, activation = 'softmax'))

model.layers[0].trainable = False
model.summary()
model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001, rho = 0.9, epsilon=1e-08,decay=0.0),
            loss='sparse_categorical_crossentropy', metrics=['acc'])

history = model.fit( train_ds, epochs=5, batch_size=128,validation_data=val_ds)
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label = 'val_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.ylim([0.001, 2])
plt.legend(loc=0)

test_loss, test_acc = model.evaluate(val_ds, verbose=2)
