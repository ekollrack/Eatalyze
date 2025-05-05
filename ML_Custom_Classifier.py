import numpy as np
import pandas as pd
from pathlib import Path
import os.path
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.metrics import classification_report
from tensorflow.keras import models, layers

#read in the dataset (unessesarily complex due to the way the dataset is structured - there are already predefined training and testing batches, but we dont use them)
image_dir = Path(r"C:\Users\Aaron Wilson\.cache\kagglehub\datasets\kmader\food41\versions\5\images")
filepaths = list(image_dir.glob(r'**/*.jpg'))
labels = list(map(lambda x: os.path.split(os.path.split(x)[0])[1], filepaths))
filepaths = pd.Series(filepaths, name='Filepath').astype(str)
labels = pd.Series(labels, name='Label')
images = pd.concat([filepaths, labels], axis=1)

#this reads the labels in correctly (annoying)
category_samples = []
for category in images['Label'].unique():
    category_slice = images.query("Label == @category")
    category_samples.append(category_slice.sample(100, random_state=1))
    image_df = pd.concat(category_samples, axis=0).sample(
    frac=1.0, random_state=1).reset_index(drop=True)

# display the label and counts (was for debugging)
print(image_df['Label'].value_counts())

#split the data into train and test
train_df, test_df = train_test_split(image_df, train_size=0.7, shuffle=True)

#use keras to configure image generators (preprocessing - norm to 1)
#  rescale=1./255
train_generator = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    validation_split=0.2)
test_generator = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255)

#setup training, validation, and test generators
train_images = train_generator.flow_from_dataframe(dataframe=train_df,
                                                   x_col='Filepath',
                                                   y_col='Label',
                                                   target_size=(224, 224),
                                                   color_mode='rgb',
                                                   class_mode='categorical',
                                                   batch_size=32,
                                                   shuffle=True,
                                                   subset='training')

val_images = train_generator.flow_from_dataframe(dataframe=train_df,
                                                 x_col='Filepath',
                                                 y_col='Label',
                                                 target_size=(224, 224),
                                                 color_mode='rgb',
                                                 class_mode='categorical',
                                                 batch_size=32,
                                                 shuffle=True,
                                                 subset='validation')

test_images = test_generator.flow_from_dataframe(dataframe=test_df,
                                                 x_col='Filepath',
                                                 y_col='Label',
                                                 target_size=(224, 224),
                                                 color_mode='rgb',
                                                 class_mode='categorical',
                                                 batch_size=32,
                                                 shuffle=False)

#build custom model using keras
model = models.Sequential([
    layers.Input(shape=(224, 224, 3)),
    
    #layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    #layers.BatchNormalization(),
    #layers.MaxPooling2D(pool_size=(2, 2)),

    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=(2, 2)),

    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=(2, 2)),

    layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=(2, 2)),

    #add or remove this layer to increase or decrease # of params
    #layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
    #layers.BatchNormalization(),
    #layers.MaxPooling2D(pool_size=(2, 2)),

    #can be used for more paramters (commented out due to inneficiency)
    #layers.Flatten(),
    #layers.Dense(512, activation='relu'),

    #need this next line to cut down parameters to reduce training time (to remove huge flatten+dense operation)
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.4),
    layers.Dense(101, activation='softmax')  # 101 food classes (this softmax is interesting as it will smooth the results - 
                                                #meaning that we oftentimes get predicted entries that are "proximaly close" to the correct answer
                                                # donuts are not similar to dumplings, but they are close on the "labels" list
])

#used to compare vs other network architectures
print(model.summary())

#compile the model and train
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy'])

#impliment early stop for overfitting (also will be used for saving the best model intermittently when we have a big epoch # - model checkpoint)
history = model.fit(
    train_images,
    validation_data=val_images,
    epochs=10,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True)])


#print the results
results = model.evaluate(test_images, verbose=0)
print("Test Accuracy: {:.2f}%".format(results[1] * 100))

#save the trained model to a file
model.save("food_cnn_model_custom.h5")