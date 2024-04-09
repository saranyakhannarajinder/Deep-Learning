#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install tensorflow matplotlib scikit-learn seaborn')


# In[2]:


import os
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.efficientnet import EfficientNetB0, preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix


# In[3]:


# training and validation image folders
train_dir = '/Users/mmx/Downloads/FaceExpressions/dataset/train'
validation_dir = '/Users/mmx/Downloads/FaceExpressions/dataset/validation'


# In[4]:


# data augmentation for training set and rescaling
train_datagen = ImageDataGenerator(
    rescale=1./255, rotation_range=40, width_shift_range=0.2, height_shift_range=0.2,
    shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest',
    preprocessing_function=preprocess_input
)
validation_datagen = ImageDataGenerator(
    rescale=1./255, preprocessing_function=preprocess_input
)
train_generator = train_datagen.flow_from_directory(
    train_dir, target_size=(224, 224), batch_size=32, class_mode='categorical'
)
validation_generator = validation_datagen.flow_from_directory(
    validation_dir, target_size=(224, 224), batch_size=32, class_mode='categorical'
)


# In[5]:


# use EfficientNetB0 for our model
base_model = EfficientNetB0(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
base_model.trainable = False  

# adding layers on top
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(6, activation='softmax')(x)  

# compiling the model
model = Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])


# In[6]:


# checkpoints and early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10)
checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True)

# start learning
start_time = time.time()
history = model.fit(
    train_generator, epochs=1, validation_data=validation_generator, verbose=1,
    callbacks=[early_stopping, checkpoint], steps_per_epoch=100, validation_steps=50
)
# calculate time
training_time = time.time() - start_time
print(f"Model trained in: {training_time:.2f} seconds")


# In[7]:


# plot training vs validation accuracy
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy Over Epochs')
plt.legend()
plt.show()

# plot training vs validation loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss Over the Epochs')
plt.legend()
plt.show()


# In[8]:


model.load_weights('best_model.h5')

# making predictions
pred = model.predict(validation_generator, steps=np.ceil(validation_generator.samples / validation_generator.batch_size), verbose=1)
pred_classes = np.argmax(pred, axis=1)

# getting true labels and class labels
true_classes = validation_generator.classes[:len(pred_classes)]
class_labels = list(validation_generator.class_indices.keys())

# confusion matrix
conf_matrix = confusion_matrix(true_classes, pred_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.title('Confusion Matrix')
plt.ylabel('Actual labels')
plt.xlabel('Predicted labels')
plt.show()


# In[9]:


# report
print(classification_report(true_classes, pred_classes, target_names=class_labels))


# saving the train model

# In[10]:


import pickle


# In[11]:


filename = 'trained_model.sav'
pickle.dump(model, open(filename,'wb'))


# loadding the save model

# In[12]:


loaded_model = pickle.load(open('trained_model.sav', 'rb'))


# Refrences: https://www.youtube.com/watch?v=WLwjvWq0GWA
#         https://www.youtube.com/watch?v=dkvgzL3gJVY

# In[ ]:




