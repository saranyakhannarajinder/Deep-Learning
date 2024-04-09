#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[1]:


#importing all the required modules
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')


# In[2]:


#lists all the files in a directory
import os
dataset_dir = r'C:\Users\manit\Downloads\FaceExpressions\dataset'
 
print("files in the datset:")
for file in os.listdir(dataset_dir):
    print(file)


# In[3]:


#setting up the parameters
batch_size = 32
img_height = 224  
img_width = 224  


# In[16]:


#loading and setting up the taraing dataset 
train_dataset = image_dataset_from_directory(
    dataset_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)
#setting up the validation dataset
validation_dataset = image_dataset_from_directory(
    dataset_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

class_names = train_dataset.class_names


# In[5]:


#building Data agumentation Pipeline
data_augmentation = tf.keras.Sequential([
  tf.keras.layers.RandomFlip("horizontal_and_vertical"),
  tf.keras.layers.RandomRotation(0.2),
  tf.keras.layers.RandomZoom(0.2),
  tf.keras.layers.RandomContrast(0.2),
])
#preprocessing the dataset
def prepare_dataset(dataset, augment=False):
    if augment:
        dataset = dataset.map(lambda x, y: (data_augmentation(x, training=True), y),
                              num_parallel_calls=tf.data.AUTOTUNE)
    return dataset.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

train_dataset = prepare_dataset(train_dataset, augment=True)
validation_dataset = prepare_dataset(validation_dataset)


# In[6]:


#building the transfer learning model with VGG16
base_model = VGG16(include_top=False, weights='imagenet', input_shape=(img_height, img_width, 3))
base_model.trainable = False  

inputs = Input(shape=(img_height, img_width, 3))
x = base_model(inputs, training=False)  
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)
outputs = Dense(len(class_names), activation='softmax')(x)

model = Model(inputs=inputs, outputs=outputs)


# In[7]:


#training the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# In[8]:


#validating the model
history = model.fit(train_dataset,
                    epochs=3,
                    validation_data=validation_dataset)


# In[9]:


#unfreezing the top layers
base_model.trainable = True
for layer in base_model.layers[:-4]:
    layer.trainable = False
#tuning the model with lower learning rate
model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),  
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
history_fine = model.fit(train_dataset,
                         epochs=5,  
                         initial_epoch=history.epoch[-1],
                         validation_data=validation_dataset)


# In[10]:


# Assuming val_labels and predictions have been defined earlier as shown
val_labels = np.concatenate([y for x, y in validation_dataset], axis=0)
predictions = model.predict(validation_dataset)
predicted_labels = np.argmax(predictions, axis=1)

# Calculate the metrics
accuracy = accuracy_score(val_labels, predicted_labels)
precision = precision_score(val_labels, predicted_labels, average='weighted')
recall = recall_score(val_labels, predicted_labels, average='weighted')
f1 = f1_score(val_labels, predicted_labels, average='weighted')

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")


# In[14]:


import matplotlib.pyplot as plt
#plotting accuracy values 
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc='upper left')
# plotting loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()


# In[17]:


class_names = train_dataset.class_names


# In[18]:


#confusion matrics
cm = confusion_matrix(val_labels, predicted_labels)
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=train_dataset.class_names, yticklabels=train_dataset.class_names)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()


# In[19]:


#classification Report
from sklearn.metrics import classification_report
print(classification_report(val_labels, predicted_labels, target_names=train_dataset.class_names))


# In[20]:


#saving the trained model


# In[22]:


import pickle


# In[27]:


from tensorflow.keras.models import load_model
model = load_model('trained_model')


# In[ ]:




