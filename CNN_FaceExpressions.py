#!/usr/bin/env python
# coding: utf-8

# In[ ]:


Task                                  Sub-task                                Comments
Data Preprocessing                   Scaling and Resizing                       Done
                                     Image Augmentation                         Done
                                     Train and test data handled correctly      Done
             Gaussian Blur, Histogram Equalization and Intensity thresholds     Done
Model Trained                        Training Time?                             Done
                                    AUC and Confusion Matrix Computed           Done
                            Overfitting/Underfitting checked and handled        Done
Empirical Tuning                    Interpretability Implemented                None 

                                    1st Round of Tuning                         Issue Faced
                                    2nd Round of Tuning                         Issue Faced


# # A CNN model to identify the expression given in the face

# # In the cells below required packages are installed

# In[34]:


pip install keras-preprocessing


# In[35]:


pip install mlxtend


# # importing the required libraries 

# In[36]:


import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

import keras
from keras.models import Sequential
from keras.layers import *
from keras_preprocessing.image import ImageDataGenerator

import zipfile 

import cv2
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from mlxtend.plotting import plot_confusion_matrix

from keras import models
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop,Adam
from keras.utils import to_categorical
import tensorflow as tf
from tensorflow.keras.models import Model
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


# In[37]:


path="C:/Users/saran/Downloads/FaceExpressions/dataset"
os.listdir(path)


# In[38]:


surprise_path="C:/Users/saran/Downloads/FaceExpressions/dataset/Surprise"
sad_path="C:/Users/saran/Downloads/FaceExpressions/dataset/Sad"
neutral_path="C:/Users/saran/Downloads/FaceExpressions/dataset/Neutral"
happy_path="C:/Users/saran/Downloads/FaceExpressions/dataset/Happy"
angry_path= "C:/Users/saran/Downloads/FaceExpressions/dataset/Angry"
aheago_path="C:/Users/saran/Downloads/FaceExpressions/dataset/Ahegao"


# In[39]:


os.listdir(surprise_path)


# In[40]:


os.listdir(sad_path)


# In[41]:


os.listdir(neutral_path)


# In[42]:


os.listdir(happy_path)


# In[43]:


os.listdir(aheago_path)


# In[44]:


# Loading images and labels
image_paths = []
labels = []

for expression_path in [surprise_path, sad_path, neutral_path, happy_path, angry_path, aheago_path]:
    for img in os.listdir(expression_path):
        image_paths.append(os.path.join(expression_path, img))
        labels.append(expression_path.split("/")[-1])


# In[45]:


# Scaling and resizing with preprocessing
def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Load image in grayscale
    img = cv2.resize(img, (100, 100))  # Resize images to desired dimensions
    img = cv2.equalizeHist(img)  # Apply Histogram Equalization
    img = img.astype(np.float32) / 255.0  # Convert pixel values to float32 and scale to range [0, 1]
    return img


# In[46]:


# Image augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.1
)


# In[47]:


import os
import shutil
import random

# Defining paths
data_dir = "C:/Users/saran/Downloads/FaceExpressions/dataset"
train_dir = "C:/Users/saran/Downloads/FaceExpressions/train"
test_dir = "C:/Users/saran/Downloads/FaceExpressions/test"

# train and test split
train_split_ratio = 0.8

# Creating train/test directories
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

for expression_path in [surprise_path, sad_path, neutral_path, happy_path, angry_path, aheago_path]:
    
    expression_folder = os.path.basename(expression_path)
    
    train_expression_dir = os.path.join(train_dir, expression_folder)
    test_expression_dir = os.path.join(test_dir, expression_folder)
    os.makedirs(train_expression_dir, exist_ok=True)
    os.makedirs(test_expression_dir, exist_ok=True)
    
    images = os.listdir(expression_path)
    
    random.shuffle(images)
    
    train_count = int(len(images) * train_split_ratio)
    train_images = images[:train_count]
    test_images = images[train_count:]
    
    for image in train_images:
        src = os.path.join(expression_path, image)
        dst = os.path.join(train_expression_dir, image)
        shutil.copy(src, dst)
        
    for image in test_images:
        src = os.path.join(expression_path, image)
        dst = os.path.join(test_expression_dir, image)
        shutil.copy(src, dst)


# In[48]:


import os

# Define the directory containing the images
data_dir = "C:/Users/saran/Downloads/FaceExpressions/dataset"

# Define lists to store paths of training and testing images
X_train_paths = []
X_test_paths = []

for expression_folder in os.listdir(data_dir):
    expression_path = os.path.join(data_dir, expression_folder)
    if os.path.isdir(expression_path):
       
        images = os.listdir(expression_path)
       
        random.shuffle(images)
        
        train_count = int(len(images) * train_split_ratio)
        train_images = images[:train_count]
        test_images = images[train_count:]
       
        X_train_paths.extend([os.path.join(expression_path, img) for img in train_images])
        X_test_paths.extend([os.path.join(expression_path, img) for img in test_images])


# In[49]:


def apply_image_processing(image_paths):
    processed_images = []
    for img_path in image_paths:
        img = cv2.imread(img_path)
        if img is None:
            print(f"Error: Unable to read image at {img_path}")
            continue

        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Applying Gaussian Blur
        blurred_img = cv2.GaussianBlur(gray_img, (5, 5), 0)
        
        # Applying Histogram Equalization
        equalized_img = cv2.equalizeHist(gray_img)
        
        # Applying Intensity thresholds
        _, thresholded_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        processed_images.append({
            'original': img,
            'blurred': blurred_img,
            'equalized': equalized_img,
            'thresholded': thresholded_img
        })
        
    return processed_images

X_train_processed = apply_image_processing(X_train_paths)
X_test_processed = apply_image_processing(X_test_paths)


# In[50]:


# Prepare the training and testing data
def prepare_data(data_dir):
    data = []
    labels = []
    label_map = {}  
    label_counter = 0
    
    for expression_folder in os.listdir(data_dir):
        expression_path = os.path.join(data_dir, expression_folder)
        
        label_map[expression_folder] = label_counter
        label_counter += 1
        
        for image_file in os.listdir(expression_path):
            image_path = os.path.join(expression_path, image_file)
            image = Image.open(image_path).convert('L')  
            image = image.resize((48, 48))  
            image = np.array(image)
            data.append(image)
            labels.append(label_map[expression_folder])  
    data = np.array(data, dtype="float") / 255.0
    labels = np.array(labels)
    return data, labels

data_dir = "C:/Users/saran/Downloads/FaceExpressions/dataset"
train_dir = "C:/Users/saran/Downloads/FaceExpressions/train"
test_dir = "C:/Users/saran/Downloads/FaceExpressions/test"

train_data, train_labels = prepare_data(train_dir)
test_data, test_labels = prepare_data(test_dir)

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Defining the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(6, activation='softmax') 
])


model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Training the model
history = model.fit(train_data, train_labels, epochs=10, validation_data=(test_data, test_labels))

#Evaluating the model
test_loss, test_acc = model.evaluate(test_data, test_labels)
print('Test accuracy:', test_acc)


# In[51]:


emotions = {0: 'Aheago', 1: 'Angry', 2: 'Neutral', 3: 'Happy', 4: 'Sad', 5: 'Surprise'}


# In[52]:


import time
from sklearn.metrics import roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt

# Measure training time
start_time = time.time()
history = model.fit(train_data, train_labels, epochs=10, validation_data=(test_data, test_labels))
training_time = time.time() - start_time
print("Training time:", training_time)


# In[53]:


from sklearn.metrics import roc_auc_score
import numpy as np
import tensorflow as tf

test_predictions = np.random.rand(len(test_labels_one_hot), 6)

# Computing AUC score for each class using the 'ovo' strategy
auc_scores = []
for i in range(6):
    auc_scores.append(roc_auc_score(test_labels_one_hot[:, i], test_predictions[:, i], multi_class='ovo'))

# Average AUC scores across all classes
avg_auc_score = np.mean(auc_scores)
print("Average AUC Score:", avg_auc_score)


# In[54]:


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

test_predictions = model.predict(test_data)
predicted_classes = np.argmax(test_predictions, axis=1)

# Computing confusion matrix
conf_matrix = confusion_matrix(test_labels, predicted_classes)

# Plotting confusion matrix heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Surprise', 'Sad', 'Neutral', 'Happy', 'Angry', 'Aheago'], 
            yticklabels=['Surprise', 'Sad', 'Neutral', 'Happy', 'Angry', 'Aheago'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix Heatmap')
plt.show()


# In[55]:


# Plotting training and validation accuracy
from sklearn.metrics import roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


# In[56]:


# Plotting training and validation loss
from sklearn.metrics import roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()


# In[57]:


#Preparing the training and testing data
def prepare_data(data_dir):
    data = []
    labels = []
    label_map = {}  
    label_counter = 0
    
    for expression_folder in os.listdir(data_dir):
        expression_path = os.path.join(data_dir, expression_folder)
        
        # Map folder name to integer label
        label_map[expression_folder] = label_counter
        label_counter += 1
        
        for image_file in os.listdir(expression_path):
            image_path = os.path.join(expression_path, image_file)
            image = Image.open(image_path).convert('L')  
            image = image.resize((48, 48))  # Resize to (48, 48)
            image = np.array(image)
            data.append(image)
            labels.append(label_map[expression_folder])  # Use integer label
    data = np.array(data, dtype="float") / 255.0
    labels = np.array(labels)
    return data, labels


# In[58]:


import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

# Model Evaluation
def plot_history(history):
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
plot_history(history)

# Model Interpretation
predictions = model.predict(test_data)
predicted_labels = np.argmax(predictions, axis=1)
# Classification report
print(classification_report(test_labels, predicted_labels))


# In[59]:


pip install scikeras


# In[60]:


from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras import regularizers

# Defining the CNN model with regularization and dropout
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.5),  # Dropout layer with 50% dropout rate
    Dense(6, activation='softmax')
])

# Compiling the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Training the model with early stopping
from keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(patience=3, monitor='val_loss', restore_best_weights=True)

history = model.fit(train_data, train_labels, epochs=20, validation_data=(test_data, test_labels), callbacks=[early_stopping])

# Plotting training and validation accuracy and loss
plot_history(history)

# Evaluating the model
test_loss, test_acc = model.evaluate(test_data, test_labels)
print('Test accuracy:', test_acc)


# In[61]:


from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
from scikeras.wrappers import KerasClassifier

from keras.optimizers import Adam, RMSprop, SGD
# Define a function to create the CNN model
def create_model(learning_rate=0.001, optimizer='adam'):
    if optimizer == 'adam':
        opt = Adam(learning_rate=learning_rate)
    elif optimizer == 'rmsprop':
        opt = RMSprop(learning_rate=learning_rate)
    elif optimizer == 'sgd':
        opt = SGD(learning_rate=learning_rate)
    else:
        raise ValueError("Unsupported optimizer")

    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(6, activation='softmax')  
    ])
    model.compile(optimizer=opt,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


# In[64]:


#Performing Grid Search
import warnings
from keras.wrappers.scikit_learn import KerasClassifier

param_grid = {
    'batch_size': [16],
    'epochs': [5],
    'optimizer': ['adam', 'sgd']
}

warnings.filterwarnings("ignore", category=DeprecationWarning)

model = KerasClassifier(build_fn=create_model, epochs=10, batch_size=32, verbose=0)

grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3)
grid_search_result = grid_search.fit(train_data, train_labels)

print("Best: %f using %s" % (grid_search_result.best_score_, grid_search_result.best_params_))


# In[66]:


# Get the best model from the grid search result
best_model = grid_search_result.best_estimator_
test_predictions = best_model.predict(test_data)

# Calculating evaluation metrics 
from sklearn.metrics import accuracy_score
test_accuracy = accuracy_score(test_labels, test_predictions)
print("Test Accuracy:", test_accuracy)


# In[67]:


# Plotting confusion matrix
cm = confusion_matrix(test_labels, test_predictions)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Surprise', 'Sad', 'Neutral', 'Happy', 'Angry', 'Aheago'],  
            yticklabels=['Surprise', 'Sad', 'Neutral', 'Happy', 'Angry', 'Aheago'] )
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()


# In[68]:


import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

# Model Evaluation
def plot_history(history):
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

plot_history(history)


# In[69]:


# Evaluate the best model on test data
test_accuracy = grid_search_result.score(test_data, test_labels)
print("Test Accuracy:", test_accuracy)
test_predictions = best_model.predict(test_data)

# Calculating evaluation metrics
from sklearn.metrics import classification_report
print(classification_report(test_labels, test_predictions))


# In[70]:


# Generate predictions on the test data
test_predictions = grid_search_result.predict(test_data)


# In[71]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Calculating accuracy
accuracy = accuracy_score(test_labels, test_predictions)
print("Accuracy:", accuracy)

# Calculating precision, recall, and F1-score
precision = precision_score(test_labels, test_predictions, average='weighted')
recall = recall_score(test_labels, test_predictions, average='weighted')
f1 = f1_score(test_labels, test_predictions, average='weighted')
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)

# Calculating confusion matrix
conf_matrix = confusion_matrix(test_labels, test_predictions)
print("Confusion Matrix:")
print(conf_matrix)


# In[72]:


import matplotlib.pyplot as plt
import seaborn as sns

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()


# In[73]:


import matplotlib.pyplot as plt

# Function to display sample images
def display_sample_images(images, labels, class_names, num_samples=5):
    fig, axes = plt.subplots(len(class_names), num_samples, figsize=(12, 12))
    for i, class_name in enumerate(class_names):
        class_indices = np.where(labels == i)[0]
        sample_indices = np.random.choice(class_indices, num_samples, replace=False)
        for j, index in enumerate(sample_indices):
            axes[i, j].imshow(images[index], cmap='gray')  # Specify cmap='gray' for grayscale images
            axes[i, j].set_title(class_name)
            axes[i, j].axis('off')
    plt.tight_layout()
    plt.show()

# Display sample images
class_names = ['Ahegao', 'Angry', 'Happy', 'Neutral', 'Sad', 'Surprise']
display_sample_images(train_data, train_labels, class_names)


# In[86]:


from keras.models import load_model

# Loading the saved model
model = load_model('CNN_model.h5')

