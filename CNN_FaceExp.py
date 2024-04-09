#!/usr/bin/env python
# coding: utf-8

# # A CNN model to identify the expression given in the face

# # In the cells below required packages are installed

# In[1]:


pip install keras-preprocessing


# In[2]:


pip install mlxtend


# # importing the required libraries 

# In[3]:


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


# In[4]:


path="C:/Users/saran/Downloads/FaceExpressions/dataset"
os.listdir(path)


# In[5]:


surprise_path="C:/Users/saran/Downloads/FaceExpressions/dataset/Surprise"
sad_path="C:/Users/saran/Downloads/FaceExpressions/dataset/Sad"
neutral_path="C:/Users/saran/Downloads/FaceExpressions/dataset/Neutral"
happy_path="C:/Users/saran/Downloads/FaceExpressions/dataset/Happy"
angry_path= "C:/Users/saran/Downloads/FaceExpressions/dataset/Angry"
aheago_path="C:/Users/saran/Downloads/FaceExpressions/dataset/Ahegao"


# In[6]:


os.listdir(surprise_path)


# In[7]:


os.listdir(sad_path)


# In[8]:


os.listdir(neutral_path)


# In[9]:


os.listdir(happy_path)


# In[10]:


os.listdir(aheago_path)


# In[12]:


# Load images and labels
image_paths = []
labels = []

for expression_path in [surprise_path, sad_path, neutral_path, happy_path, angry_path, aheago_path]:
    for img in os.listdir(expression_path):
        image_paths.append(os.path.join(expression_path, img))
        labels.append(expression_path.split("/")[-1])


# In[24]:


# Scaling and resizing with preprocessing
def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Load image in grayscale
    img = cv2.resize(img, (100, 100))  # Resize images to desired dimensions
    img = cv2.equalizeHist(img)  # Apply Histogram Equalization
    img = img.astype(np.float32) / 255.0  # Convert pixel values to float32 and scale to range [0, 1]
    return img


# In[25]:


# Image augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.1
)


# In[31]:


import os
import shutil
import random

# Define paths
data_dir = "C:/Users/saran/Downloads/FaceExpressions/dataset"
train_dir = "C:/Users/saran/Downloads/FaceExpressions/train"
test_dir = "C:/Users/saran/Downloads/FaceExpressions/test"

# Define train/test split ratio
train_split_ratio = 0.8

# Create train/test directories
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Iterate over expression folders
for expression_path in [surprise_path, sad_path, neutral_path, happy_path, angry_path, aheago_path]:
    
    # Extract expression folder name
    expression_folder = os.path.basename(expression_path)
    
    # Create train/test subdirectories for the expression
    train_expression_dir = os.path.join(train_dir, expression_folder)
    test_expression_dir = os.path.join(test_dir, expression_folder)
    os.makedirs(train_expression_dir, exist_ok=True)
    os.makedirs(test_expression_dir, exist_ok=True)
    
    # List images in expression folder
    images = os.listdir(expression_path)
    
    # Shuffle images
    random.shuffle(images)
    
    # Split images into train and test sets
    train_count = int(len(images) * train_split_ratio)
    train_images = images[:train_count]
    test_images = images[train_count:]
    
    # Move images to respective directories
    for image in train_images:
        src = os.path.join(expression_path, image)
        dst = os.path.join(train_expression_dir, image)
        shutil.copy(src, dst)
        
    for image in test_images:
        src = os.path.join(expression_path, image)
        dst = os.path.join(test_expression_dir, image)
        shutil.copy(src, dst)


# In[34]:


import os

# Define the directory containing the images
data_dir = "C:/Users/saran/Downloads/FaceExpressions/dataset"

# Define lists to store paths of training and testing images
X_train_paths = []
X_test_paths = []

# Iterate over the subdirectories in the data directory
for expression_folder in os.listdir(data_dir):
    expression_path = os.path.join(data_dir, expression_folder)
    if os.path.isdir(expression_path):
        # List images in the current expression folder
        images = os.listdir(expression_path)
        # Shuffle the list of images
        random.shuffle(images)
        # Split the images into training and testing sets
        train_count = int(len(images) * train_split_ratio)
        train_images = images[:train_count]
        test_images = images[train_count:]
        # Add the paths of training and testing images to the respective lists
        X_train_paths.extend([os.path.join(expression_path, img) for img in train_images])
        X_test_paths.extend([os.path.join(expression_path, img) for img in test_images])

# Now, X_train_paths and X_test_paths contain the paths of training and testing images, respectively


# In[35]:


def apply_image_processing(image_paths):
    processed_images = []
    for img_path in image_paths:
        # Load image
        img = cv2.imread(img_path)
        if img is None:
            print(f"Error: Unable to read image at {img_path}")
            continue

        # Convert image to grayscale
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian Blur
        blurred_img = cv2.GaussianBlur(gray_img, (5, 5), 0)
        
        # Apply Histogram Equalization
        equalized_img = cv2.equalizeHist(gray_img)
        
        # Apply Intensity thresholds
        _, thresholded_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        processed_images.append({
            'original': img,
            'blurred': blurred_img,
            'equalized': equalized_img,
            'thresholded': thresholded_img
        })
        
    return processed_images

# Assuming X_train_paths and X_test_paths are lists of image paths
X_train_processed = apply_image_processing(X_train_paths)
X_test_processed = apply_image_processing(X_test_paths)


# In[36]:


# Prepare the training and testing data
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
            image = image.resize((48, 48))  
            image = np.array(image)
            data.append(image)
            labels.append(label_map[expression_folder])  
    data = np.array(data, dtype="float") / 255.0
    labels = np.array(labels)
    return data, labels

# Define paths
data_dir = "C:/Users/saran/Downloads/FaceExpressions/dataset"
train_dir = "C:/Users/saran/Downloads/FaceExpressions/train"
test_dir = "C:/Users/saran/Downloads/FaceExpressions/test"

# Prepare training and testing data
train_data, train_labels = prepare_data(train_dir)
test_data, test_labels = prepare_data(test_dir)

# Step 3: Define and train the model
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Define the CNN model
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

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(train_data, train_labels, epochs=10, validation_data=(test_data, test_labels))

# Step 4: Evaluate the model
test_loss, test_acc = model.evaluate(test_data, test_labels)
print('Test accuracy:', test_acc)


# In[37]:


emotions = {0: 'Aheago', 1: 'Angry', 2: 'Neutral', 3: 'Happy', 4: 'Sad', 5: 'Surprise'}


# In[39]:


import time
from sklearn.metrics import roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt

# Measure training time
start_time = time.time()
history = model.fit(train_data, train_labels, epochs=10, validation_data=(test_data, test_labels))
training_time = time.time() - start_time
print("Training time:", training_time)


# In[40]:



from sklearn.metrics import roc_auc_score

# Convert labels to one-hot encoding
test_labels_one_hot = tf.keras.utils.to_categorical(test_labels, num_classes=6)

# Compute AUC score for each class using the 'ovo' strategy
auc_scores = []
for i in range(6):
    auc_scores.append(roc_auc_score(test_labels_one_hot[:, i], test_predictions[:, i], multi_class='ovo'))

# Average AUC scores across all classes
avg_auc_score = np.mean(auc_scores)
print("Average AUC Score:", avg_auc_score)


# In[42]:


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Get predictions for test data
test_predictions = model.predict(test_data)
predicted_classes = np.argmax(test_predictions, axis=1)

# Compute confusion matrix
conf_matrix = confusion_matrix(test_labels, predicted_classes)

# Plot confusion matrix heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Surprise', 'Sad', 'Neutral', 'Happy', 'Angry', 'Aheago'], 
            yticklabels=['Surprise', 'Sad', 'Neutral', 'Happy', 'Angry', 'Aheago'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix Heatmap')
plt.show()


# In[43]:


# Plot training and validation accuracy
from sklearn.metrics import roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


# In[44]:


# Plot training and validation loss
from sklearn.metrics import roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()


# In[45]:


# Step 2: Prepare the training and testing data
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


# In[46]:


import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

# Model Evaluation
# Plot training and validation accuracy and loss
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


# In[47]:


pip install scikeras


# In[48]:


from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras import regularizers

# Define the CNN model with regularization and dropout
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

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model with early stopping
from keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(patience=3, monitor='val_loss', restore_best_weights=True)

history = model.fit(train_data, train_labels, epochs=20, validation_data=(test_data, test_labels), callbacks=[early_stopping])

# Plot training and validation accuracy and loss
plot_history(history)

# Evaluate the model
test_loss, test_acc = model.evaluate(test_data, test_labels)
print('Test accuracy:', test_acc)


# In[51]:


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


# In[53]:


import warnings
from keras.wrappers.scikit_learn import KerasClassifier

# Suppress DeprecationWarning for KerasClassifier
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Create KerasClassifier instance
model = KerasClassifier(build_fn=create_
                        model, epochs=10, batch_size=32, verbose=0)

# Perform grid search
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3)
grid_search_result = grid_search.fit(train_data, train_labels)

# Print best results
print("Best: %f using %s" % (grid_search_result.best_score_, grid_search_result.best_params_))


# In[ ]:


# Perform hyperparameter tuning using GridSearchCV
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
from scikeras.wrappers import KerasClassifier

from keras.optimizers import Adam, RMSprop, SGD
param_grid = {
    'optimizer': ['adam', 'rmsprop', 'sgd']
}


model = KerasClassifier(model=create_model, epochs=10, batch_size=32, verbose=0, learning_rate=0.001)
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3)
grid_search_result = grid_search.fit(train_data, train_labels)

print("Best: %f using %s" % (grid_search_result.best_score_, grid_search_result.best_params_))


# In[60]:


# Get the best model from the grid search result
best_model = grid_search_result.best_estimator_

# Generate predictions on the test data
test_predictions = best_model.predict(test_data)

# Calculate evaluation metrics manually
from sklearn.metrics import accuracy_score
test_accuracy = accuracy_score(test_labels, test_predictions)
print("Test Accuracy:", test_accuracy)


# In[61]:


# Plot confusion matrix
cm = confusion_matrix(test_labels, test_predictions)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Surprise', 'Sad', 'Neutral', 'Happy', 'Angry', 'Aheago'],  
            yticklabels=['Surprise', 'Sad', 'Neutral', 'Happy', 'Angry', 'Aheago'] )
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()


# In[66]:


import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

# Model Evaluation
# Plot training and validation accuracy and loss
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


# In[68]:


# Evaluate the best model on test data
test_accuracy = grid_search_result.score(test_data, test_labels)
print("Test Accuracy:", test_accuracy)

# Generate predictions on the test data
test_predictions = best_model.predict(test_data)

# Calculate evaluation metrics manually
# For example, you can use classification_report from sklearn.metrics
from sklearn.metrics import classification_report
print(classification_report(test_labels, test_predictions))

# Further analysis or fine-tuning based on the evaluation results
# This could include adjusting hyperparameters, exploring different architectures, or data preprocessing techniques.


# In[69]:


# Generate predictions on the test data
test_predictions = grid_search_result.predict(test_data)


# In[70]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Calculate accuracy
accuracy = accuracy_score(test_labels, test_predictions)
print("Accuracy:", accuracy)

# Calculate precision, recall, and F1-score
precision = precision_score(test_labels, test_predictions, average='weighted')
recall = recall_score(test_labels, test_predictions, average='weighted')
f1 = f1_score(test_labels, test_predictions, average='weighted')
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)

# Calculate confusion matrix
conf_matrix = confusion_matrix(test_labels, test_predictions)
print("Confusion Matrix:")
print(conf_matrix)


# In[71]:


import matplotlib.pyplot as plt
import seaborn as sns

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()


# In[74]:


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


# In[79]:


model.save('CNN_model.h5')


# In[92]:


import cv2
import numpy as np
from tensorflow.keras.models import Model

class GradCAM:
    def __init__(self, model, classIdx, layerName=None):
        self.model = model
        self.classIdx = classIdx
        self.layerName = layerName
        if self.layerName is None:
            self.layerName = self.find_target_layer()
            
    def find_target_layer(self):
        for layer in reversed(self.model.layers):
            if len(layer.output_shape) == 4:
                return layer.name
        raise ValueError("Could not find 4D layer. Cannot apply GradCAM.")
        
    def compute_heatmap(self, image, eps=1e-8):
        gradModel = Model(inputs=[self.model.inputs],
                          outputs=[self.model.get_layer(self.layerName).output, self.model.output])

        with tf.GradientTape() as tape:
            inputs = tf.cast(image, tf.float32)
            (convOutputs, predictions) = gradModel(inputs)
            loss = predictions[:, self.classIdx]

        grads = tape.gradient(loss, convOutputs)
        castConvOutputs = tf.cast(convOutputs > 0, "float32")
        castGrads = tf.cast(grads > 0, "float32")
        guidedGrads = castConvOutputs * castGrads * grads
        convOutputs = convOutputs[0]
        guidedGrads = guidedGrads[0]

        weights = tf.reduce_mean(guidedGrads, axis=(0, 1))
        cam = tf.reduce_sum(tf.multiply(weights, convOutputs), axis=-1)

        (w, h) = (image.shape[2], image.shape[1])
        heatmap = cv2.resize(cam.numpy(), (w, h))
        numer = heatmap - np.min(heatmap)
        denom = (heatmap.max() - heatmap.min()) + eps
        heatmap = numer / denom
        heatmap = (heatmap * 255).astype("uint8")
        
        return heatmap

    def overlay_heatmap(self, heatmap, image, alpha=0.5, colormap=cv2.COLORMAP_VIRIDIS):
        heatmap = cv2.applyColorMap(heatmap, colormap)
        output = cv2.addWeighted(image, alpha, heatmap, 1 - alpha, 0)
        return (heatmap, output)


# In[2]:


import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
# Define the GradCAM class
class GradCAM:
    # Implement the GradCAM class here


# Load the saved model


# Now you can use loaded_model for predictions, evaluation, etc.

# Define test_images, cnn model, test_labels, and emotions
    test_images = ["C:/Users/saran/Downloads/FaceExpressions/test"]  # Define your list of test images
loaded_model = load_model('CNN_model.h5') # Define your trained CNN model
test_labels = ["C:/Users/saran/Downloads/FaceExpressions/test"]  # Define your list of test labels
emotions = ['Ahegao', 'Angry', 'Happy', 'Neutral', 'Sad', 'Surprise']  # Define your list of emotion labels

plt.figure(figsize=[16, 16])
for i in range(36):
    img = cv2.imread(test_images[i], cv2.IMREAD_GRAYSCALE)  # Read the image using OpenCV
    p_dist = cnn.predict(np.expand_dims(img, axis=0))
    k = np.argmax(p_dist)
    p = np.max(p_dist)

    # Instantiate GradCAM with the CNN model and predicted class index
    grad_cam = GradCAM(cnn, k)

    # Compute heatmap
    heatmap = grad_cam.compute_heatmap(np.expand_dims(img, axis=0))

    # Overlay heatmap on the original image
    overlaid_img = grad_cam.overlay_heatmap(heatmap, img, alpha=0.5)

    plt.subplot(6, 6, i + 1)
    plt.imshow(overlaid_img, cmap='gray')  # Display the overlaid image in grayscale
    plt.title(f'{emotions[test_labels[i]]} - ({emotions[k]} - {p:.4f})')
    plt.axis('off')

plt.tight_layout()
plt.show()


# In[ ]:




