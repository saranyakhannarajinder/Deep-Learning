#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
from keras.models import load_model
import numpy as np
from PIL import Image

# Function to preprocess the image
def preprocess_image(image):
    # Resize the image to the required input shape of the model
    image = image.resize((img_width, img_height))
    # Convert the image to a numpy array
    image = np.asarray(image) / 255.0  # Normalize pixel values to [0, 1]
    # Add an additional dimension to the image array for batch size
    image = np.expand_dims(image, axis=0)
    return image

# Load the saved model
model = load_model('CNN_model.h5')

# Set the input image dimensions according to your model
img_width, img_height = 150, 150

# Emotion labels
emotions = {0: 'Aheago', 1: 'Angry', 2: 'Neutral', 3: 'Happy', 4: 'Sad', 5: 'Surprise'}

# Title and instructions for the app
st.title('Facial Expression Classifier')
st.write('Upload a picture of a face.')

# File uploader widget
file = st.file_uploader("Choose an image...", type=["jpg", "png"])

# Once a file is uploaded
if file is not None:
    # Read the image file
    image = Image.open(file)
    # Display the image
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    
    # Preprocess the image
    processed_image = preprocess_image(image)
    
    # Make predictions
    prediction = model.predict(processed_image)
    
    # Determine the predicted emotion
    predicted_emotion = emotions[np.argmax(prediction)]
    
    # Display the predicted emotion
    st.write('Prediction:', predicted_emotion)


# In[ ]:




