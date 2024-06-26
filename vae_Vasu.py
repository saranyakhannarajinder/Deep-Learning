# -*- coding: utf-8 -*-
"""VAE.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1eRrpmkP1p6mc-xMw2TP0DX1WZILVLlbn
"""

import tensorflow as tf
from tensorflow.keras import layers, Model, backend as K
from google.colab import drive
import os
import numpy as np
import pandas as pd
from PIL import Image

from google.colab import drive
drive.mount('/content/drive')



dataset_path = '/content/drive/My Drive/FaceExpressions/dataset'

# List files in the dataset folder
print("Files in the dataset folder:")
for file in os.listdir(dataset_path):
    print(file)

from tensorflow.keras.preprocessing import image_dataset_from_directory

# Parameters
batch_size = 32
img_height = 64
img_width = 64

# Loading the dataset
train_dataset = image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

validation_dataset = image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)


AUTOTUNE = tf.data.experimental.AUTOTUNE

train_dataset = train_dataset.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.cache().prefetch(buffer_size=AUTOTUNE)

def preprocess_image(image_path):
    img = Image.open(image_path)
    img = img.resize((64, 64))  # Resize images to 64x64
    img = np.array(img) / 255.0  # Normalize pixel values
    return img

# Load and preprocess images from all expression directories in batches
batch_size = 32
image_data = []
for expression_dir in os.listdir(dataset_path):
    expression_dir_path = os.path.join(dataset_path, expression_dir)
    if os.path.isdir(expression_dir_path):
        images_in_dir = os.listdir(expression_dir_path)
        num_batches = len(images_in_dir) // batch_size
        for i in range(num_batches):
            batch_images = []
            for img_name in images_in_dir[i * batch_size: (i + 1) * batch_size]:
                img_path = os.path.join(expression_dir_path, img_name)
                img = preprocess_image(img_path)
                batch_images.append(img)
            image_data.extend(batch_images)
image_data = np.array(image_data)

latent_dim = 100

encoder_inputs = layers.Input(shape=(64, 64, 3))
x = layers.Conv2D(64, 3, strides=2, padding='same', activation='relu')(encoder_inputs)
x = layers.Conv2D(128, 3, strides=2, padding='same', activation='relu')(x)
x = layers.Flatten()(x)
z_mean = layers.Dense(latent_dim)(x)
z_log_var = layers.Dense(latent_dim)(x)

def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0., stddev=1.)
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

z = layers.Lambda(sampling)([z_mean, z_log_var])

decoder_inputs = layers.Input(shape=(latent_dim,))
x = layers.Dense(8 * 8 * 128, activation='relu')(decoder_inputs)
x = layers.Reshape((8, 8, 128))(x)
x = layers.Conv2DTranspose(64, 3, strides=2, padding='same', activation='relu')(x)
decoder_outputs = layers.Conv2DTranspose(3, 3, strides=2, padding='same', activation='sigmoid')(x)

encoder = Model(encoder_inputs, [z_mean, z_log_var, z], name='encoder')
decoder = Model(decoder_inputs, decoder_outputs, name='decoder')
vae_outputs = decoder(encoder(encoder_inputs)[2])
vae = Model(encoder_inputs, vae_outputs, name='vae')

# Resize vae_outputs to match the dimensions of encoder_inputs
vae_outputs_resized = tf.image.resize(vae_outputs, (64, 64))

# Define VAE loss function
reconstruction_loss = tf.keras.losses.mean_squared_error(encoder_inputs, vae_outputs_resized)
reconstruction_loss *= 64 * 64 * 3  # Scale up to match image dimensions
kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
kl_loss = tf.reduce_mean(kl_loss)
kl_loss *= -0.5
vae_loss = tf.reduce_mean(reconstruction_loss + kl_loss)

vae.add_loss(vae_loss)
vae.compile(optimizer='adam')

vae.fit(image_data, epochs=100, batch_size=32)

num_faces_to_generate = 12
random_latent_vectors = np.random.normal(size=(num_faces_to_generate, latent_dim))
decoded_faces = decoder.predict(random_latent_vectors)

# Plot the generated faces
import matplotlib.pyplot as plt

plt.figure(figsize=(20, 20))
for i in range(num_faces_to_generate):
    plt.subplot(1, num_faces_to_generate, i + 1)
    plt.imshow(decoded_faces[i])
    plt.axis('off')
plt.show()

