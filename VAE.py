#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install tensorflow


# In[2]:


import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


# In[3]:


from google.colab import drive
drive.mount('/content/drive')


# In[4]:


dataset_path = '/content/drive/My Drive/dataset'


# In[6]:


import tensorflow as tf

def decode_image(file_path, target_size=(64, 64)):
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, target_size)
    img = img / 255.0
    return img

def create_dataset(image_paths, batch_size=32):
    dataset = tf.data.Dataset.from_tensor_slices(image_paths)
    dataset = dataset.map(decode_image, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset

# all image paths
image_paths = glob.glob(dataset_path + '/**/*.jpg', recursive=True)

#training and test sets
train_paths, test_paths = train_test_split(image_paths, test_size=0.2, random_state=42)


train_dataset = create_dataset(train_paths, batch_size=32)
test_dataset = create_dataset(test_paths, batch_size=32)


# In[11]:


from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, Conv2DTranspose, Lambda, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras import backend as K
import numpy as np
import tensorflow as tf

# Network parameters
image_shape = (64, 64, 3)
latent_dim = 2

# Encoder architecture
inputs = Input(shape=image_shape, name='encoder_input')
x = Conv2D(32, 3, activation='relu', strides=2, padding='same')(inputs)
x = Conv2D(64, 3, activation='relu', strides=2, padding='same')(x)
x = Flatten()(x)
x = Dense(16, activation='relu')(x)
z_mean = Dense(latent_dim, name='z_mean')(x)
z_log_var = Dense(latent_dim, name='z_log_var')(x)
z = Lambda(lambda p: p[0] + K.exp(p[1] / 2) * K.random_normal(K.shape(p[0])), output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

# Instantiate encoder
encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')

# Decoder architecture
latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
x = Dense(16 * 16 * 64, activation='relu')(latent_inputs)
x = Reshape((16, 16, 64))(x)
x = Conv2DTranspose(64, 3, activation='relu', strides=2, padding='same')(x)
x = Conv2DTranspose(32, 3, activation='relu', strides=2, padding='same')(x)

#final layer
outputs = Conv2DTranspose(3, 3, activation='sigmoid', padding='same', name='decoder_output')(x)

# Instantiate decoder
decoder = Model(latent_inputs, outputs, name='decoder')

# Instantiate VAE
outputs = decoder(encoder(inputs)[2])
vae = Model(inputs, outputs, name='vae_mlp')

# VAE loss
reconstruction_loss = binary_crossentropy(K.flatten(inputs), K.flatten(outputs))
reconstruction_loss *= image_shape[0] * image_shape[1]
kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
kl_loss = K.sum(kl_loss, axis=-1)
kl_loss *= -0.5
vae_loss = K.mean(reconstruction_loss + kl_loss)
vae.add_loss(vae_loss)
vae.compile(optimizer='adam')

# Summary
vae.summary()


# In[12]:


print(K.int_shape(inputs))
print(K.int_shape(outputs))


# In[13]:


# Training VAE
vae.fit(train_dataset, epochs=30, validation_data=test_dataset)


# In[15]:


import matplotlib.pyplot as plt
def plot_latent_space(decoder, n=10, figsize=15):
    # Display a grid
    digit_size = 64
    figure = np.zeros((digit_size * n, digit_size * n, 3))
    grid_x = np.linspace(-3, 3, n)
    grid_y = np.linspace(-3, 3, n)[::-1]
    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.array([[xi, yi]])
            x_decoded = decoder.predict(z_sample)
            face = x_decoded[0].reshape(digit_size, digit_size, 3)
            figure[i * digit_size: (i + 1) * digit_size,
                   j * digit_size: (j + 1) * digit_size] = face
    plt.figure(figsize=(figsize, figsize))
    plt.imshow(figure, aspect='auto')
    plt.axis('off')
    plt.show()

plot_latent_space(decoder)


# In[17]:


def interpolate_between_two_points(decoder, start_point, end_point, n=10):
    start_point = start_point.flatten()
    end_point = end_point.flatten()


    z_samples = np.linspace(start_point, end_point, n)

    # Prediction using the decoder
    images = decoder.predict(z_samples)

    plt.figure(figsize=(20, 4))
    for i, image in enumerate(images):
        ax = plt.subplot(1, n, i + 1)
        plt.imshow(image.reshape(64, 64, 3))
        plt.axis('off')
    plt.show()
start_point = np.array([[-3, -3]])
end_point = np.array([[3, 3]])

interpolate_between_two_points(decoder, start_point, end_point)


# In[21]:


encoder.save('vae_encoder.h5')
decoder.save('vae_decoder.h5')
vae.save('vae_full_model.h5')


# In[22]:


vae.save_weights('vae_original_weights.h5')


# In[23]:


beta = 0.01
# Computing VAE loss
reconstruction_loss = binary_crossentropy(K.flatten(inputs), K.flatten(outputs))
reconstruction_loss *= image_shape[0] * image_shape[1]
kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
kl_loss = K.sum(kl_loss, axis=-1)
kl_loss *= -0.5 * beta
vae_loss = K.mean(reconstruction_loss + kl_loss)

#VAE model with the new loss
vae = Model(inputs, outputs, name='vae_mlp')
vae.add_loss(vae_loss)
vae.compile(optimizer='adam')


# In[24]:


class LossCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        print(f"Epoch {epoch+1}: Loss = {logs['loss']}, Validation Loss = {logs['val_loss']}")

# Train
vae.fit(train_dataset, epochs=30, validation_data=test_dataset, callbacks=[LossCallback()])


# In[28]:


def generate_and_plot_images(decoder, epoch, latent_dim):
    # Generate images
    z_new = np.random.normal(size=(10, latent_dim))
    images = decoder.predict(z_new)

    # Plot images
    fig, axes = plt.subplots(nrows=1, ncols=10, figsize=(20, 2))
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i, :, :, :], interpolation='nearest')
        ax.axis('off')
    plt.show()

#generate and plot images after each epoch
class GenerateImagesCallback(tf.keras.callbacks.Callback):
    def __init__(self, decoder, latent_dim):
        self.decoder = decoder
        self.latent_dim = latent_dim

    def on_epoch_end(self, epoch, logs=None):
        generate_and_plot_images(self.decoder, epoch, self.latent_dim)

# Instantiate
generate_images_callback = GenerateImagesCallback(decoder, latent_dim)

# Train with the new callback
vae.fit(train_dataset, epochs=2, validation_data=test_dataset, callbacks=[LossCallback(), generate_images_callback])


# In[30]:


encoder.save('vae_encoder.h5')
decoder.save('vae_decoder.h5')
vae.save('vae_full_model.h5')


# In[ ]:




