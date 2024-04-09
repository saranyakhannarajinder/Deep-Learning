#!/usr/bin/env python
# coding: utf-8

# # 1. GAN model to generate images from the dataset

# # Loading the libraries required and storing the path of the image in attribute

# In[25]:


import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


surprise_path=r"C:\Users\saran\Downloads\FaceExpressions\dataset\Surprise"
sad_path=r"C:\Users\saran\Downloads\FaceExpressions\dataset\Sad"
neutral_path=r"C:\Users\saran\Downloads\FaceExpressions\dataset\Neutral"
happy_path=r"C:\Users\saran\Downloads\FaceExpressions\dataset\Happy"
angry_path= r"C:\Users\saran\Downloads\FaceExpressions\dataset\Angry"


# In[26]:


class FacialExpressionDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path)
        if self.transform:
            image = self.transform(image)
        return image

# Define transformations
transform = transforms.Compose([
    transforms.Resize((64, 64)),  
    transforms.ToTensor(),         
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  
])

# Collect image paths for all expressions
all_image_paths = []
all_image_paths.extend([os.path.join(surprise_path, img) for img in os.listdir(surprise_path)])
all_image_paths.extend([os.path.join(sad_path, img) for img in os.listdir(sad_path)])
all_image_paths.extend([os.path.join(neutral_path, img) for img in os.listdir(neutral_path)])
all_image_paths.extend([os.path.join(happy_path, img) for img in os.listdir(happy_path)])
all_image_paths.extend([os.path.join(angry_path, img) for img in os.listdir(angry_path)])

# Create dataset
dataset = FacialExpressionDataset(all_image_paths, transform=transform)

# Create DataLoader
batch_size = 64
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


# # Defining a function for generator and discriminator class

# In[30]:


class Generator(nn.Module):
    def __init__(self, nz, ngf, nc):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            
            # Add more layers as needed
            
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.main(x)

class Discriminator(nn.Module):
    def __init__(self, nc, ndf):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x)


# In[32]:


latent_size = 128


# In[33]:


discriminator = nn.Sequential(
    # 3 x 64 x 64

    nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(64),
    nn.LeakyReLU(0.2, inplace=True),
    # 64 x 32 x 32

    nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(128),
    nn.LeakyReLU(0.2, inplace=True),
    # 128 x 16 x 16

    nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(256),
    nn.LeakyReLU(0.2, inplace=True),
    # 256 x 8 x 8

    nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(512),
    nn.LeakyReLU(0.2, inplace=True),
    # 512 x 4 x 4

    nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0, bias=False),
    # 1 x 1 x 1

    nn.Flatten(),
    nn.Sigmoid()) # we generally use sigmoid function at the end of all network to finally 
                # get the bounded result


# In[40]:


# Define the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# In[41]:


# Assuming 'device' is already defined
discriminator = discriminator.to(device)


# In[42]:


def device_loader(model, device):
    return model.to(device)


# In[44]:


generator = nn.Sequential(
    

    nn.ConvTranspose2d(latent_size, 512, kernel_size=4, stride=1, padding=0, bias=False),
    nn.BatchNorm2d(512),
    nn.ReLU(inplace=True),
   

    nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(256),
    nn.ReLU(inplace=True),
  

    nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(128),
    nn.ReLU(inplace=True),
  

    nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(64),
    nn.ReLU(inplace=True),
   

    nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1, bias=False),
    nn.Tanh()
   
)


# In[46]:


# Assuming 'device' is already defined
generator = generator.to(device)


# In[47]:


def device_loader(model, device):
    return model.to(device)


# In[48]:


generator = device_loader(generator, device)


# In[54]:


import torch.nn.functional as F
def train_discriminator(r_images,optimizer):
    
    optimizer.zero_grad()
    # prediction
    # setting the target for prediction
    # calculating loss
    # getting score
    r_preds=discriminator(r_images)
    r_targets=torch.ones(r_images.size(0),1,device=device)
    r_loss = F.binary_cross_entropy(r_preds, r_targets)
    r_score = torch.mean(r_preds).item()

    
    latent = torch.randn(batch_size, latent_size, 1, 1, device=device)
    g_images = generator(latent) 
    # Pass fake images through discriminator
    g_targets = torch.zeros(g_images.size(0), 1, device=device)
    g_preds = discriminator(g_images)
    g_loss = F.binary_cross_entropy(g_preds, g_targets)
    g_score = torch.mean(g_preds).item()

    # Updating discriminator weights
    loss = r_loss + g_loss
    loss.backward()
    optimizer.step()
    return loss.item(), r_score,g_score
  


# In[63]:


def train_generator(optimizer):
    optimizer.zero_grad()
    
    # Generating fake images
    latent = torch.randn(batch_size, latent_size, 1, 1, device=device)
    g_images = generator(latent)
    
    # Passing fake images through the discriminator
    preds = discriminator(g_images)
    
    # Setting targets for the generator 
    targets = torch.ones(batch_size, 1, device=device)
    
    # Calculating loss 
    loss = F.binary_cross_entropy(preds, targets)
    
    loss.backward()
    
    optimizer.step()
    
    return loss.item()


# In[64]:


def denormalized(img):
    # Denormalize the image tensor 
    mean = torch.tensor([0.5, 0.5, 0.5]) 
    std = torch.tensor([0.5, 0.5, 0.5])  
    img = img * std[:, None, None] + mean[:, None, None]
    return img


# In[66]:


latent_noise = torch.randn(64, latent_size, 1, 1, device=device)
os.makedirs('new_gen_img', exist_ok=True)
def fit(epochs, lr, index=1):
    
    torch.cuda.empty_cache()
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    
    
    for epoch in range(epochs):
        for r_images, _ in tqdm(dataloader):
            loss_d, r_score, g_score = train_discriminator(r_images, d_optimizer)
            loss_g = train_generator(g_optimizer)
        
        # saving images with save_image module of torchvision
        g_images=generator(latent_noise)
        fimg_name='images-{0:0=4d}.png'.format(epoch+index,)
        save_image(denormalized(g_images),os.path.join('new_gen_img',fimg_name),nrow=4)
        
        print(" {} out of {} epochs, loss_g: {:.4f}, loss_d: {:.4f}, real_score: {:.4f}, fake_score: {:.4f}".format(
            epoch+1, epochs, loss_g, loss_d, r_score, g_score))


# In[67]:


records=fit(10,0.0002)


# In[68]:


Image.open('./new_gen_img/images-0010.png')


# In[69]:


records=fit(50,0.0002)


# In[70]:


Image.open('./new_gen_img/images-0050.png')


# In[ ]:




