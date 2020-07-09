#!/usr/bin/env python
# coding: utf-8
import numpy as np
import torch
from torchvision import datasets
import torchvision.transforms as transforms
import torch.nn as nn
import cv2
import config
from cnn_model import Denoiser
import matplotlib.pyplot as plt

def load_data():
    
    # convert data to torch.FloatTensor
    transform = transforms.ToTensor()
    
    # load the training and test datasets
    train_data = datasets.MNIST(root='data', train=True, download=True, transform=transform)
    test_data = datasets.MNIST(root='data', train=False, download=True, transform=transform)
    
    # prepare data loaders
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=config.batch_size, num_workers=config.num_workers)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=config.batch_size, num_workers=config.num_workers)
    
    return train_loader, test_loader


def train(train_loader, model):
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    for epoch in range(1, config.n_epochs + 1):
        train_loss = 0.0
        
        for data in train_loader:
            images, labels = data
            
            noisy_images = add_noise(images)
                    
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing *noisy* images to the model
            denoised_images = model(noisy_images)
            # calculate the loss
            # the "target" is still the original, not-noisy images
            loss = criterion(denoised_images, images)
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # update running training loss
            train_loss += loss.item() * images.size(0)
                
        # print avg training statistics 
        train_loss = train_loss /len(train_loader)
        print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch, train_loss))
        

def add_noise(images):
    """Adds noise to images"""
    noisy_images = images + config.noise_factor * torch.randn(*images.shape)
    noisy_images = np.clip(noisy_images, 0., 1.)
    return noisy_images



def show_images(noisy_images, denoised_images):

    # output is resized into a batch of iages
    denoised_images = denoised_images.view(config.batch_size, 1, 28, 28)
    # use detach when it's an output that requires_grad
    denoised_images = denoised_images.detach().numpy()
    
    # plot the first ten input images and then reconstructed images
    fig, axes = plt.subplots(nrows=2, ncols=10, sharex=True, sharey=True, figsize=(25,4))
    
    # input images on top row, reconstructions on bottom
    for noisy_images, row in zip([noisy_images, denoised_images], axes):
        for img, ax in zip(noisy_images, row):
            ax.imshow(np.squeeze(img), cmap='gray')
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            plt.show()



if __name__ == '__main__':
    train_loader, test_loader = load_data()
    
    # Initialize model
    model = Denoiser()
    print(model)
    print('Start training...')
    train(train_loader, model)
    
    # test on test images
    # obtain one batch of test images
    dataiter = iter(test_loader)
    test_images, labels = dataiter.next()
    noisy_images_test = add_noise(test_images)
    # get sample outputs
    denoised_images_test = model(noisy_images_test)
    # prep images for display
    noisy_images_test = noisy_images_test.numpy()
    show_images(noisy_images_test, denoised_images_test)


   
