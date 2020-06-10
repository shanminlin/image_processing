# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 22:29:40 2020

@author: SS
"""
import matplotlib.pyplot as plt                        
import numpy as np
from glob import glob
import torch
import torchvision.models as models
from PIL import Image, ImageFile 
import torchvision.transforms as transforms
import os
import os.path
from torchvision import datasets
import config
import torch.optim as optim
import torch.nn as nn

def preprocess(data_dir):
    
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    
    batch_size = config.batch_size
    data_dir = config.data_dir
    
    data_transforms = {'train': transforms.Compose([transforms.RandomResizedCrop(224),
                                                    transforms.RandomHorizontalFlip(),
                                                    transforms.RandomRotation(10),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize(mean = [0.485, 0.456, 0.406], 
                                                                         std = [0.229, 0.224, 0.225])]), 
                       'valid': transforms.Compose([transforms.Resize(256),
                                                    transforms.CenterCrop(224),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize(mean = [0.485, 0.456, 0.406], 
                                                                         std = [0.229, 0.224, 0.225])]), 
                       'test': transforms.Compose([transforms.Resize(256),
                                                   transforms.CenterCrop(224),
                                                   transforms.ToTensor(),
                                                   transforms.Normalize(mean = [0.485, 0.456, 0.406], 
                                                                   std = [0.229, 0.224, 0.225])])}
    
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), 
                                              data_transforms[x]) for x in ['train', 'valid', 'test']}
    
    loaders = {x: torch.utils.data.DataLoader(image_datasets[x], 
                                              batch_size= batch_size, 
                                              shuffle = True) for x in ['train', 'valid', 'test']}
    return image_datasets, loaders

def modified_model(image_datasets):

    model = models.resnet18(pretrained = True)
    
    for param in model.parameters():
        param.requires_grad = False
    
    num_of_input_features = model.fc.in_features
    num_of_classes = len(image_datasets['train'].classes)
    
    model.fc = nn.Linear(num_of_input_features, num_of_classes)
    
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        model = model.cuda()
        
    print(model)
    return model

def train(model, loaders, save_path):
    """returns trained model"""
    # cross entropy loss for classification task
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    
    # initialize tracker for minimum validation loss
    valid_loss_min = np.Inf 
    
    n_epochs = config.n_epochs
    for epoch in range(1, n_epochs+1):
        # initialize variables to monitor training and validation loss
        train_loss = 0.0
        valid_loss = 0.0

        model.train()
        for batch_idx, (data, target) in enumerate(loaders['train']):
            # move to GPU
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            # average training loss
            train_loss += (1 / (batch_idx + 1)) * (loss.data - train_loss)
        
        # vaidation
        model.eval()
        for batch_idx, (data, target) in enumerate(loaders['valid']):
            # move to GPU
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()
            ## update the average validation loss
            output = model(data)
            loss = criterion(output, target)
            valid_loss += (1 / (batch_idx + 1)) * (loss.data - valid_loss)

            
        # print training/validation statistics 
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(epoch, train_loss, valid_loss))
        
        # save the model if validation loss has decreased
        if valid_loss <= valid_loss_min:
            torch.save(model.state_dict(), save_path)
            
            # Updating the validation loss minimum
            valid_loss_min = valid_loss
    
    # return trained model
    return model

 

def test(loaders, model):

    # monitor test loss and accuracy
    test_loss = 0.
    correct = 0.
    total = 0.
    
    criterion = nn.CrossEntropyLoss()
    
    model.eval() # !!! must include!!!
    
    for batch_idx, (data, target) in enumerate(loaders['test']):
        # move to GPU
        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the loss
        loss = criterion(output, target)
        # update average test loss 
        test_loss = test_loss + ((1 / (batch_idx + 1)) * (loss.data - test_loss))
        # convert output probabilities to predicted class
        pred = output.data.max(1, keepdim=True)[1]
        # compare predictions to true label
        correct += np.sum(np.squeeze(pred.eq(target.data.view_as(pred))).cpu().numpy())
        total += data.size(0)
            
    print('Test Loss: {:.6f}\n'.format(test_loss))

    print('\nTest Accuracy: %2d%% (%2d/%2d)' % (100. * correct / total, correct, total))


def predict_breed(model, img_path, class_names):
    # load the image and return the predicted breed
    img = Image.open(img_path)
    
    data_transforms = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])
    
    img_preprocessed = data_transforms(img)
    img_preprocessed.unsqueeze_(0)
    
    image = img_preprocessed.cuda()
    
    model.eval()
    
    output = model(image)
    
    _, prediction = torch.max(output.data,1)
    predict_breed = class_names[prediction-1]
    plt.imshow(Image.open(img_path))
    plt.show()
    print ('You look like a {}'.format(predict_breed))

if __name__ == '__main__':
    # Load data
    images = np.array(glob(config.input_image_dir))
    print('There are {} total dog images.'.format(len(images)))
    image_datasets, loaders = preprocess(config.data_dir)
    
    model = modified_model(image_datasets)
    
    # Check if trained model exists
    if os.path.exists('model.pt'):
        model.load_state_dict(torch.load('model.pt'))
        
    else:   
        model = train(model, loaders, 'model.pt')

    
    test(loaders, model)
    
    # list of class names by index so that name can be accessed like class_names[0]
    class_names = [item[4:].replace("_", " ") for item in image_datasets['train'].classes]

    for file in images[0:4]:
        predict_breed(model, file, class_names)