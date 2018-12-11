#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 22 11:06:08 2018

@author: TrungKien
"""

import numpy as np
import matplotlib.pyplot as plt 
import os.path

## torch
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


numberPoint = 15
sizePoint   = 3

## TO CORRECT
#def add_Noise(data):
#    batch_size  = data.shape[0]
#    dim         = data.shape[2]
#    if numberPoint>1:
#        ### GENERATING Noise MNIST
#        epsilon = torch.rand(batch_size, numberPoint)
#        for isample in range(batch_size):
#            x_noise = torch.round(epsilon[isample]*(dim-sizePoint)*(dim-sizePoint)).long()
#            x_sub   = torch.arange(sizePoint)
#            x_sub   = x_sub.repeat(sizePoint,1)
#            x_ind  = (x_sub*dim+x_sub.transpose(0,1)).flatten()
#            N       = torch.ones((dim*dim))
#            for ind in list(x_ind):
#                N[x_noise+ind] = 0
#            data[isample].mul_(N.reshape(1,dim,dim))
#            
#        return data
#    else:
#        return data

def add_Noise(data, prob=0.5):
    """salt and pepper noise for mnist"""
    rnd = torch.rand_like(data)
    noisy = torch.ones_like(data).float()
    noisy[rnd < prob] = 0.
    noisy_data = data.mul(noisy)
    return noisy_data    

def loss_function(X_recon,X,mu,logvar):
    BCE = F.binary_cross_entropy(X_recon,X.view(-1,784),reduction = 'sum')
    KLD = -0.5*torch.sum(1+logvar-mu.pow(2)-logvar.exp())
    #KLD = torch.min(KLD,torch.FloatTensor([1.0e16]))
    return BCE+KLD

def loss_L1(X_recon,X):
    loss = nn.L1Loss()
    return loss(X_recon,X.view(-1,784))#(torch.sum(torch.abs(X_recon-X))/X.numel()).item()

        
class DVAE(nn.Module):
    def __init__(self,checkpoint = None,
                 image_size = 28,
                 intermediate_dim = 512,
                 latent_dim = 16,
                 batch_size = 128,
                 epoch = 50,
                 noise = None):
        super(DVAE, self).__init__()
        self.checkpoint = checkpoint
        self.encoder 	= None
        self.decoder 	= None
        self.vae 		= None
        self.noiseModel= noise
        ## PARAMS
        self.inputDim  = image_size*image_size
        self.intermidiateDim = intermediate_dim
        self.latentDim = latent_dim
        self.batchSize = batch_size
        self.epoch = epoch
        ## LAYERS
        self.dense1  = nn.Linear(self.inputDim,self.intermidiateDim)
        self.dense21 = nn.Linear(self.intermidiateDim,self.latentDim)
        self.dense22 = nn.Linear(self.intermidiateDim,self.latentDim)
        #sampling
        self.dense3 = nn.Linear(self.latentDim,self.intermidiateDim)
        self.dense4 = nn.Linear(self.intermidiateDim,self.inputDim)
 
        
    ###########################################################
    ## RETURN
    def get_corruptedInputExample(self,X):
        if self.noiseModel is None:
            return X
        else:
            return self.noiseModel(X)
    ###########################################################
    ## MODEL
    def encoding(self,X):
        ## encoder
        X = F.relu(self.dense1(X))
        return self.dense21(X), self.dense22(X)
    def sampling(self,mu,logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)
    def decoding(self,z):
        H = F.relu(self.dense3(z))
        return torch.sigmoid(self.dense4(H))
    
    def denoising(self,X):
        ## encoding
        mu,logvar = self.encoding(X.view(-1,self.inputDim))
        ## sampling
        z   = self.sampling(mu,logvar)
        ## decoding and return
        return self.decoding(z)
        
    def forward(self,X):
        ## corrupting inputs
        XC = self.get_corruptedInputExample(X)
        ## encoding
        mu,logvar = self.encoding(XC.view(-1,self.inputDim))
        ## sampling
        z   = self.sampling(mu,logvar)
        ## decoding and return
        return self.decoding(z),mu,logvar
        
    #############################################################
    ## STORING
    def save(self):
        torch.save(self.state_dict(),self.checkpoint)
    
    def load(self,PATH):
        self.load_state_dict(torch.load(PATH))
        self.eval()
    #############################################################
    ## TRAINING
    def fit(self,optimizer,train_loader,valid_loader):
        trainLoss = []
        validLoss = []
        for epoch in range(self.epoch):
            ## training
            trainL = self.learning(epoch,train_loader,optimizer)
            # evaluating
            validL = self.testing(epoch,valid_loader)
            # getting loss value
            trainLoss.append(trainL)
            validLoss.append(validL)
        return trainLoss, validLoss
    
    def lossL1(self,data_loader):
        loss = 0.0
        for idx, data in enumerate(data_loader):
            # get the inputs
            inputs, labels = data
#            inputs = Variable(inputs)
            # get predicting
            outputs,mu,std = self.forward(inputs)
            # getting error
            loss += loss_L1(outputs,inputs)
        return loss/len(data_loader.dataset) 
            
    
    def learning(self,epoch,train_loader,optimizer):
        self.train()
        train_loss = 0.0
        for i, data in enumerate(train_loader):
            # get the inputs
            inputs, labels = data
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs,mu,std = self.forward(inputs)
            loss = loss_function(outputs,inputs,mu,std)
            #loss = criterion(outputs, inputs.view(-1,1,784))
            loss.backward()
            optimizer.step()
        
            # print statistics
            train_loss += loss.item()
            if i % 60 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, i * len(inputs), len(train_loader.dataset),
                100. * i / len(train_loader),
                loss.item() / len(inputs)))
        train_loss /= len(train_loader.dataset)
        print ('Epoch {}: Train Cost:{:.4f}'.format(epoch,train_loss))
        return train_loss
    
    def testing(self,epoch,test_loader):
        self.eval()
        test_loss = 0.0
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                # get the inputs
                inputs, labels= data
                # forward + backward + optimize
                outputs,mu,std = self.forward(inputs)
                loss = loss_function(outputs,inputs,mu,std)
            #    print statistics
                test_loss += loss.item()
        test_loss /= len(test_loader.dataset)
        print ('Test Cost:{:.4f}'.format(test_loss))
        return test_loss

if __name__ == '__main__':
    ## MAIN function
    batch_size = 128
    # loading data
    #(x_train, y_train), (x_test, y_test) = mnist.load_data()
    trans = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0., ), (1.0,))])
    trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                            download=True, transform = trans)
    testset = torchvision.datasets.MNIST(root='./data', train=False,
                                            download=True, transform = trans)
    train_loader = torch.utils.data.DataLoader(dataset = trainset,
                                               batch_size = batch_size,
                                               shuffle = True)
    test_loader = torch.utils.data.DataLoader(dataset = testset,
                                               batch_size = batch_size,
                                               shuffle = False)
    print 'training {0} samples .vs testing {1} samples'.format(
            len(train_loader),len(test_loader))
    
    
    # displaying data
    #for data_idx,(x,label) in enumerate(train_loader):
    #    print data_idx
    # loading model
    checkpoint_dir = 'model/'
    if not os.path.isdir(checkpoint_dir):
            os.mkdir(checkpoint_dir)
    checkpoint = os.path.join(checkpoint_dir, "dvae_mlp_mnist.h5")
    model = DVAE(checkpoint,epoch = 30, noise = add_Noise)
    # training
    #optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    optimizer = optim.Adam(model.parameters(),lr=1e-3)  
    trainLoss, testLoss = model.fit(optimizer,train_loader,test_loader)

    
    data_iter = iter(test_loader)
    batch_data,Y = data_iter.next()
    ## corrupted inputs
    corrupted_data = model.get_corruptedInputExample(batch_data)
    
    from torchvision.utils import save_image
    recon_batch= model.denoising(corrupted_data)
    n = min(batch_data.size(0), 100)
    comparison = torch.cat([corrupted_data[:n],
                          recon_batch.view(batch_size, 1, 28, 28)[:n]])
    save_image(comparison.cpu(),
             'test_DVAE_torch.png', nrow=10, normalize=True)
    
    nn.L1Loss()(corrupted_data.view(-1,784),batch_data.view(-1,784)).item()
    nn.L1Loss()(recon_batch.view(-1,784),batch_data.view(-1,784)).item()
