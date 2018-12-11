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
from torch.autograd import Variable


## TO CORRECT
def sampling(args):
    """Reparameterization trick by sampling fr an isotropic unit Gaussian.
    # Arguments:
        args (tensor): mean and log of variance of Q(z|X)
    # Returns:
        z (tensor): sampled latent vector
    """
    z_mean, z_log_var = args
    batch = z_mean.size()[0]
    dim = z_mean.size()[1]
    # by default, random_normal has mean=0 and std=1.0
    m = torch.distributions.normal.Normal(0,1.0)
    epsilon = m.sample(sample_shape = (batch,dim))
    return z_mean + torch.exp(0.5 * z_log_var) * epsilon
    

def show_MNIST(args):
    img = args
    figure = np.reshape(img,(28,28))
    plt.figure(figsize=(5, 5))
    plt.imshow(figure, cmap='Greys_r')
    plt.show()
## CLASS def
#class LOSS(nn.Module):
#    def __init__(self):
#        super(LOSS, self).__init__()
#        self.mse_loss = nn.MSELoss(reduction = "sum")
#    def forward(self,X_recon,X,mu,logvar):
#        mse = self.mse_loss(X_recon,X)
#        KLD = -0.5*torch.sum(1+logvar-mu**2-logvar.exp())
#        return mse+KLD
#loss_mse = LOSS()
def loss_function(X_recon,X,mu,logvar):
    BCE = F.binary_cross_entropy(X_recon,X.view(-1,784),reduction = 'sum')
    KLD = -0.5*torch.sum(1+logvar-mu.pow(2)-logvar.exp())
    #KLD = torch.min(KLD,torch.FloatTensor([1.0e16]))
    return BCE+KLD

def loss_L1(X_recon,X):
    loss = nn.L1Loss()
    return loss(X_recon,X.view(-1,784))#(torch.sum(torch.abs(X_recon-X))/X.numel()).item()

        
class VAE(nn.Module):
    def __init__(self,checkpoint = None,
                 image_size = 28,
                 intermediate_dim = 512,
                 latent_dim = 16,
                 batch_size = 128,
                 epoch = 50):
        super(VAE, self).__init__()
        self.checkpoint = checkpoint
        self.encoder 	= None
        self.decoder 	= None
        self.vae 		= None
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
    
    def forward(self,X):
        mu,logvar = self.encoding(X.view(-1,self.inputDim))
        z   = self.sampling(mu,logvar)
        return self.decoding(z),mu,logvar
        
    
    def save(self):
        torch.save(self.state_dict(),self.checkpoint)
    
    def load(self,PATH):
        self.load_state_dict(torch.load(PATH))
        self.eval()

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
    checkpoint = os.path.join(checkpoint_dir, "vae_mlp_mnist.h5")
    model = VAE(checkpoint,epoch = 40)
    # training
    #optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    optimizer = optim.Adam(model.parameters(),lr=1e-3)  
    trainLoss, testLoss = model.fit(optimizer,train_loader,test_loader)
    
#    plt.figure(figsize=(5, 10))
#    plt.plot(range(30),trainLoss,'r--',
#             range(30),testLoss,'b--')
#    plt.show()
    
    data_iter = iter(test_loader)
    batch_data,Y = data_iter.next()
    from torchvision.utils import save_image
    recon_batch,mu,var = model(batch_data)
    n = min(batch_data.size(0), 100)
    comparison = torch.cat([batch_data[:n],
                          recon_batch.view(batch_size, 1, 28, 28)[:n]])
    save_image(comparison.cpu(),
             'test_VAE_torch.png', nrow=10, normalize=True)
#print 'saving model at {}'.format(checkpoint)
#model.save()
#if not os.path.isdir(checkpoint_dir):
#    print 'saving model at {}'.format(checkpoint)
#    model.save()
#else:
#    print 'loading model from {}'.format(checkpoint)
#    model.load(checkpoint)
#if __name__ == '__main__':
    #(x_train, y_train), (x_test, y_test) = mnist.load_data()
