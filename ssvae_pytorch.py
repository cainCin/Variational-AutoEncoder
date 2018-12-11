#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  7 15:14:28 2018

@author: TrungKien
Based on https://github.com/uber/pyro/blob/dev/examples/vae/ss_vae_M2.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.utils import save_image

## to correct
from mnist_cached import MNISTCached, mkdir_p, setup_data_loaders

def loss_function(X_recon,X,mu,logvar,Y_pred=None,Y = None):
    if Y is not None:
        BCE1 = F.binary_cross_entropy(Y_pred,Y,reduction = 'sum')
    else:
        BCE1 = 0.0
    BCE2 = F.binary_cross_entropy(X_recon,X.view(-1,784),reduction = 'sum')
    BCE = 0.1*BCE1+BCE2
    KLD = -0.5*torch.sum(1+logvar-mu.pow(2)-logvar.exp())
    return BCE+KLD

def losses_sup(xs,ys):
    # zero the parameter gradients
    optimizer.zero_grad()
    # forward + backward + optimize
    outputs,mu,std, = model.forward(xs,ys)
    y_pred = model.encoding_y(xs)
    # loss_function
    loss = loss_function(outputs,xs,mu,std,y_pred,ys)
    loss.backward()
    optimizer.step()
    return loss.item()

def losses_unsup(xs):
    # zero the parameter gradients
    optimizer.zero_grad()
    # forward + backward + optimize
    outputs,mu,std = model.forward(xs)
    # loss_function
    loss = loss_function(outputs,xs,mu,std)
    loss.backward()
    optimizer.step()
    return loss.item()

class SSVAE(nn.Module):
    """
    """
    def __init__(self, output_size=10, 
                         input_size=784, 
                         latent_size=50, 
                         hidden_size=512,
                         epochs = 50,
                         disp   = None,
                         batch_size = 200,
                         ):

        super(SSVAE, self).__init__()

        ## VARIABLES: initialize the class with all arguments provided to the constructor
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.output_size = output_size
        self.number_epoch= epochs
        self.batch_size = batch_size
        self.disp = disp
        # define and instantiate the neural networks representing
        # the paramters of various distributions in the model
        ## LAYERS 
        # ENCODE Y
        self.dense_y1   = nn.Linear(self.input_size,self.hidden_size)
        self.dense_y2   = nn.Linear(self.hidden_size,self.output_size)
        # ENCODE Z
        self.dense_z1   = nn.Linear(self.input_size + self.output_size,self.hidden_size)
        self.dense_z21  = nn.Linear(self.hidden_size,self.latent_size)
        self.dense_z22  = nn.Linear(self.hidden_size,self.latent_size)
        # DECODE
        self.dense3     = nn.Linear(self.latent_size,self.hidden_size)
        self.dense4     = nn.Linear(self.hidden_size,self.input_size)
    
    ## Model
    def encoding_y(self,x):
        H   = F.relu(self.dense_y1(x))
        return F.softmax(self.dense_y2(H),1)
    
    def classifying(self,xs):
        alpha = self.encoding_y(xs)
        return torch.distributions.one_hot_categorical.OneHotCategorical(alpha).sample()
        
    def encoding_z(self,xs,ys = None):
        if ys is None:
            alpha = self.encoding_y(xs)
            ys = torch.distributions.one_hot_categorical.OneHotCategorical(alpha).sample()
        H   = F.relu(self.dense_z1(torch.cat([xs,ys],1)))
        mu  = self.dense_z21(H)
        logvar = self.dense_z22(H)
        return mu,logvar
    
    def decoding(self,Z):
        H = F.relu(self.dense3(Z))
        return torch.sigmoid(self.dense4(H))
    
    def sampling(self,mu,logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)
    
    def forward(self,xs,ys = None):
        mu,logvar   = self.encoding_z(xs.view(-1,self.input_size),ys)
        z   = self.sampling(mu,logvar)
        return self.decoding(z),mu,logvar  
    
    ## Training phase ###############################
    def train_epoch(self,data_loaders,periodic_interval_batches):
        self.train()
        # compute number of batches for an epoch
        sup_batches = len(data_loaders["sup"])
        unsup_batches = len(data_loaders["unsup"])
        batches_per_epoch = sup_batches + unsup_batches
    
        # initialize variables to store loss values
        epoch_losses_sup = 0.
        epoch_losses_unsup = 0.
    
        # setup the iterators for training data loaders
        sup_iter = iter(data_loaders["sup"])
        unsup_iter = iter(data_loaders["unsup"])
        
        # count the number of supervised batches seen in this epoch
        ctr_sup = 0
        for i in range(batches_per_epoch):
    
            # whether this batch is supervised or not
            is_supervised = (i % periodic_interval_batches == 1) and ctr_sup < sup_batches
    
            # extract the corresponding batch
            if is_supervised:
                (xs, ys) = next(sup_iter)
                ctr_sup += 1
            else:
                (xs, ys) = next(unsup_iter)
    
            # run the inference for each loss with supervised or un-supervised
            # data as arguments
            if is_supervised:
                new_loss = losses_sup(xs, ys)
                epoch_losses_sup += new_loss
            else:
                new_loss = losses_unsup(xs)
                epoch_losses_unsup += new_loss
    
        # return the values of all losses
        return epoch_losses_sup, epoch_losses_unsup
    
    def loss_epoch(self,data_loader,batch_size,disp = None):
        """
        compute the accuracy over the supervised validation set or the testing set
        """
        self.eval()
        predictions, actuals = [], []
    
        # use the appropriate data loader
        for i,(xs, ys) in enumerate(data_loader):
            # use classification function to compute all predictions for each batch
            predictions.append(self.classifying(xs))
            actuals.append(ys)
            if disp is not None and i==0:
                xs_recon,_,_ = model.forward(xs)
                n = min(xs.size(0), 100)
                comparison = torch.cat([xs.view(batch_size, 1, 28, 28)[:n],
                                      xs_recon.view(batch_size, 1, 28, 28)[:n]])
                save_image(comparison.cpu(),disp, nrow=10)
    
        # compute the number of accurate predictions
        accurate_preds = 0
        for pred, act in zip(predictions, actuals):
            for i in range(pred.size(0)):
                v = torch.sum(pred[i] == act[i])
                accurate_preds += (v.item() == 10)
    
        # calculate the accuracy between 0 and 1
        accuracy = (accurate_preds * 1.0) / (len(predictions) * batch_size)
        return accuracy
    
    def fit(self,data_loaders,periodic_interval_batches):
        # initializing local variables to maintain the best validation accuracy
        # seen across epochs over the supervised training set
        # and the corresponding testing set and the state of the networks
        best_valid_acc, corresponding_test_acc = 0.0, 0.0
        valid_acc = []
        test_acc = []
        # run inference for a certain number of epochs
        for i in range(0, self.number_epoch):
            model.train()
            # get the losses for an epoch
            epoch_losses_sup, epoch_losses_unsup = \
                self.train_epoch(data_loaders, periodic_interval_batches)
    
            # compute average epoch losses i.e. losses per example
            avg_epoch_losses_sup = epoch_losses_sup/sup_num
            avg_epoch_losses_unsup = epoch_losses_unsup/unsup_num
            # store the loss and validation/testing accuracies in the logfile
            str_loss_sup = " {}".format(avg_epoch_losses_sup)
            str_loss_unsup = " {}".format(avg_epoch_losses_unsup)
            str_print = "epoch {}: avg losses {}\n".format(i, "{} {}".format(str_loss_sup, str_loss_unsup))
            
            validation_accuracy = self.loss_epoch(data_loaders["valid"], self.batch_size)
            str_print += " validation accuracy {}\n".format(validation_accuracy)
            valid_acc.append(validation_accuracy)
    
            # this test accuracy is only for logging, this is not used
            # to make any decisions during training
            test_accuracy = self.loss_epoch(data_loaders["test"], self.batch_size, 
                                         disp = 'temps/recon_'+str(i)+'.png')
            str_print += " test accuracy {}\n".format(test_accuracy)
            test_acc.append(test_accuracy)
            # update the best validation accuracy and the corresponding
            # testing accuracy and the state of the parent module (including the networks)
            if best_valid_acc < validation_accuracy:
                best_valid_acc = validation_accuracy
                corresponding_test_acc = test_accuracy
            print str_print 
        str_print = "best validation accuracy {} corresponding testing accuracy {} ".format(best_valid_acc, corresponding_test_acc)
        print str_print 
        return valid_acc, test_acc



if __name__ == "__main__":
    ## ARGS
    sup_num = 500
    num_epochs = 50
    batch_size = 200
    ## load mnist cached
    data_loaders = setup_data_loaders(MNISTCached, 
                                      use_cuda = False, 
                                      batch_size = batch_size, 
                                      sup_num=sup_num)
    periodic_interval_batches = int(MNISTCached.train_data_size / (1.0 * sup_num))

    # number of unsupervised examples
    unsup_num = MNISTCached.train_data_size - sup_num
    
    model = SSVAE(batch_size = 200,epochs = num_epochs)
    optimizer = optim.Adam(model.parameters(),lr=0.00042, betas = (0.9,0.999)) 
    
    valid_acc,test_acc = model.fit(data_loaders,periodic_interval_batches)
    
    import matplotlib.pyplot as plt 
    plt.figure(figsize=(8,5))
    plt.plot(range(num_epochs),valid_acc,'r--',
         range(num_epochs),test_acc,'b--')
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.grid()
    plt.legend(["valid accuracy","test accuracy"])
    plt.title('Semi-supervised learning via VAE (500 labels .vs 49500 unlabels)')
    plt.savefig('temps/accuracy.png')
    plt.show()
    
    
        

        