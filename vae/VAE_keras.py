# LIBRARIES
import numpy as np
import matplotlib.pyplot as plt 
import os.path
import argparse

from keras.models import Model
from keras.layers import Input, Lambda
from keras.layers import Conv2D, Activation, MaxPool2D, Flatten, Dense
from keras import backend as K
from keras.losses import mse, binary_crossentropy
#from keras.models import Sequential

# Test Data
from keras.datasets import mnist


# DEFAULTS:
#############################################################    
# reparameterization trick
# instead of sampling from Q(z|X), sample eps = N(0,I)
# z = z_mean + sqrt(var)*eps
def sampling(args):
	"""Reparameterization trick by sampling fr an isotropic unit Gaussian.
	# Arguments:
	    args (tensor): mean and log of variance of Q(z|X)
	# Returns:
	    z (tensor): sampled latent vector
	"""
	z_mean, z_log_var = args
	batch = K.shape(z_mean)[0]
	dim = K.int_shape(z_mean)[1]
	# by default, random_normal has mean=0 and std=1.0
	epsilon = K.random_normal(shape=(batch, dim))
	return z_mean + K.exp(0.5 * z_log_var) * epsilon
# CLASS definition
class VAE(object):
  def __init__(self, 
               image_size=28,
               intermediate_dim = 512,
               batch_size = 128,
               latent_dim = 2,
               epochs = 50,
               checkpoint=None, 
               sample_dir=None):
	# INITIALIZING
    self.original_dim = image_size * image_size
    self.intermediate_dim = intermediate_dim
    self.batch_size = batch_size
    self.latent_dim = latent_dim
    self.epochs 	= epochs

    self.checkpoint = checkpoint
    self.sample_dir = sample_dir
    self.encoder 	= None
    self.decoder 	= None
    self.vae 		= None

  def build_model(self,args):# VAE model = encoder + decoder
      # build encoder model
      input_shape = (self.original_dim, )
      inputs = Input(shape=input_shape, name='encoder_input')
      x = Dense(self.intermediate_dim, activation='relu')(inputs)
      z_mean = Dense(self.latent_dim, name='z_mean')(x)
      z_log_var = Dense(self.latent_dim, name='z_log_var')(x)
      # use reparameterization trick to push the sampling out as input
      # note that "output_shape" isn't necessary with the TensorFlow backend
      z = Lambda(sampling, output_shape=(self.latent_dim,), name='z')([z_mean, z_log_var])
      # instantiate encoder model
      encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
      self.encoder = encoder
      # build decoder model
      latent_inputs = Input(shape=(self.latent_dim,), name='z_sampling')
      x = Dense(self.intermediate_dim, activation='relu')(latent_inputs)
      outputs = Dense(self.original_dim, activation='sigmoid')(x)
      # instantiate decoder model
      decoder = Model(latent_inputs, outputs, name='decoder')
      self.decoder = decoder
      # instantiate VAE model
      outputs = decoder(encoder(inputs)[2])
      vae = Model(inputs, outputs, name='vae_mlp')
      # VAE loss = mse_loss or xent_loss + kl_loss
      if args.mse:
          reconstruction_loss = mse(inputs, outputs)
      else:
          reconstruction_loss = binary_crossentropy(inputs,
													outputs)
          
      reconstruction_loss *= original_dim
      kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
      kl_loss = K.sum(kl_loss, axis=-1)
      kl_loss *= -0.5
      vae_loss = K.mean(reconstruction_loss + kl_loss)
      vae.add_loss(vae_loss)
      vae.compile(optimizer='adam')
      self.vae = vae
      
      return self.vae
  
  def train(self,args):
    load_weight,x_train,y_train,x_test,y_test = args
    if load_weight:
        if os.path.isfile(self.checkpoint):
            # LOADING model
            self.vae.load_weights(self.checkpoint)
    else:
		# TRAINING model
		self.vae.fit(x_train,
                epochs=self.epochs,
                batch_size=self.batch_size,
                validation_data=(x_test, None))
		# STORING model
		self.vae.save_weights(self.checkpoint)
    return self.vae

  def print_model(self):
    print self.vae.summary()

  def plot_input(self,data):
    x_test, y_test = data
    # display a 30x30 2D manifold of digits
    n = np.sqrt(x_test.shape[0]).astype(int)
    digit_size = np.sqrt(self.original_dim).astype(int)
    figure = np.zeros((digit_size * n, digit_size * n))

    for i in range(n):
        for j in range(n):
            digit = x_test[i*n+j].reshape(digit_size, digit_size)
            figure[i * digit_size: (i + 1) * digit_size,
                   j * digit_size: (j + 1) * digit_size] = digit
    ### DISPLAY
    figPath = 'fig'
    if not os.path.isdir(figPath):
        os.mkdir(figPath)
    filename = os.path.join(figPath, "input_MNIST.png")
    plt.figure(figsize=(5, 5))
    plt.imshow(figure, cmap='Greys_r')
    plt.savefig(filename)
    plt.show()
    
  def plot_output(self,data):
    x_test, y_test = data
    # display a 100x100 2D manifold of digits
    n = np.sqrt(x_test.shape[0]).astype(int)
    digit_size = np.sqrt(self.original_dim).astype(int)
    figure = np.zeros((digit_size * n, digit_size * n))

    for i in range(n):
        for j in range(n):
            x_decoded = self.vae.predict(np.array([x_test[i*n+j]]),
                                         batch_size=self.batch_size)
            digit = x_decoded[0].reshape(digit_size, digit_size)
            figure[i * digit_size: (i + 1) * digit_size,
                   j * digit_size: (j + 1) * digit_size] = digit

    ### DISPLAY
    figPath = 'fig'
    if not os.path.isdir(figPath):
        os.mkdir(figPath)
    filename = os.path.join(figPath, "output_MNIST.png")
    plt.figure(figsize=(5, 5))
    plt.imshow(figure, cmap='Greys_r')
    plt.savefig(filename)
    plt.show()

  def plot_latent(self,data):
    x_test, y_test = data  
    z_mean, _, _ = self.encoder.predict(x_test,
                                   batch_size=self.batch_size)
    plt.figure(figsize=(12, 10))
    plt.scatter(z_mean[:, 0], z_mean[:, 1], c=y_test)
    plt.colorbar()
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.show()
		
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    help_ = "Load h5 model trained weights"
    parser.add_argument("-w", "--weights", help=help_, action='store_true')
    help_ = "Use mse loss instead of binary cross entropy (default)"
    parser.add_argument("-m",
                        "--mse",
                        help=help_, action='store_true')
    args = parser.parse_args()
    
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    image_size = x_train.shape[1]
    original_dim = image_size * image_size
    x_train = np.reshape(x_train, [-1, original_dim])
    x_test = np.reshape(x_test, [-1, original_dim])
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255
    x_train.shape
    # BUILT Model
    C       = VAE(image_size=image_size, 
                  epochs = 40,
                  latent_dim = 16,
                  checkpoint='vae_mlp_mnist.h5')
    model   = C.build_model(args)
    # TRAINING model
    model   = C.train((args.weights,x_train,y_train,x_test,y_test))
    # PRINTING model
    C.print_model()
    # DISPLAY
    C.plot_input((x_test[0:36],1))
    C.plot_output((x_test[0:36],1))
    C.plot_latent((x_test,y_test))
	
    