image_dim = 256*256*3

latent_dim = 20

from tensorflow.keras.layers import Input,Lambda, Flatten,Dense, Conv2D, Conv2DTranspose
from tensorflow.keras.models import Model
import tensorflow as tf
import numpy as np
from tensorflow.keras import backend as K

def sampling(args):
    z_mean,z_log_sigma = args
    epsilon = K.random_normal(shape = (z_mean.shape[0],latent_dim),mean = 0, stddev = 0.1)

    return z_mean + K.exp(z_log_sigma)*epsilon
#%%
input = Input(shape = (256,256,3))
x = Conv2D(filters = 32,kernel_size=(3,3),strides = 2,activation='relu')(input)
x = Conv2D(filters = 64,kernel_size=(3,3),strides = 2,activation='relu')(x)
x = Flatten()(x)

z_mean = Dense(latent_dim,activation='relu')(x)
z_log_sigma = Dense(latent_dim,activation='relu')(x)
z = Lambda(sampling)([z_mean,z_log_sigma])

encoder = Model(inputs = input, outputs = [z_mean,z_log_sigma],name = 'encoder')

#%%
latent_inputs = Input(shape = (latent_dim,),name = 'z_sampling')
y = Dense(64*74*74)(latent_inputs)
y = Conv2DTranspose(filters=64, kernel_size=3, strides=2, padding='same',
                activation='relu')(y)

y = Conv2DTranspose(filters=32, kernel_size=3, strides=2, padding='same',
                activation='relu')(y)
            # No activation
output= Conv2DTranspose(filters=1, kernel_size=3, strides=1, padding='same')(y)

decoder = Model(inputs = latent_inputs,outputs = output,name = 'decoder')

#%%

output_decoder = decoder(encoder(input)[2])
vae = Model(input,output_decoder,name = 'vae')
