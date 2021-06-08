import tensorflow as tf
from tensorflow.keras.layers import Layer,Dense,Conv2D,Conv2DTranspose,Input,Flatten,Reshape
from tensorflow.keras.models import Model
from tensorflow import keras
import sys
sys.path.append('../../config')
import config
from config import encoder as encoder_config, decoder as decoder_config,conv2d_config,dense_config,conv2dtranspose_config
import functools as ft


class Sampling(Layer):
    """Sample from a normal distribution with mean: z_mean and sigma: z_log_var"""

    def call(self, inputs):
        """Return a Gaussian distribution with reparametrisation trick"""
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
        self.add_loss(kl_loss)
        self.add_metric(kl_loss,name='kl_loss',aggregation='mean')
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon



def Encoder(encoder_config):
    """
    Build and return a convolutional encoder

    Args:
        latent_dim (int): Dimension of the latent rapresentation
        (default = 200)

    Returns:
       encoder (Model): Encoder with Convolutional and Dense layers
    """

    encoder_inputs = Input(shape=encoder_config['input_shape'])
    x = Conv2D(encoder_config['filter_conv_layer'][0], encoder_config['kernel_size_conv_layer'][0], activation= encoder_config['activation_conv_layer'][0], strides= encoder_config['stride_conv_layer'][0], padding= encoder_config['padding_conv_layer'][0])(encoder_inputs)
    for index in range(1,encoder_config['conv_layers']):
        x = Conv2D(**conv2d_config(filters=encoder_config['filter_conv_layer'][index]))(x)


    x = Flatten()(x)
    for index in range(encoder_config['dense_layers']):
        x = Dense(**dense_config(units=encoder_config['node_dense_layer'][index]))(x)

    latent_dim = encoder_config['latent_dim']
    z_mean = Dense(latent_dim, name="z_mean")(x)
    z_log_var = Dense(latent_dim, name="z_log_var")(x)

    encoder = Model(encoder_inputs, [z_mean, z_log_var], name="encoder")

    return encoder

def Decoder(decoder_config):
    """
    Build and return a convolutional decoder

    Args:
        latent_dim (int): Dimension of the latent rapresentation
        (default = 200)

    Returns:
       dencoder (Model): Decoder with Convolutional and Dense layers
       """

    latent_inputs = Input(shape=(decoder_config['latent_dim'],))
    x = Dense(ft.reduce(lambda x,y:x*y,decoder_config['reshape_layer']), activation="relu")(latent_inputs)
    x = Reshape(decoder_config['reshape_layer'])(x)
    for index in range(decoder_config['conv_transpose_layers']):
        x = Conv2DTranspose(**conv2dtranspose_config(filters =decoder_config['filter_conv_transpose_layers'][index]))(x)

    decoder_outputs = Conv2DTranspose(3, 3, activation="sigmoid", padding="same")(x)
    decoder = Model(latent_inputs, decoder_outputs, name="decoder")

    return decoder




class VarationalConvAE(Model):
    """ Combines encoder,sampling and decoder to build a variational autoencoder"""
    def __init__(self, **kwargs):
        super(VarationalConvAE, self).__init__(**kwargs)
        self.encoder = Encoder(encoder_config)
        self.decoder = Decoder(decoder_config)
        self.sampling = Sampling()


    def call(self,x):
        z_mean, z_log_var = self.encoder(x)
        z = self.sampling([z_mean, z_log_var])
        reconstruction = self.decoder(z)
        reconstruction_loss = tf.reduce_mean(
            tf.reduce_sum(
                keras.losses.binary_crossentropy(x, reconstruction), axis=(1, 2)
            )
        )
        self.add_loss(reconstruction_loss)
        self.add_metric(reconstruction_loss,name='reconstruction_loss',aggregation='mean')
        return reconstruction
