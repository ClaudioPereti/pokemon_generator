import tensorflow as tf
from tensorflow.keras.layers import Layer,Dense,Conv2D,Conv2DTranspose


class Sampling(layers.Layer):
    """
    Sample from a normal distribution with mean: z_mean and sigma: z_log_var

    Attributes:
        z_mean (float): Mean of the gaussian distribution
        z_log_var (float): Variance of the gaussian distribution
        dim (int): Dimension of the gaussian distribution
        batch (int): Batch of the sample

    """

    def __init__(self,z_mean,z_log_var):
        super(Sampling,self).__init__()
        self.z_mean = z_mean
        self.z_log_var = z_log_var
        self.dim = tf.shape(z_mean)[0]
        self.batch = tf.shape(z_mean)[1]


    def call(self):
        """Return a Gaussian distribution with reparametrisation trick"""

        epsilon = tf.keras.backend.random_normal(shape=(self.batch, self.dim))
        #reparametrisation trick
        return self.z_mean + tf.exp(0.5 * self.z_log_var) * epsilon
