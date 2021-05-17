import tensorflow as tf
from tensorflow.keras.layers import Layer,Dense,Conv2D,Conv2DTranspose,Input,Flatten,Reshape
from tensorflow.keras.models import Model
from tensorflow import keras


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


def Encoder(latent_dim=40):
    """
    Build and return a convolutional encoder

    Args:
        latent_dim (int): Dimension of the latent rapresentation
        (default = 40)

    Returns:
       encoder (Model): Encoder with Convolutional and Dense layers
    """

    encoder_inputs = Input(shape=( 256, 256, 3))
    x = Conv2D(32, 4, activation="relu", strides=2, padding="same")(encoder_inputs)
    x = Conv2D(64, 4, activation="relu", strides=2, padding="same")(x)
    x = Flatten()(x)
    x = Dense(200, activation="relu")(x)
    z_mean = Dense(latent_dim, name="z_mean")(x)
    z_log_var = Dense(latent_dim, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])
    encoder = Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")

    return encoder

def Decoder():
    """Build and return a convolutional decoder"""

    latent_inputs = Input(shape=(latent_dim,))
    x = Dense(64 * 64 * 64, activation="relu")(latent_inputs)
    x = Reshape((64, 64, 64))(x)
    x = Conv2DTranspose(64, 4, activation="relu", strides=2, padding="same")(x)
    x = Conv2DTranspose(32, 4, activation="relu", strides=2, padding="same")(x)
    decoder_outputs = Conv2DTranspose(1, 3, activation="sigmoid", padding="same")(x)
    decoder = Model(latent_inputs, decoder_outputs, name="decoder")

    return decoder




class VarationalConvAE(Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VarationalConvAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.binary_crossentropy(data, reconstruction), axis=(1, 2)
                )
            )
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }
