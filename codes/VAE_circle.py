import tensorflow as tf
from tensorflow import GradientTape, exp, keras, reduce_mean, reduce_sum, shape, square
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.losses import Huber, binary_crossentropy, mse
from tensorflow.keras.metrics import Mean
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


latent_dim = 8


def encoder_model_circle(inp_shape=(2821,), latent_dim=latent_dim):
    encoder_inputs = keras.Input(shape=inp_shape)
    x = layers.Reshape((2821, 1))(encoder_inputs)
    x = layers.ZeroPadding1D((379, 0))(x)
    x = layers.Conv1D(16, activation="relu", kernel_size=9, strides=1, padding="same")(
        x
    )
    x = layers.Conv1D(16, activation="relu", kernel_size=9, strides=1, padding="same")(
        x
    )
    x = layers.Conv1D(16, activation="relu", kernel_size=9, strides=4, padding="same")(
        x
    )
    x = layers.Conv1D(32, activation="relu", kernel_size=9, strides=1, padding="same")(
        x
    )
    x = layers.Conv1D(32, activation="relu", kernel_size=9, strides=4, padding="same")(
        x
    )
    x = layers.Conv1D(64, activation="relu", kernel_size=9, strides=1, padding="same")(
        x
    )
    x = layers.Conv1D(64, activation="relu", kernel_size=9, strides=5, padding="same")(
        x
    )
    x = layers.Flatten()(x)
    x = layers.Dense(1024, activation="relu")(x)
    x = layers.Dense(512, activation="relu")(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dense(16, activation="relu")(x)

    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    z = Sampling(name="z")([z_mean, z_log_var])

    return encoder_inputs, z_mean, z_log_var, z


def decoder_model_circle(latent_dim=latent_dim):
    latent_inputs = keras.Input(shape=(latent_dim,), name="z_sampling")

    x = layers.Dense(16, activation="relu")(latent_inputs)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dense(512, activation="relu")(x)
    x = layers.Dense(1024, activation="relu")(x)
    x = layers.Dense(
        2560,
        activation="relu",
        kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
        bias_regularizer=regularizers.L2(1e-4),
        activity_regularizer=regularizers.L2(1e-5),
    )(x)
    x = layers.Reshape((40, 64))(x)

    x = layers.Conv1DTranspose(
        64, activation="relu", kernel_size=9, strides=5, padding="same"
    )(x)
    x = layers.Conv1DTranspose(
        32, activation="relu", kernel_size=9, strides=1, padding="same"
    )(x)
    x = layers.Conv1DTranspose(
        32, activation="relu", kernel_size=9, strides=4, padding="same"
    )(x)
    x = layers.Conv1DTranspose(
        16, activation="relu", kernel_size=9, strides=1, padding="same"
    )(x)
    x = layers.Conv1DTranspose(
        16, activation="relu", kernel_size=9, strides=4, padding="same"
    )(x)
    x = layers.Conv1DTranspose(
        1, activation="relu", kernel_size=9, strides=1, padding="same"
    )(x)

    x = layers.Cropping1D(cropping=(379, 0))(x)
    decoder_outputs = layers.Flatten()(x)

    return latent_inputs, decoder_outputs


class VAE(Model):
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
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
                tf.keras.losses.mean_squared_error(data, reconstruction)
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

    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstruction = self.decoder(z)
        reconstruction_loss = reduce_mean(binary_crossentropy(inputs, reconstruction))
        reconstruction_loss *= 1912
        kl_loss = 1 + z_log_var - square(z_mean) - exp(z_log_var)
        kl_loss = reduce_mean(kl_loss)
        kl_loss *= -0.5
        total_loss = reconstruction_loss + kl_loss
        self.add_metric(kl_loss, name="kl_loss", aggregation="mean")
        self.add_metric(total_loss, name="total_loss", aggregation="mean")
        self.add_metric(
            reconstruction_loss, name="reconstruction_loss", aggregation="mean"
        )
        return reconstruction


def vae_circle():
    encoder_inputs, z_mean, z_log_var, z = encoder_model_circle()
    encoder = Model(encoder_inputs, (z_mean, z_log_var, z), name="VAE_encoder")
    encoder.summary()

    decoder_inputs, decoder_outputs = decoder_model_circle()
    decoder = Model(decoder_inputs, decoder_outputs, name="VAE_decoder")
    decoder.summary()

    return VAE(encoder, decoder)


vae = vae_circle()
