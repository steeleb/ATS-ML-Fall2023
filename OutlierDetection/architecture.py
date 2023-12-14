# ml/ai modules
import tensorflow as tf
# Let's import some different things we will use to build the neural network
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Input, Dropout, Softmax

class Sampling(tf.keras.layers.Layer):
  def call(self, inputs):
    mean, log_var = inputs
    return tf.random.normal(tf.shape(log_var)) * tf.exp(log_var / 2) + mean

def build_encoder(x_train, settings):
  # create input layer
  input_layer = tf.keras.layers.Input(shape=[x_train.shape[1]])

  # add dropout layer
  layers = tf.keras.layers.Dropout(rate=settings["dropout_rate"])(input_layer)

  # create hidden layers of encoder
  for hidden, activation in zip(settings["encoder_hiddens"], settings["encoder_activations"]):
    layers = tf.keras.layers.Dense(
      units=hidden,
      activation=activation,
      use_bias=True,
      kernel_regularizer=tf.keras.regularizers.l1_l2(l1=settings["l1"], l2=settings["l2"]),
      bias_initializer=tf.keras.initializers.RandomNormal(seed=settings["random_seed"]),
      kernel_initializer=tf.keras.initializers.RandomNormal(seed=settings["random_seed"]),
      )(layers)
  
  # calculate mean and log variance of latent space
  codings_mean = tf.keras.layers.Dense(settings["codings"])(layers) #mu
  codings_log_var = tf.keras.layers.Dense(settings["codings"])(layers) #log_var

  # sample from latent space
  codings = Sampling()([codings_mean, codings_log_var])

  # construct the model
  variational_encoder = tf.keras.Model(
    inputs=[input_layer], 
    outputs=[codings_mean, codings_log_var, codings]
    )
  variational_encoder.summary()
  
  return variational_encoder

def build_decoder(x_train, settings):
  decoder_input = tf.keras.layers.Input(shape=[settings["codings"]])

  layers = decoder_input

  # create hidden layers of decoder
  for hidden, activation in zip(settings["decoder_hiddens"], settings["decoder_activations"]):
    layers = tf.keras.layers.Dense(
      units=hidden,
      activation=activation,
      use_bias=True,
      kernel_regularizer=tf.keras.regularizers.l1_l2(l1=settings["l2"], l2=settings["l2"]),
      bias_initializer=tf.keras.initializers.RandomNormal(seed=settings["random_seed"]),
      kernel_initializer=tf.keras.initializers.RandomNormal(seed=settings["random_seed"]),
      )(layers)
  
  # create output layer
  output = tf.keras.layers.Dense(
    units=x_train.shape[1],
    activation="linear",
    use_bias=True,
    kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0, l2=0),
    bias_initializer=tf.keras.initializers.RandomNormal(seed=settings["random_seed"]),
    kernel_initializer=tf.keras.initializers.RandomNormal(seed=settings["random_seed"]),
    )(layers)

  variational_decoder = tf.keras.Model(
    inputs=[decoder_input], 
    outputs=[output]
    )
  
  variational_decoder.summary()

  return variational_decoder

# define loss function
def latent_loss(codings_log_var, codings_mean):
  ll = -0.5 * tf.reduce_sum(
    1 + codings_log_var - tf.exp(codings_log_var) - tf.square(codings_mean),
    axis=-1)
  return ll


# define VAE model, adapted from https://keras.io/examples/generative/vae/
class VAE(tf.keras.Model):
  def __init__(self, encoder, decoder, **kwargs):
    super().__init__(**kwargs)
    self.encoder = encoder
    self.decoder = decoder
    self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
    self.reconstruction_loss_tracker = tf.keras.metrics.Mean(
      name="reconstruction_loss"
    )
    self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")

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
          tf.keras.losses.mean_squared_error(data, reconstruction)
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
  
  def test_step(self, data):
    z_mean, z_log_var, z = self.encoder(data)
    reconstruction = self.decoder(z)
    # Compute reconstruction loss
    reconstruction_loss = tf.reduce_mean(
        tf.reduce_sum(
            tf.keras.losses.mean_squared_error(data, reconstruction)
        )
    )
    # Compute KL divergence
    kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
    kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
    # Compute total loss
    total_loss = reconstruction_loss + kl_loss
    # Update the metrics
    self.total_loss_tracker.update_state(total_loss)
    self.reconstruction_loss_tracker.update_state(reconstruction_loss)
    self.kl_loss_tracker.update_state(kl_loss)
    # Return a dict mapping metric names to current value
    return {
        "loss": self.total_loss_tracker.result(),
        "reconstruction_loss": self.reconstruction_loss_tracker.result(),
        "kl_loss": self.kl_loss_tracker.result(),
    }
  
  def call(self, inputs):
    z_mean, z_log_var, z = self.encoder(inputs)
    reconstruction = self.decoder(z)
    return reconstruction


def compile_model(model, settings):
  model.compile(
      optimizer=tf.keras.optimizers.Adam(
        learning_rate=settings["learning_rate"],
        ),
      loss=tf.keras.losses.MeanSquaredError()
  )
  return model

