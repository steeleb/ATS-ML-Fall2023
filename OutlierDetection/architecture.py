# high level modules
import imp
# ml/ai modules
import tensorflow as tf
# Let's import some different things we will use to build the neural network
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Input, Dropout, Softmax

class Sampling(tf.keras.layers.Layer):
  def call(self, inputs):
    mean, log_var = inputs
    return tf.random.normal(tf.shape(log_var)) * tf.exp(log_var / 2) + mean

def build_encoder(x_train, y_train, settings):
  # create input layer
  input_layer = tf.keras.layers.Input(shape=x_train.shape[1:])
  
  # create hidden layers of encoder
  for hidden, activation in zip(settings["encoder_hiddens"], settings["encoder_activations"]):
    layers = tf.keras.layers.Dense(
      units=hidden,
      activation=activation,
      use_bias=True,
      kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0, l2=0),
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

def build_decoder(x_train, y_train, settings):
  decoder_input = tf.keras.layers.Input(shape=[settings["codings"]])

  # create hidden layers of decoder
  for hidden, activation in zip(settings["decoder_hiddens"], settings["decoder_activations"]):
    layers = tf.keras.layers.Dense(
      units=hidden,
      activation=activation,
      use_bias=True,
      kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0, l2=0),
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
latent_loss = -0.5 * tf.reduce_sum(
    1 + codings_log_var - tf.exp(codings_log_var) - tf.square(codings_mean),
    axis=-1
)

def build_VAE(x_train, y_train, settings):
  _, _, codings = build_encoder(x_train, y_train, settings)
  reconstruction = build_decoder(x_train, y_train, settings)(codings)
  VAE = tf.keras.Model(
    inputs=[x_train], 
    outputs=[reconstruction]
    )
  VAE.add_loss(tf.reduce_mean(latent_loss) / x_train.shape[1])
  VAE.summary()
  return VAE

def compile_model(model, settings):
  model.compile(
      optimizer=tf.keras.optimizers.Adam(
        learning_rate=settings["learning_rate"],
        ),
      loss=tf.keras.losses.MeanSquaredError()
  )
  return model

