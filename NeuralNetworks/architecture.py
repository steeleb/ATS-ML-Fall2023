# high level modules
import imp
# ml/ai modules
import tensorflow as tf
# Let's import some different things we will use to build the neural network
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Input, Dropout, Softmax

def build_model(x_train, y_train, settings):
  # create input layer
  input_layer = tf.keras.layers.Input(shape=x_train.shape[1:])
  
  # # create hidden layers each with specific number of nodes
  # assert len(settings["hiddens"]) == len(
  #   settings["activations"]
  # ), "hiddens and activations settings must be the same length."
  
  # add dropout layer
  layers = tf.keras.layers.Dropout(rate=settings["dropout_rate"])(input_layer)
  
  for hidden, activation in zip(settings["hiddens"], settings["activations"]):
    layers = tf.keras.layers.Dense(
      units=hidden,
      activation=activation,
      use_bias=True,
      kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0, l2=0),
      bias_initializer=tf.keras.initializers.RandomNormal(seed=settings["random_seed"]),
      kernel_initializer=tf.keras.initializers.RandomNormal(seed=settings["random_seed"]),
      )(layers)
  
  # create output layer
  output_layer = tf.keras.layers.Dense(
    units=1,
    activation="linear",
    use_bias=True,
    bias_initializer=tf.keras.initializers.RandomNormal(seed=settings["random_seed"] + 1),
    kernel_initializer=tf.keras.initializers.RandomNormal(seed=settings["random_seed"] + 2),
  )(layers)
  
  # construct the model
  model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
  model.summary()
  
  return model


def compile_model(model, settings):
  model.compile(
      optimizer=tf.keras.optimizers.Adam(
        learning_rate=settings["learning_rate"],
        ),
      loss=tf.keras.losses.MeanSquaredError()
  )
  return model

