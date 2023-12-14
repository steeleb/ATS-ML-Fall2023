settings = {
  "overfit" : { 
    "codings": 5,
    "encoder_hiddens": [20, 10],
    "encoder_activations": ["relu", "relu"],
    "decoder_hiddens": [10, 20],
    "decoder_activations": ["relu", "relu"],
    "learning_rate": 0.001,
    "random_seed": 57,
    "max_epochs": 5000,
    "batch_size": 64,
    "patience": 100,
    "dropout_rate": 0.2,
    "l1": 0,
    "l2": 0
  },
  "basic" : {
      "codings": 3,
      "encoder_hiddens": [20, 10],
      "encoder_activations": ["relu", "relu"],
      "decoder_hiddens": [10, 20],
      "decoder_activations": ["relu", "relu"],
      "learning_rate": 0.001,
      "random_seed": 57,
      "max_epochs": 5000,
      "batch_size": 32,
      "patience": 100,
      "dropout_rate": 0.2,
      "l1": 0,
      "l2": 0
  }
}