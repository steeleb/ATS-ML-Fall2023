settings = {
  "basic_overfit" : { 
    "codings": 5,
    "encoder_hiddens": [20, 10],
    "encoder_activations": ["relu", "relu"],
    "decoder_hiddens": [10, 20],
    "decoder_activations": ["relu", "relu"],
    "learning_rate": 0.001,
    "random_seed": 57,
    "max_epochs": 1000,
    "batch_size": 64,
    "patience": 200,
    "dropout_rate": 0.2
  }
}