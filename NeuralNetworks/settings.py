settings = {
  "basic" : { 
    "hiddens": [10, 10, 10],
    "activations": ["relu", "relu", "relu"],
    "learning_rate": 0.001,
    "random_seed": 57,
    "max_epochs": 1000,
    "batch_size": 32,
    "patience": 50,
    "dropout_rate": 0
  },
  "with_dropout" : { 
    "hiddens": [10, 10, 10],
    "activations": ["relu", "relu", "relu"],
    "learning_rate": 0.001,
    "random_seed": 57,
    "max_epochs": 1000,
    "batch_size": 32,
    "patience": 50,
    "dropout_rate": 0.25
  }
}
