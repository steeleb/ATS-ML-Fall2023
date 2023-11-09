settings = {
  "basic" : { 
    "hiddens": [20, 20, 20, 20, 20],
    "activations": ["relu", "relu", "relu", "relu", "relu"],
    "learning_rate": 0.001,
    "random_seed": 57,
    "max_epochs": 1000,
    "batch_size": 64,
    "patience": 200,
    "dropout_rate": 0
  },
  "super_overfit" : {
    "hiddens": [30, 30, 30, 30, 30],
    "activations": ["relu", "relu", "relu", "relu", "relu"],
    "learning_rate": 0.001,
    "random_seed": 57,
    "max_epochs": 1000,
    "batch_size": 128,
    "patience": 500,
    "dropout_rate": 0      
  }
}
