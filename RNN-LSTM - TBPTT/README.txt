This folder contains the implementation of RNN and LSTM emulators for time-series regression in the northern Adriatic Sea.

To run the ML models the run_training.py script must be launched. This code reads the config.yaml file ('configs' folder), in which the following can be defined: output directory, model to use (the definition of the models can be found in models.py ('src' folder)), model configuration parameters, loss function, among other.

The data folder contains the predictand and predictors files, the last ones constructed based on the coarse data ('data/coarse data'). For each location the scripts to construct its predictors are provided.