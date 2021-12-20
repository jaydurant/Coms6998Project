# Coms6998Project

## Regularization of RNNs and LSTMs

The aim of this project is to dentify regularization methods to increase performance of for RNNs and LSTMS. This is done by training two RNN and LSTM  Models with varying regularization methods and hyperparameters (e.g. Optimizer, Epochs, Batch Size) in order to construct models with increased robustness. The value is in reducing overfitting on a variety of tasks can be reduced and more complex RNN and LSTM Models can be utilized for increased performance on a whole host of tasks.

This repo contains two models for RNN and LSTMS networks

- PredRNN++
- ConvLSTM

Utilizing Moving MNIST  dataset for training and validation

The dataset contains 10,000 sequences used for training and validation each of length 20 and 2 digits in 64x64 frames.

This repo has various regularization methods applied to each of the models applied above. For PredRNN++, the model can be trained by running the following command. Arguments in this file can be changed to change regularization method and number of iterations to run.

```
sh predrnn_mnist_train.sh
```

By editing the commands in this file various regularization methods can be applied.

For this repo the directories for predrnn are described below

- models: contains the files needed to build and run predrnn++
- results/mnist_predrnn: contains gif results from testing
- layers: contains the spatial temporal lstm cell
- utils: contains functions to compute metrics, preprocess frames, and train the model




