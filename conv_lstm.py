"""

## Introduction

The
[Convolutional LSTM](https://papers.nips.cc/paper/2015/file/07563a3fe3bbe7e3ba84431ad9d055af-Paper.pdf)
architectures bring together time series processing and computer vision by
introducing a convolutional recurrent cell in a LSTM layer. In this example, we will explore the
Convolutional LSTM model in an application to next-frame prediction, the process
of predicting what video frames come next given a series of past frames.
"""

"""
## Setup
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend as K

import io
import os
import imageio
from IPython.display import Image, display
from ipywidgets import widgets, Layout, HBox

"""
## Dataset Construction

For this, we will be using the
[Moving MNIST](http://www.cs.toronto.edu/~nitish/unsupervised_video/)
dataset.

We will download the dataset and then construct and
preprocess training and validation sets.

For next-frame prediction, our model will be using a previous frame,
which we'll call `f_n`, to predict a new frame, called `f_(n + 1)`.
To allow the model to create these predictions, we'll need to process
the data such that we have "shifted" inputs and outputs, where the
input data is frame `x_n`, being used to predict frame `y_(n + 1)`.
"""

# Download and load the dataset.
fpath = keras.utils.get_file(
    "moving_mnist.npy",
    "http://www.cs.toronto.edu/~nitish/unsupervised_video/mnist_test_seq.npy",
)
dataset = np.load(fpath)

# Swap the axes representing the number of frames and number of data samples.
dataset = np.swapaxes(dataset, 0, 1)
# We'll pick out 1000 of the 10000 total examples and use those.
dataset = dataset[:1000, ...]
# Add a channel dimension since the images are grayscale.
dataset = np.expand_dims(dataset, axis=-1)

# Split into train and validation sets using indexing to optimize memory.
indexes = np.arange(dataset.shape[0])
np.random.shuffle(indexes)
train_index = indexes[: int(0.9 * dataset.shape[0])]
val_index = indexes[int(0.9 * dataset.shape[0]) :]
train_dataset = dataset[train_index]
val_dataset = dataset[val_index]

# Normalize the data to the 0-1 range.
train_dataset = train_dataset / 255
val_dataset = val_dataset / 255

# We'll define a helper function to shift the frames, where
# `x` is frames 0 to n - 1, and `y` is frames 1 to n.
def create_shifted_frames(data):
    x = data[:, 0 : data.shape[1] - 1, :, :]
    y = data[:, 1 : data.shape[1], :, :]
    return x, y


# Apply the processing function to the datasets.
x_train, y_train = create_shifted_frames(train_dataset)
x_val, y_val = create_shifted_frames(val_dataset)

# parsing

parser = argparse.ArgumentParser(description='ConvLSTM Next Frame Prediction')

# model
parser.add_argument('--model_name', type=str, default='convlstm')
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--batch_size', type=int, default=5)

# regularization
parser.add_argument('--dropout', type=float, default=0.0)
parser.add_argument('--recurrent_dropout', type=float, default=0.0)
parser.add_argument('--kernel_regularizer', type=str, default=None)

# saving files
parser.add_argument('--experiment_name', type=str, default='exp1', help='filename to save the training experiment data')
parser.add_argument('--checkpoint_filename', type=str, default='_{epoch:02d}-{val_loss:.4f}.hdf5')

args = parser.parse_args()

"""
## Model Construction

To build a Convolutional LSTM model, we will use the
`ConvLSTM2D` layer, which will accept inputs of shape
`(batch_size, num_frames, width, height, channels)`, and return
a prediction movie of the same shape.
"""

# Construct the input layer with no definite frame size.
inp = layers.Input(shape=(None, *x_train.shape[2:]))

# We will construct 3 `ConvLSTM2D` layers with batch normalization,
# followed by a `Conv3D` layer for the spatiotemporal outputs.
x = layers.ConvLSTM2D(
    filters=64,
    kernel_size=(5, 5),
    padding="same",
    return_sequences=True,
    dropout=args.dropout,
    recurrent_dropout=args.recurrent_dropout,
    kernel_regularizer=args.kernel_regularizer,
    activation="relu",
)(inp)
x = layers.BatchNormalization()(x)
x = layers.ConvLSTM2D(
    filters=64,
    kernel_size=(3, 3),
    padding="same",
    dropout=args.dropout,
    recurrent_dropout=args.recurrent_dropout,
    kernel_regularizer=args.kernel_regularizer,
    return_sequences=True,
    activation="relu",
)(x)
x = layers.BatchNormalization()(x)
x = layers.ConvLSTM2D(
    filters=64,
    kernel_size=(1, 1),
    padding="same",
    dropout=args.dropout,
    recurrent_dropout=args.recurrent_dropout,
    kernel_regularizer=args.kernel_regularizer,
    return_sequences=True,
    activation="relu",
)(x)
x = layers.Conv3D(
    filters=1, kernel_size=(3, 3, 3), activation="sigmoid", padding="same"
)(x)


"""
## Metrics for images
"""
def PSNR(y_true, y_pred):
    psnr = 0
    max_pixel = 1.0
    for i in range(19):
      psnr += tf.image.psnr(y_pred[:, i, ...], y_true[:, i, ...], max_val=1.0)

    psnr = psnr / 19
    return K.mean(psnr, axis=-1)

def SSIM(y_true, y_pred):
    ssim = 0
    for i in range(19):
      ssim += tf.image.ssim(y_pred[:,i, ...], y_true[:, i, ...], 
                            max_val=1.0, filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03)
    ssim = ssim / 19  
    return K.mean(ssim, axis=-1)

# Next, we will build the complete model and compile it.
model = keras.models.Model(inp, x)
model.compile(
    loss=keras.losses.binary_crossentropy, optimizer=keras.optimizers.Adam(),
    metrics=[PSNR, SSIM])
"""
## Model Training

With our model and data constructed, we can now train the model.
"""

# Define some callbacks to improve training.
early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss", patience=10)
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=5)

# Create a callback that saves the model's weights every 5 epochs
checkpoint_filepath = '.convlstm_results/model_save/'
checkpoint_filepath = checkpoint_filepath + args.experiment_name + args.checkpoint_filename
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_loss',
    mode='min',
    save_best_only=True, verbose=1)

# Fit the model to the training data.
history = model.fit(
    x_train,
    y_train,
    batch_size=args.batch_size,
    epochs=args.epochs,
    validation_data=(x_val, y_val),
    callbacks=[early_stopping, checkpoint_callback, reduce_lr],
)


# Save the training history
df = pd.DataFrame(data=history.history)
filename = '.convlstm_results/training_result/' + args.experiment_name + '.csv'
df.to_csv(filename, index=False)


"""
## Predicted Videos

Finally, we'll pick a few examples from the validation set
and construct some GIFs with them to see the model's
predicted videos.
"""

# Select a few random examples from the dataset.
examples = val_dataset[np.random.choice(range(len(val_dataset)), size=5)]

# Iterate over the examples and predict the frames.
predicted_videos = []
idx = 0
for example in examples:
    # Pick the first/last ten frames from the example.
    frames = example[:10, ...]
    original_frames = example[10:, ...]
    new_predictions = np.zeros(shape=(10, *frames[0].shape))

    # Predict a new set of 10 frames.
    for i in range(10):
        # Extract the model's prediction and post-process it.
        frames = example[: 10 + i + 1, ...]
        new_prediction = model.predict(np.expand_dims(frames, axis=0))
        new_prediction = np.squeeze(new_prediction, axis=0)
        predicted_frame = np.expand_dims(new_prediction[-1, ...], axis=0)

        # Extend the set of prediction frames.
        new_predictions[i] = predicted_frame

    # Create and save GIFs for each of the ground truth/prediction images.
    for frame_set in [original_frames, new_predictions]:
        # Construct a GIF from the selected video frames.
        current_frames = np.squeeze(frame_set)
        current_frames = current_frames[..., np.newaxis] * np.ones(3)
        current_frames = (current_frames * 255).astype(np.uint8)
        current_frames = list(current_frames)

        # Construct a GIF from the frames.
        with io.BytesIO() as gif:
            imageio.mimsave(gif, current_frames, "GIF", fps=5)
            imageio.mimsave(f'.convlstm_results/prediction/prediction_{idx}.gif', current_frames)
            predicted_videos.append(gif.getvalue())
            idx += 1
