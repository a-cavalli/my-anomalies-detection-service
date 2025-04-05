import argparse
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from pathlib import Path


def build_model(shape):
    # Define the input layer
    input_layer = tf.keras.layers.Input(shape=(shape,))

    # Define the encoding layers
    encoded = tf.keras.layers.Dense(32, activation='relu')(input_layer)
    encoded = tf.keras.layers.Dense(16, activation='relu')(encoded)

    # Define the decoding layers
    decoded = tf.keras.layers.Dense(16, activation='relu')(encoded)
    decoded = tf.keras.layers.Dense(32, activation='relu')(decoded)

    # Define the output layer
    output_layer = tf.keras.layers.Dense(shape)(decoded)

    # Create the autoencoder model
    autoencoder = tf.keras.models.Model(input_layer, output_layer)

    # Compile the model
    autoencoder.compile(optimizer='adam', loss='mean_squared_error')

    return autoencoder


def train_model(X_train):
    # Fit the model to the time series data
    model = build_model(X_train.shape[1])
    history = model.fit(X_train, X_train, epochs=10, batch_size=32)

    # Save the model
    Path("model").mkdir(parents=True, exist_ok=True)
    model.save("model/anomalies_detection_model.h5")

    return history


def plot_loss(history):
    plt.plot(history.history['loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')

    # Save the plot
    Path("evaluation").mkdir(parents=True, exist_ok=True)
    plt.savefig('evaluation/training_loss.png')


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Train an autoencoder model on the given dataset.")
    parser.add_argument("--train-dataset", type=str, required=True, help="Input path to the training dataset file (CSV format).")
    args = parser.parse_args()

    # Load the dataset
    X_train = pd.read_csv(args.train_dataset)

    # Train the model and get the training history
    history = train_model(X_train)

    # Plot and save the training loss
    plot_loss(history)


if __name__ == "__main__":
    main()