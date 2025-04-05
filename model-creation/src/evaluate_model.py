import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd

from pathlib import Path


def load_model(model_path):
    return tf.keras.models.load_model(model_path)


def evaluate_model(model, X_test):
    reconstructed_X = model.predict(X_test)

    # Calculate the reconstruction error for each point in the time series
    reconstruction_error = np.square(X_test - reconstructed_X).mean(axis=1)

    err = X_test
    fig, ax = plt.subplots(figsize=(20, 6))

    a = err.loc[reconstruction_error >= np.mean(reconstruction_error)]  # anomaly
    ax.plot(err, color='blue', label='Normal')
    ax.scatter(a.index, a, color='red', label='Anomaly')
    plt.legend()

    # Save the plot
    Path("model").mkdir(parents=True, exist_ok=True)
    plt.savefig("evaluation/result.png")


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Load and evaluate an autoencoder model on a test dataset.")
    parser.add_argument("--model", type=str, required=True, help="Path to the trained model file (HDF5 format).")
    parser.add_argument("--test-dataset", type=str, required=True, help="Path to the testing dataset file (CSV format).")
    args = parser.parse_args()

    # Load the trained model
    model = load_model(args.model)

    # Load the test dataset
    X_test = pd.read_csv(args.test_dataset)

    # Evaluate the model using the test dataset
    evaluate_model(model, X_test)


if __name__ == "__main__":
    main()