import sys
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch
import torch.optim as optim
import pandas as pd
import json

sys.path.append('code/')
from autoencoder_model import Autoencoder
from preprocess_data import preprocess_data
from stream_data import stream_batches
from train import train
from detect_anomalies import detect_anomalies

# Load configuration from JSON file
config_path = 'config.json'
with open(config_path, 'r') as config_file:
    config = json.load(config_file)

# Extract configuration variables
TRAIN_DATA = config['train_data']
TEST_DATA = config['test_data']
ANOMALY_THRESHOLD = config['anomaly_threshold']
FEATURES = config['features']
BATCH_SIZE = config['batch_size']
LEARNING_RATE = config['learning_rate']
LOSS_FUNCTION = config['loss_function']
OPTIMIZER = config['optimizer']
NUM_EPOCHS = config['num_epochs']
MAX_TEST_BATCHES = config['max_test_batches']

# Example input dimensions
input_dim = len(FEATURES)  # Based on TLC data features like fare amount, trip distance, passenger count, etc.
model = Autoencoder(input_dim)

# Load and preprocess data
df = pd.read_parquet(TRAIN_DATA)  # Example file path
processed_data = preprocess_data(df, FEATURES)

# Prepare data in batches (simulating streaming)
batch_size = BATCH_SIZE
dataset = TensorDataset(processed_data)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

loss_function_dict = {
    "mean_squared_error": nn.MSELoss(),
    "mean_absolute_error": nn.L1Loss()
}

optimzer_dict = {
    "adam": optim.Adam(model.parameters(), lr=LEARNING_RATE),
    "sgd": optim.SGD(model.parameters(), lr=LEARNING_RATE)
}

# Train the model
trained_model = train(
    model=model,
    dataloader=dataloader,
    criterion=loss_function_dict[LOSS_FUNCTION],
    optimizer=optimzer_dict[OPTIMIZER],
    num_epochs=NUM_EPOCHS,
)


# Simulate streaming data and detect anomalies
test_ctr = 0
number_anamolies = 0
for batch in stream_batches(TEST_DATA, FEATURES):
    batch_tensor = batch.clone().detach() 
    anomalies = detect_anomalies(trained_model, batch_tensor, threshold=ANOMALY_THRESHOLD)
    number_anamolies += torch.sum(anomalies).item()
    print(f"Detected anomalies in batch: {number_anamolies}")
    test_ctr += 1
    if test_ctr == MAX_TEST_BATCHES:
        break



