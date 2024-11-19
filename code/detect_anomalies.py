import torch

def detect_anomalies(model, inputs, threshold):
    with torch.no_grad():  # Disable gradient computation for inference
        reconstructed = model(inputs)
        mse = torch.mean((inputs - reconstructed) ** 2, dim=1)
        anomalies = mse > threshold
    return anomalies