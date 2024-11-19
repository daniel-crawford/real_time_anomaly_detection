from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import torch

def preprocess_data(df, features):
    df = df[features]
    df = df.dropna()
    
    # Normalize the features
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(df)
    
    return torch.tensor(scaled_features, dtype=torch.float32)



