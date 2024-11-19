import pandas as pd
import time

from preprocess_data import preprocess_data

def stream_batches(file_path, features, batch_size=32):
    df = pd.read_parquet(file_path)
    processed_data = preprocess_data(df, features)
    for i in range(0, len(processed_data), batch_size):
        yield processed_data[i:i+batch_size]
        time.sleep(1)  # Simulate stream arrival every second
