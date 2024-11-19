# Real Time Anomaly Detection with pyTorch Autoencoder

## Project Overview

This project focuses on real-time anomaly detection using a pyTorch-based autoencoder. Anomaly detection is crucial in various domains such as fraud detection, network security, and predictive maintenance. The autoencoder model is trained to reconstruct normal data patterns, and anomalies are identified when the reconstruction error exceeds a certain threshold.

## Features

- Real-time data processing
- Customizable autoencoder architecture
- Threshold-based anomaly detection

## Installation

To get started, clone the repository and install the required dependencies:

```bash
git clone https://github.com/yourusername/real_time_anomaly_detection.git
cd real_time_anomaly_detection
pip install -r requirements.txt
```

## Config

"train_data": This specifies the path to the training dataset. In this case, it points to a Parquet file containing data for training the model.
Demo: NYC TLC OCT-23 (yellow_tripdata_2023-10) [https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page]

"test_data": This specifies the path to the testing dataset. It points to a Parquet file containing data for testing the model.
Demo: NYC TLC NOV-23 (yellow_tripdata_2023-11) [https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page]

"anomaly_threshold": This value is used to determine the threshold for classifying data points as anomalies. A lower value means stricter anomaly detection.
Demo: Common default values range from 0.01 to 0.05, but it depends on the 

specific use case.
"features": This is a list of feature names that will be used as input for the model. These are the columns in your dataset that the model will learn from.
Demo: ["fare_amount", "trip_distance", "passenger_count"].

"batch_size": This defines the number of samples that will be propagated through the network at one time. A larger batch size can lead to faster training but requires more memory.
Demo: 32 is a common default value.

"learning_rate": This is the step size at each iteration while moving toward a minimum of the loss function. It controls how much to change the model in response to the estimated error each time the model weights are updated.
Demo: 0.001 is a common default value.

"loss_function": This specifies the function that the model will use to evaluate the error between the predicted output and the actual output. "mean_squared_error" is commonly used for regression tasks. 
Demo: "mean_squared_error", other option is "mean_absolute_error"

"optimizer": This defines the algorithm to use for updating the model parameters. "adam" is a popular choice because it adapts the learning rate during training.
Default: "adam" is a common default value, other option is "sgd"

"num_epochs": This specifies the number of complete passes through the training dataset. More epochs can lead to better training but also increase the risk of overfitting.
Demo: 10 is a common starting point, but it can vary widely depending on the problem.

"max_test_batches": This limits the number of batches to use during testing. It can be useful to speed up testing by not using the entire test set.
Demo: No default, as it depends on the testing strategy.
These values are typically set based on the specific requirements of your machine learning task and the characteristics of your dataset.

## Usage

To run the anomaly detection, use the following command:

```bash
python detect_anomalies.py
```

## Demo Data

The repository includes a demo dataset located in the `demo_data/` directory. This dataset contains both normal and anomalous data points for testing and demonstration purposes. You can use this data to see how the autoencoder performs in detecting anomalies.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your changes.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

## Contact

For any questions or suggestions, please open an issue or contact the project maintainer at daniel.a.crawford2021 [a] the google domain :P
