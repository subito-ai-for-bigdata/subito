# SuBiTO: Stream-Based Training Optimization for Neural Networks

SuBiTO is an intelligent framework designed to optimize the trade-offs between training time and accuracy in real-time machine learning applications over Big Streaming Data. It addresses the challenges faced by Neural Networks (NNs) deployed in high-speed, high-volume environments by continuously adjusting model parameters to maintain a balance between performance and computational efficiency.

## Features:
- **Automated Parameter Tuning**: Dynamically adjusts the number, size, and type of NN layers based on incoming data streams.
- **Stream Synopses**: Optimizes the size of ingested data via specific stream synopsis parameters for faster processing.
- **Epoch Optimization**: Fine-tunes the number of training epochs to strike the best balance between training time and accuracy.
- **Real-Time Adaptation**: Continuously learns and adapts as new data arrives, suggesting optimal parameter sets for deployment.
- **Concept Drift Detection**: Identifies changes in data distribution over time, allowing human operators to adjust parameters on-the-fly.
- **Scalability**: Designed to scale with the volume and velocity of streaming data, making it ideal for real-time applications.

SuBiTO helps machine learning systems maintain accuracy without sacrificing speed, ensuring seamless real-time predictions while adjusting to evolving data patterns.

## Usage:
- **Real-time Data Streams**: Deploy SuBiTO in applications requiring fast predictions from rapidly arriving data.
- **Dynamic Neural Network Architectures**: Let SuBiTO optimize NN configurations to achieve both high accuracy and low latency.
- **Concept Drift Detection**: Use the built-in concept drift detection to manage data shifts in your system.

## Installation:
```
git clone https://github.com/your-username/SuBiTO.git
cd SuBiTO
pip install -r requirements.txt
```

## Contributing:
Feel free to open issues, suggest improvements, or submit pull requests. Contributions are always welcome!
