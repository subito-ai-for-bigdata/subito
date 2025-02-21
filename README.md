# SuBiTO: Synopsis-Based Training Optimization for Neural Networks

[SuBiTO](https://subito-ai-for-bigdata.github.io/) is an intelligent framework designed to optimize the trade-offs between training time and accuracy in real-time machine learning applications over Big Streaming Data. It tackles the challenges faced by Neural Networks (NNs) deployed in high-speed, high-volume environments by continuously adjusting model parameters to ensure efficiency with minimal downtime.

## Features:
- **Automated Parameter Tuning**: Dynamically adjusts the number, size, and type of NN layers based on incoming data streams.
- **Stream Synopses**: Optimizes the size of ingested data via specific stream synopsis parameters to ensure faster processing and reduce unnecessary computation.
- **Epoch Optimization**: Fine-tunes the number of training epochs to achieve the best balance between training time and accuracy.
- **Bayesian Optimization**: The optimization process is powered by Bayesian Optimization using Gaussian Process Regression (GPR) models and acquisition functions, enabling efficient exploration vs exploitation of the parameter space for optimal configurations.
- **Real-Time Adaptation**: Continuously learns and adapts as new data arrives, suggesting optimal parameter sets for deployment.

SuBiTO is designed to help machine learning systems achieve high accuracy while optimizing speed, with minimal downtime, maintaining a balance between training efficiency and model performance. 

## Subito Dashboard:
The SuBiTO dashboard, developed with Streamlit 2024, provides an intuitive interface for human operators (e.g., content moderators, community managers) to manage system parameters. Operators can set valid ranges for parameters such as synopses compression ratio, possible epochs, and types of NN layers. After executing the SuBiTO Optimizer, the dashboard displays Pareto Optimal solutions, showcasing the top-3 NN architectures with their expected training loss and accuracy. Operators can deploy any of these alternatives directly, with the selected model being dynamically deployed in the Training Pipeline at runtime. Additionally, a live bar chart shows the statistics of the dynamically updated Prediction Pipeline.

## Usage:
- **Real-time Data Streams**: Deploy SuBiTO in applications requiring fast predictions from rapidly arriving data.
- **Dynamic Neural Network Architectures**: Let SuBiTO optimize NN configurations to achieve both high accuracy and low latency.
- **Optimization with Bayesian Techniques**: Leverage the power of Bayesian Optimization to find the best set of parameters with minimal computational overhead.

## Scenarios:
- **NSFW Image Classification Dataset:**
In this scenario, SuBiTO is used to classify images from an NSFW (Not Safe For Work) dataset. The dataset is divided into 5 categories. SuBiTO optimizes the training of a neural network to classify these images based on the features of the visual content. By dynamically adjusting the model’s parameters in real-time, SuBiTO ensures high classification accuracy while maintaining minimal downtime and efficient training.

- **UCF10 Video Classification Dataset:**
The UCF10 dataset, a subset of the UCF50 video dataset, consists of 10 distinct video action classes. In this use case, SuBiTO can optimize the training of a neural network model for video classification. It adjusts parameters to capture the spatio-temporal features of the videos, improving both model performance and training time.

**SuBiTO's adaptability and optimization techniques make it an excellent choice for real-time machine learning applications in both image and video domains.**

## Installation:
```
git clone https://github.com/your-username/SuBiTO.git
cd SuBiTO
pip install -r requirements.txt
```
## Execution:
1. Start Ngrok: ngrok start --all
2. Run Kafka
3. Stream data to kafka: python stream_data.py
4. Colab set new ngrok ports
5. Start Prediction pipeline: python prediction_pipeline.py
6. Start Production pipeline: python production_pipeline.py
7. Start SuBiTO Optimizer through the .ipynb notebook
8. Start Streamlit app: stremlit run streamlit_app.py
   
> Each pipeline (Prediction, Production, SuBiTO Optimizer) can and should be executed on different machines.

## Publication

**SuBiTO: Synopsis-based Training Optimization for Continuous Real-Time Neural Learning over Big Streaming Data**.
Errikos Streviniotis, George Klioumis, Nikos Giatrakos.
*In Proceedings of the 39th Annual AAAI Conference on Artificial Intelligence (AAAI'25)*, Philadelphia, Pennsylvania, USA, March 2025.

If you use this work, please cite it as follows:

```bibtex
@inproceedings{SuBiTO,
  author    = {Errikos Streviniotis and George Klioumis and Nikos Giatrakos},
  title     = {SuBiTO: Synopsis-based Training Optimization for Continuous Real-Time Neural Learning over Big Streaming Data (Demo Paper)},
  booktitle = {Proceedings of the 39th Annual AAAI Conference on Artificial Intelligence (AAAI'25)},
  year      = {2025},
  address   = {Philadelphia, Pennsylvania, USA},
  month     = {March},
}
```

## Contributing:
Feel free to open issues, suggest improvements, or submit pull requests. Contributions are always welcome!
> More info, videos and presentations can be found on the official website of [SuBiTO](https://subito-ai-for-bigdata.github.io/).
