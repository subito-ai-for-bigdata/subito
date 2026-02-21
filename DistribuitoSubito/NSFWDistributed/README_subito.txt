1. Start Ngrok: ngrok start --all
2. Run Kafka: start_kafka_og.bat 
3. Stream data to kafka: python stream_data.py
4. Colab set new ngrok ports
5. Add config.json file to colab
6. Start Prediction pipeline: python prediction_pipeline.py
7. Start Production pipeline: python production_pipeline.py
8. Start Streamlit app: stremlit run streamlit_app.py