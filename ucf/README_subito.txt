1. Start Ngrok: ngrok start --all
2. Run Kafka: start_kafka_og.bat 
3. Stream data to kafka: python stream_data.py
4. Colab set new ngrok ports
5. Add config.json file to colab
6. Start Prediction pipeline: python prediction_pipeline.py
7. Start Production pipeline: python production_pipeline.py
8. Start Streamlit app: stremlit run streamlit_app.py


ATTENTION:
- Streamlit does not refresh (show results) when subito finishes
--> Check that ngrok ports are figured correctly

- Running in the GPU is stuck on Production or Prediction
-->Delete the inner of tmp folder in C drive (its kafka related only)
-->Delete the inner of C/kafka/logs folder as well
-->Rerun start_kafka_og
--If it fails
-->Redelete everything as above and restart PC
--If it succeeds rerun stream_data.py
--Set ngrok ports correctly on Colab (they have changed)