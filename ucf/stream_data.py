import os
import math
import time
import json
from kafka import KafkaProducer
import numpy as np
from sklearn.model_selection import train_test_split
import cv2

# Producer part
def error_callback(exc):
  raise Exception('Error while sendig data to kafka: {0}'.format(str(exc)))


def write_to_kafka(partition_name, topic_name, x_tmp, y_tmp):
  """
  Writes data to a Kafka topic partition.

  :param partition_name: The name of the partition ('frames' or other).
  :param topic_name: The Kafka topic to which data will be written.
  :param x_tmp: Iterable containing data to be written (e.g., frames for 'frames' partition).
  :param y_tmp: Iterable containing labels associated with the frames (if applicable).
  :return: None.

  This function sends data to the specified Kafka topic. If `partition_name` is 'frames',
  the data is converted to bytes before transmission. Otherwise, labels are converted to
  strings and sent. The process continues until `tmp_filter_train` is reached or all
  data points are processed. Messages are sent sequentially
  """
  print("Starting to write...")
  producer = KafkaProducer(bootstrap_servers=['127.0.0.1:9092'])
  count = 0
  for (x, y) in zip(x_tmp, y_tmp):
    if partition_name == 'frames':
      # Tranform data to bytes
      msg = x.tobytes()
      producer.send(topic_name, key=bytes(count), value=msg, partition=0).add_errback(error_callback)
      producer.flush()
      count += 1
    else:
      yy = np.uint8(y.item())
      str_msg = yy.astype(str)
      # Tranform data to bytes and send to producer
      producer.send(topic_name, key=bytes(count), value=bytes(str_msg, 'utf-8'), partition=1).add_errback(
        error_callback)
      producer.flush()
      count += 1
    if count >= tmp_filter_train:
      break
  producer.close()
  print("Wrote {0} messages into topic: {1}".format(count, topic_name))


def write_to_kafka_test(partition_name, topic_name, x_tmp, y_tmp):
  """
  Writes data to a Kafka topic partition.

  :param partition_name: The name of the partition ('frames' or other).
  :param topic_name: The Kafka topic to which data will be written.
  :param x_tmp: Iterable containing data to be written (e.g., frames for 'frames' partition).
  :param y_tmp: Iterable containing labels associated with the frames (if applicable).
  :return: None.

  This function sends data to the specified Kafka topic. If `partition_name` is 'frames',
  the data is converted to bytes before transmission. Otherwise, labels are converted to
  strings and sent. The process continues until `tmp_filter_test` is reached or all
  data points are processed. Messages are sent sequentially
  """
  print("Starting to write...")
  producer = KafkaProducer(bootstrap_servers=['127.0.0.1:9092'])
  count = 0
  for (x, y) in zip(x_tmp, y_tmp):
    if partition_name == 'frames':
      # Tranform data to bytes
      msg = x.tobytes()
      producer.send(topic_name, key=bytes(count), value=msg, partition=0).add_errback(error_callback)
      producer.flush()
      count += 1
    else:
      yy = np.uint8(y.item())
      str_msg = yy.astype(str)
      # Tranform data to bytes and send to producer
      producer.send(topic_name, key=bytes(count), value=bytes(str_msg, 'utf-8'), partition=1).add_errback(
        error_callback)
      producer.flush()
      count += 1
    if count >= tmp_filter_test:
      break
  producer.close()
  print("Wrote {0} messages into topic: {1}".format(count, topic_name))


def frames_extraction(video_path):
  """
  Extracts frames from a video at regular intervals.

  :param video_path: Path to the video file.
  :return: List of extracted frames resized to 64x64 pixels.

  This function reads a video file, determines the number of frames, and calculates an interval
  (`skip_frames_window`) to extract frames evenly across the video's duration. The extracted frames
  are resized to 64x64 pixels before being stored in a list and returned.
  """
  # Store the video frames
  frames_list = []

  # Read the video
  video_reader = cv2.VideoCapture(video_path)

  # Get total number of frames (of this video)
  video_frames_count = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))

  # Calculate the interval after which frames will be stored (the step)
  skip_frames_window = max(int(video_frames_count/SEQUENCE_LENGTH), 1)

  # Iterate over video-frames
  for frame_counter in  range(SEQUENCE_LENGTH):
    # Adjust the pointer of current frame
    video_reader.set(cv2.CAP_PROP_POS_FRAMES, frame_counter * skip_frames_window)

    # Read the corresponding frame
    success, frame = video_reader.read()

    if not success:
      break

    # Resize-normalize the frame and save it to the corresponding list
    resized_frame = cv2.resize(frame, (64, 64))
    frames_list.append(resized_frame)
    # normalized_frame = resized_frame / 255
    # frames_list.append(normalized_frame)

  video_reader.release()

  return frames_list


def create_dataset():
  """
  Creates a dataset by extracting frames from videos in predefined classes.

  :return: A tuple containing:
      - features (numpy array): Extracted frames for each video.
      - labels (numpy array): Corresponding class labels.
      - video_files_paths (list): Paths of the processed video files.

  This function iterates over the predefined classes in `CLASSES_LIST`, extracts frames from videos
  in each class folder, and stores the processed frames, labels, and file paths. Only videos with
  the required `SEQUENCE_LENGTH` of frames are included.
  """
  data_dir = "UCF50"
  # Lists that contain the extracted features, the labels and the paths of the videos
  features = []
  labels = []
  video_files_paths = []

  # Iterate through all (selected) classes
  for class_index, class_name in enumerate(CLASSES_LIST):
    print(f'Extracting Data of Class: {class_name}')

    # Get the videos that are contained in each (selected) class
    files_list = os.listdir(os.path.join(data_dir, class_name))
    for file_name in files_list:
      video_file_path = os.path.join(data_dir, class_name, file_name)
      frames = frames_extraction(video_file_path)
      if len(frames) == SEQUENCE_LENGTH:
        features.append(frames)
        labels.append(class_index)
        video_files_paths.append(video_file_path)

  # Convert lists to numpy arrays
  features = np.asarray(features)
  labels = np.asarray(labels)

  return features, labels, video_files_paths


if __name__ == "__main__":
  try:
    with open('config_video.json') as json_file:
      config = json.load(json_file)
  except:
    print("config_video.json not found")
    exit()
    args = sys.argv[1:]
  SEQUENCE_LENGTH = config['sequence_length']
  tmp_filter_train = config['stream_batch_train'] * SEQUENCE_LENGTH
  tmp_filter_test = config['stream_batch_test'] * SEQUENCE_LENGTH
  CLASSES_LIST = config['classes_list']
  features, labels, video_files_paths = create_dataset()
  features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.25, shuffle=True)

  features_train_frames = []
  for i in range(0, len(features_train) - 1):
    for element in features_train[i]:
      features_train_frames.append(element)
  print(len(features_train_frames))
  print(tmp_filter_train)
  labels_train_frames = []
  for i in range(0, len(labels_train) - 1):
    for j in range(0, SEQUENCE_LENGTH):
      labels_train_frames.append(labels_train[i])
  print(len(labels_train_frames))
  print(tmp_filter_train)
  features_test_frames = []
  for i in range(0, len(features_test) - 1):
    for element in features_test[i]:
      features_test_frames.append(element)
  print(len(features_test_frames))
  print(tmp_filter_test)
  labels_test_frames = []
  for i in range(0, len(labels_test) - 1):
    for j in range(0, SEQUENCE_LENGTH):
      labels_test_frames.append(labels_test[i])
  print(len(labels_test_frames))
  print(tmp_filter_test)

  for i in range(math.floor(len(features_train_frames) / tmp_filter_train)):
    # Write to kafka the training images
    write_to_kafka("frames", "train-topic", features_train_frames[tmp_filter_train * i: tmp_filter_train * (i + 1)],
                   labels_train_frames[tmp_filter_train * i: tmp_filter_train * (i + 1)])
    # Write to kafka the labels of the training images
    write_to_kafka("labels", "train-topic", features_train_frames[tmp_filter_train * i: tmp_filter_train * (i + 1)],
                   labels_train_frames[tmp_filter_train * i: tmp_filter_train * (i + 1)])
    # Write to kafka the testing images
    if i == 0:
      write_to_kafka_test("frames", "test-topic", features_test_frames, labels_test_frames)
      # Write to kafka the labels of the testing images
      write_to_kafka_test("labels", "test-topic", features_test_frames, labels_test_frames)
    time.sleep(0)
