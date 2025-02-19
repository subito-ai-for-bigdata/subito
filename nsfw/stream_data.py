import json
import socket
import os
import sys
import random
import math
from datetime import datetime
import time
import threading
import json
from kafka import KafkaProducer, KafkaConsumer, TopicPartition
from kafka.errors import KafkaError
import pandas as pd
import tensorflow as tf
import tensorflow_io as tfio
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.datasets import fashion_mnist, mnist, cifar10, cifar100
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
import skopt
from skopt import gbrt_minimize, gp_minimize
from skopt.utils import use_named_args
from skopt.space import Real, Categorical, Integer
from tensorflow.python.keras import backend as K
import matplotlib.pyplot as plt
import pickle
import ast
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
    if partition_name == 'images':
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
    if partition_name == 'images':
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


def load_dataset(train_dir, test_dir):
  """
  Loads image dataset from specified train and test directories, assigning numerical labels to each class.

  :param train_dir: Path to the training directory containing subfolders for each class.
  :param test_dir: Path to the testing directory containing subfolders for each class.
  :return: A tuple containing ((x_train, y_train), (x_test, y_test)) and class_labels.
           - x_train, x_test: NumPy arrays of image data (normalized to [0, 1]).
           - y_train, y_test: NumPy arrays of corresponding numerical labels.
           - class_labels: List of class names corresponding to the labels.

  This function reads images from the specified directories, converts them to RGB format,
  assigns unique numerical labels to each class, and returns the processed dataset.
  """
  def load_data_and_labels(data_dir):
    x_data = []
    y_data = []
    labels = []
    classes = sorted(os.listdir(data_dir))  # Ensure consistent order of labels
    # Assign incremental labels (0, 1, ..., n-1)
    for label, class_name in enumerate(classes):
      class_dir = os.path.join(data_dir, class_name)
      if not os.path.isdir(class_dir):
        continue
      labels.append(class_name)
      # Process each image in the class directory
      for filename in os.listdir(class_dir):
        if filename.lower().endswith(".jpg"):
          img_path = os.path.join(class_dir, filename)
          image = cv2.imread(img_path)
          image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
          if image is None:
            print(f"Could not read image: {img_path}")
            continue
          x_data.append(image)
          y_data.append(label)
    return np.array(x_data), np.array(y_data), labels
  x_train, y_train, train_labels = load_data_and_labels(train_dir)
  x_test, y_test, test_labels = load_data_and_labels(test_dir)
  # Ensure labels match for train and test
  assert train_labels == test_labels, "Mismatch in class labels between train and test sets"

  return (x_train, y_train), (x_test, y_test)


def shuffle_dataset(x, y):
  """
  Randomly shuffles the dataset while maintaining the correspondence between features and labels.

  :param x: numpy array of feature data.
  :param y: numpy array of labels corresponding to the feature data.
  :return: A tuple containing (shuffled_x, shuffled_y).

  This function ensures that both the feature data and labels are shuffled
  in the same order, preserving the mapping between them.
  """
  indices = np.random.permutation(len(x))
  return x[indices], y[indices]


if __name__ == "__main__":
  try:
    with open('config.json') as json_file:
      config = json.load(json_file)
  except:
    print("config.json not found")
    exit()
    args = sys.argv[1:]
  tmp_filter_train = config['stream_batch_train']
  tmp_filter_test = config['stream_batch_test']
  if config['dataset'] == 'cifar':
    dataset = cifar10.load_data()  #@param
    # Load mnist (or fashion_mnist) dataset
    (x_train, y_train), (x_test, y_test) = dataset
    # Scale images to the [0, 1] range
    x_train = x_train / 255
    x_test = x_test / 255
  elif config['dataset'] == 'nsfw':
    train_path = "nsfw_dataset/train"  # Replace with your train folder path
    test_path = "nsfw_dataset/test"   # Replace with your test folder path
    (x_train, y_train), (x_test, y_test) = load_dataset(train_path, test_path)
    # Shuffle the training and testing datasets
    x_train, y_train = shuffle_dataset(x_train, y_train)
    x_test, y_test = shuffle_dataset(x_test, y_test)
    print(f"Training data: {x_train.shape}, {y_train.shape}")
    print(f"Testing data: {x_test.shape}, {y_test.shape}")
  if len(x_train.shape) == 3:
    # Make sure images have shape (28, 28, 1)
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    print("x_train shape:", x_train.shape)
    print(x_train.shape[0], "train samples")
    print(x_test.shape[0], "test samples")

  for i in range(math.floor(len(x_train) / tmp_filter_train)):
    # Write to kafka the training images
    write_to_kafka("images", "train-topic", x_train[tmp_filter_train * i: tmp_filter_train * (i + 1)],
                   y_train[tmp_filter_train * i: tmp_filter_train * (i + 1)])
    # Write to kafka the labels of the training images
    write_to_kafka("labels", "train-topic", x_train[tmp_filter_train * i: tmp_filter_train * (i + 1)],
                   y_train[tmp_filter_train * i: tmp_filter_train * (i + 1)])
    # Write to kafka the testing images
    if i == 0:
      write_to_kafka_test("images", "test-topic", x_test, y_test)
      # Write to kafka the labels of the testing images
      write_to_kafka_test("labels", "test-topic", x_test, y_test)
    time.sleep(0)
