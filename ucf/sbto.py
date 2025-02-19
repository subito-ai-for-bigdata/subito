import gc
import json
import multiprocessing
import pickle
import socket
import sys
import random
import math
from datetime import datetime
import time
import json
from multiprocessing import Queue

from kafka import KafkaProducer, KafkaConsumer, TopicPartition
from kafka.errors import KafkaError
import pandas as pd
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.datasets import fashion_mnist, mnist, cifar10, cifar100
from tensorflow.keras import layers, models
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam
from skopt import gbrt_minimize, gp_minimize
from skopt.utils import use_named_args
from skopt.space import Real, Categorical, Integer
from tensorflow.python.keras import backend as K
import matplotlib.pyplot as plt

try:
    with open('config_video.json') as json_file:
        config = json.load(json_file)
except:
    print("config.json not found")
    exit()

call_counter = 0
SEQUENCE_LENGTH = config['sequence_length']
tmp_filter_train = config['bo_data_train'] * SEQUENCE_LENGTH
tmp_filter_test = config['bo_data_test'] * SEQUENCE_LENGTH
size_of_batch = config['size_of_batch']
lr = config['lr']
# size_of_batch_low = config['size_of_batch_low']
# size_of_batch_high = config['size_of_batch_high']
# lr_low = config['lr_low']
# lr_high = config['lr_high']
num_of_conv_layers_low = config['num_of_conv_layers_low']
num_of_conv_layers_high = config['num_of_conv_layers_high']
num_of_pool_layers_low = config['num_of_pool_layers_low']
num_of_pool_layers_high = config['num_of_pool_layers_high']
num_of_dense_layers_low = config['num_of_dense_layers_low']
num_of_dense_layers_high = config['num_of_dense_layers_high']
num_of_lstm_layers_low = config['num_of_lstm_layers_low']
num_of_lstm_layers_high = config['num_of_lstm_layers_high']
num_of_gru_layers_low = config['num_of_gru_layers_low']
num_of_gru_layers_high = config['num_of_gru_layers_high']
num_of_rnn_layers_low = config['num_of_rnn_layers_low']
num_of_rnn_layers_high = config['num_of_rnn_layers_high']
sample_size_low = config['sample_size_low']
sample_size_high = config['sample_size_high']
num_of_epochs_low = config['num_of_epochs_low']
num_of_epochs_high = config['num_of_epochs_high']
sampling_method_id = 1
acquisition_f = config['acquisition_f']
theta_parameter = config['theta_parameter']
lamda_acc = config['lamda_acc']
bo_call_number = config['bo_call_number']
DATASET_SHAPE = [64, 64, 3]
UNIQUE_CLASS_LABELS = range(len(config['classes_list']))
# Percentage of the dataset that will be used for testing
perc_test = 1 #@param {type:"number"}


extra_results = []


dim_sample_size = Real(low=sample_size_low, high=sample_size_high, name='sample_size')
dim_epochs_number = Integer(low=num_of_epochs_low, high=num_of_epochs_high, name='epochs_number')
dim_conv_number = Integer(low=num_of_conv_layers_low, high=num_of_conv_layers_high, name='conv_number')
dim_pool_number = Integer(low=num_of_pool_layers_low, high=num_of_pool_layers_high, name='pool_number')
dim_lstm_number = Integer(low=num_of_lstm_layers_low, high=num_of_lstm_layers_high, name='lstm_number')
dim_gru_number = Integer(low=num_of_gru_layers_low, high=num_of_gru_layers_high, name='gru_number')
dim_rnn_number = Integer(low=num_of_rnn_layers_low, high=num_of_rnn_layers_high, name='rnn_number')
dim_dense_number = Integer(low=num_of_dense_layers_low, high=num_of_dense_layers_high, name='dense_number')
# dim_lr = Real(low=lr_low, high=lr_high, name='learning_rate')
# dim_batch_size = Integer(low=size_of_batch_low, high=size_of_batch_high, name='batch_size')

dimensions = [dim_sample_size,
              dim_epochs_number,
              dim_conv_number,
              dim_pool_number,
              dim_lstm_number,
              dim_gru_number,
              dim_rnn_number,
              dim_dense_number
             ]
default_parameters = [0.99, 4, 2, 2, 1, 0, 0, 1]
# default_parameters = [(sample_size_low + sample_size_high) / 2,
#                      (num_of_epochs_low + num_of_epochs_high) // 2,
#                      (num_of_conv_layers_low + num_of_conv_layers_high) // 2,
#                      (num_of_pool_layers_low + num_of_pool_layers_high) // 2,
#                      (num_of_lstm_layers_low + num_of_lstm_layers_high) // 2,
#                      (num_of_gru_layers_low + num_of_gru_layers_high) // 2,
#                      (num_of_rnn_layers_low + num_of_rnn_layers_high) // 2,
#                      (num_of_dense_layers_low + num_of_dense_layers_high) // 2]

#@markdown ##Other Configuration(s)
CONV_PADDING = 'same'
MAX_POOL_PADDING = 'same'
CONV_NEURONS_CONST = 32
CONV_NEURONS_BOUND = 256
DENSE_NEURONS_CONST = 128
DENSE_NEURONS_BOUND = 32
UNITS_CONST = 256
UNITS_BOUND = 64
streamlit_live_socket = None

def start_controller():
    """
    Initialize the socket and listen for keyboard input
    """
    # Open config file and get the desired port for socket communication
    try:
        with open('config.json') as json_file:
            config = json.load(json_file)
    except:
        print("config.json not found")
        exit()
    # Initialize an IPv4 socket with TCP (default) and try to connect to the nn
    streamlit_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    streamlit_socket.connect((config["host_address"], config['streamlit_port']))
    streamlit_live_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    streamlit_live_socket.connect((config["host_address"], config['sbto_live_port']))
    return streamlit_socket, streamlit_live_socket, config

def start_bo(config, streamlit_socket, unique_class_labels, received_images_reshaped, received_labels_decoded, received_images_reshaped_test, received_labels_decoded_test, dataset_shape):
    # Get unique labels in our (received) training dataset
    unique_class_labels = np.unique(received_labels_decoded)

    for i in range(1):
        # Get the new number of epochs from the keyboard
        data, my_stats = bo_res(unique_class_labels, received_images_reshaped, received_labels_decoded, received_images_reshaped_test, received_labels_decoded_test, dataset_shape)
        serialized_df = pickle.dumps(my_stats)
        streamlit_socket.sendall(serialized_df)
    # # Close socket on exit

    #TODO: Might need to be commented
    streamlit_socket.close()
    streamlit_live_socket.close()

    print("Sockets Closed")

@use_named_args(dimensions = dimensions)
def fitness(sample_size, epochs_number, conv_number, pool_number, dense_number, lstm_number, gru_number, rnn_number):
    print()
    print(f"EPOCHS to: {epochs_number} \n SAMPLE RATE to: {sample_size} \n NUM_OF_CONV_LAYERS to: {conv_number} \n NUM_OF_POOL_LAYERS to: {pool_number} \n NUM_OF_DENSE_LAYERS to: {dense_number} \n NUM_OF_LSTM_LAYERS to: {lstm_number} \n NUM_OF_GRU_LAYERS to: {gru_number} \n NUM_OF_RNN_LAYERS to: {rnn_number}")
    print()
    global call_counter
    call_counter += 1
    input_str = f"CALL: {call_counter}/{bo_call_number} \n EPOCHS to: {epochs_number} \n SAMPLE RATE to: {sample_size} \n NUM_OF_CONV_LAYERS to: {conv_number} \n NUM_OF_POOL_LAYERS to: {pool_number} \n NUM_OF_DENSE_LAYERS to: {dense_number}\n NUM_OF_LSTM_LAYERS to: {lstm_number} \n NUM_OF_GRU_LAYERS to: {gru_number} \n NUM_OF_RNN_LAYERS to: {rnn_number}\n"
    # Call the sampling method
    x_train, y_train = sampling_method(sampling_method_id, received_images_reshaped, received_labels_decoded, sample_size)

    layers_lst = []
    if conv_number > 0:
      if conv_number > pool_number:
        for i in range(0, conv_number - pool_number):
          layers_lst.append('conv')
        for i in range(0, pool_number):
          layers_lst.append('conv')
          layers_lst.append('pool')
      elif conv_number == pool_number:
        for i in range(0, conv_number):
          layers_lst.append('conv')
          layers_lst.append('pool')
      else:
        for i in range(0, conv_number):
          layers_lst.append('conv')
          layers_lst.append('pool')
        for i in range(conv_number, pool_number):
          layers_lst.append('pool')
    if lstm_number > 0:
      for i in range(0, lstm_number):
        layers_lst.append('lstm')
    if gru_number > 0:
      for i in range(0, gru_number):
        layers_lst.append('gru')
    if rnn_number > 0:
      for i in range(0, rnn_number):
        layers_lst.append('rnn')
    if dense_number > 0:
      for i in range(0, dense_number):
        layers_lst.append('dense')
    print(f"---------------------------->{layers_lst}")
    q = Queue()
    process_eval = multiprocessing.Process(target=my_evaluate, args=(q, x_train, y_train, received_images_reshaped_test, received_labels_decoded_test, layers_lst, epochs_number, lr, size_of_batch, CONV_NEURONS_CONST, UNITS_CONST, DENSE_NEURONS_CONST, CONV_NEURONS_BOUND, UNITS_BOUND, DENSE_NEURONS_BOUND, input_str))
    process_eval.start()
    test_acc, tr_time, train_loss, train_acc, tradeOff_metric = q.get()
    process_eval.join()


    # Print the results.
    print()
    print("Accuracy (on the testing dataset): {0:.2%}".format(test_acc))
    print(f"Training time: ", tr_time)
    print(tradeOff_metric)
    print()
    # Store the accuracy and the training speed of the corresponding model in order to be printed in the final cell
    tmp = [test_acc, tr_time, train_loss, train_acc]
    extra_results.append(tmp)
    # Delete the Keras model with these hyper-parameters from memory.
    K.clear_session()
    gc.collect()
    del x_train
    del y_train
    # Clear the Keras session, otherwise it will keep adding new
    # models to the same TensorFlow graph each time we create
    # a model with a different set of hyper-parameters.
    tf.compat.v1.reset_default_graph()
    return -tradeOff_metric


def bo_res(unique_class_labels, received_images_reshaped, received_labels_decoded, received_images_reshaped_test, received_labels_decoded_test, dataset_shape):

    gp_result = gp_minimize(func=fitness,
                            dimensions = dimensions,
                            n_calls = bo_call_number,
                            acq_func = acquisition_f,
                            noise = "gaussian",
                            n_jobs = -1,
                            x0 = default_parameters)

    df_extra = pd.DataFrame(extra_results, columns=["Accuracy", "Training Speed (sec)", "Loss Epoch", "Acc Epoch"])

    pd_tmp = pd.concat([pd.DataFrame(gp_result.x_iters, columns=["Sample Size", "Epochs", "Conv", "Pool", "Dense", "LSTM", "GRU", "RNN"]), (pd.Series(gp_result.func_vals * -1, name="Score"))], axis=1)
    final_result = pd.concat([pd_tmp, df_extra], axis=1)

    # print(gp_result.x)
    print(f" NEW EPOCHS to: {gp_result.x[1]} \n NEW SAMPLE RATE to: {gp_result.x[0]} \n NUM_OF_CONV_LAYERS to: {gp_result.x[2]} \n NUM_OF_POOL_LAYERS to: {gp_result.x[3]} \n NUM_OF_DENSE_LAYERS to: {gp_result.x[4]} \n NUM_OF_LSTM_LAYERS to: {gp_result.x[5]} \n NUM_OF_GRU_LAYERS to: {gp_result.x[6]} \n NUM_OF_RNN_LAYERS to: {gp_result.x[7]}")
    
    tmp = gp_result.x[1].__str__() +','+ gp_result.x[0].__str__()+','+gp_result.x[2].__str__() +','+ gp_result.x[3].__str__()+','+ gp_result.x[4].__str__()+','+ gp_result.x[5].__str__()+','+ gp_result.x[6].__str__()+','+ gp_result.x[7].__str__()

    return tmp, final_result


# A function that prints the occurence of each class in a list
def print_times_per_label(lst, labels_all):
  # Get unique labels in our training dataset
  unique_labels = np.unique(labels_all)
  for i in range(0, len(unique_labels)):
    print("Class", unique_labels[i], "has", lst.count(i), "samples in our dataset...")

# Select k items from a stream of items-data

# A function to randomly select k items from stream[0..n-1].
def reservoir_sampling(stream, n, k):
  i = 0     # index for elements in stream[]

  # reservoir[] is the output array.
  # Initialize it with first k elements from stream[]
  reservoir = [0] * k

  for i in range(k):
    reservoir[i] = stream[i]

  # Iterate from the (k+1)th element to Nth element
  while(i < n):
    # Pick a random index from 0 to i.
    j = random.randrange(i+1)

    # If the randomly picked
    # index is smaller than k,
    # then replace the element
    # present at the index
    # with new element from stream
    if(j < k):
      reservoir[j] = stream[i]
    i+=1

  return reservoir

# A function that finds the size of each reservoir for every class depending on its occurence in the initial dataset
# and returns the unique labels that exist in our dataset along with the corresponding percentage
def reservoir_size_per_class(init_labels):

  # Get unique labels and their counts (how many times they appear) in our training dataset
  unique_labels, counts = np.unique(init_labels, return_counts = True)

  # Transform to list
  unique_labels_lst = unique_labels.tolist()
  counts_lst = counts.tolist()

  perc_per_class = []
  for i in range(len(unique_labels_lst)):
    perc_per_class.append(counts_lst[i]/len(init_labels))

  # print(perc_per_class)

  return perc_per_class, unique_labels_lst

def sampling_method(sampling_method_id, received_images_reshaped, received_labels_decoded, sample_size):
  # print("Percentage of filtering in our training dataset was set:")
  # print(sample_size)

  if sampling_method_id == 0:
    # Simple reservoir sampling over the whole training dataset
    # Total size of the stream (or training dataset)
    n_train = len(received_images_reshaped)

    # Number of samples that will be drawn
    k_train = int(n_train * sample_size)

    # Use the indexes of dataset in order to decide which samples will be drawn
    idx_tmp_train_list = list(range(0, n_train))

    # Find the indexes in order to construct the dataset that will be used during the training process
    idx_train = reservoir_sampling(idx_tmp_train_list, n_train, k_train)
  else:
    # Reservoir sampling in each class based on the number of samples (per class) that exist in the initial dataset
    # Find the size of each reservoir for every class depending on its occurence in the initial training dataset
    class_perc, unique_ids = reservoir_size_per_class(received_labels_decoded)
    
    # Stores the indexes (from all classes) in order to construct the dataset that will be used during the training process
    idx_train = []

    # Run for every single class the reservoir sampling seperately
    for i in range(0, len(unique_ids)):
      # Find the locations of each sample belonging to our class of interest
      tmp = np.where(np.asarray(received_labels_decoded) == unique_ids[i])
      idx_of_class = tmp[0].tolist()

      # Run the reservoir sampling for the class of interest
      sampled_idx_of_class = reservoir_sampling(idx_of_class, len(idx_of_class), int(len(received_images_reshaped) * sample_size * class_perc[i]))

      # Store the (sampled) samples from this class
      for j in range(0, len(sampled_idx_of_class)):
        idx_train.append(sampled_idx_of_class[j])

  # Store the corresponding images and labels from training dataset based on the sampled indexes
  train_images_lst = []
  for i in idx_train:
    train_images_lst.append(received_images_reshaped[i])

  train_labels_lst = []
  for i in idx_train:
    train_labels_lst.append(received_labels_decoded[i])

  # Check the occurence of each class in the final training dataset
  # print_times_per_label(train_labels_lst, received_labels_decoded)

  # Tranfsorm the lists that we stored our samples into arrays
  train_images = np.asarray(train_images_lst)
  train_labels = np.asarray(train_labels_lst)

  # Verify that the desired filtering was performed in both datasets
  # print("Training dataset before sampling:")
  # print(len(received_images_reshaped))
  # print(len(received_labels_decoded))
  # print("Training dataset after sampling:")
  # print(train_images.shape)
  # print(train_labels.shape)

  return train_images, train_labels


def my_evaluate(q, x_train, y_train, features_test, labels_test, layers_lst, epochs_number, learning_rate, batch_size, CONV_NEURONS_CONST, UNITS_CONST, DENSE_NEURONS_CONST, CONV_NEURONS_BOUND, UNITS_BOUND, DENSE_NEURONS_BOUND, input_str):
  error_flag = -1

  try:
    # Function that creates the model
    model, *_ = create_model(layers_lst, 'dense', CONV_NEURONS_CONST, UNITS_CONST, DENSE_NEURONS_CONST, CONV_NEURONS_BOUND, UNITS_BOUND, DENSE_NEURONS_BOUND)

    if model == -1:
      return -1000000

    # If the just-added-layer was conv or pool then add manually a flatten layer
    if 'lstm' not in layers_lst and 'gru' not in layers_lst and 'rnn' not in layers_lst and 'dense' not in layers_lst:
      model.add(Flatten())

    # Softmax is an activation function that is used mainly for classification tasks
    # It normalizes the input vector into a probability distribution  that is proportional to the exponential of the input numbers.
    model.add(tf.keras.layers.Dense(len(UNIQUE_CLASS_LABELS), activation = "softmax"))
  except ValueError:
    print("No valid input...:(")
    error_flag = 1

  if error_flag == -1:
    model.compile(optimizer=Adam(learning_rate),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

    stringlist = []
    model.summary(print_fn=lambda x: stringlist.append(x))
    short_model_summary = "\n".join(stringlist)

    # Here we reshape the input of the network based on the type of the first layer of the network
    # If the first layer is conv
    if layers_lst[0] == 'conv':
      reshaped_x_train = x_train
      reshaped_x_test = features_test
    # If the first layer is lstm-gru-rnn
    elif layers_lst[0] == 'lstm' or layers_lst[0] == 'gru' or layers_lst[0] == 'rnn':
      num_samples, num_frames, height, width, channels = x_train.shape
      reshaped_x_train = x_train.reshape(num_samples, num_frames, height * width * channels)
      num_samples, num_frames, height, width, channels = features_test.shape
      reshaped_x_test = features_test.reshape(num_samples, num_frames, height * width * channels)
    # If the first layer is dense
    else:
      num_samples, num_frames, height, width, channels = x_train.shape
      reshaped_x_train = x_train.reshape(num_samples, num_frames * height * width * channels)
      num_samples, num_frames, height, width, channels = features_test.shape
      reshaped_x_test = features_test.reshape(num_samples, num_frames * height * width * channels)

    start = time.time()
    np.int = int
    blackbox = model.fit(x=reshaped_x_train,
                        y=y_train,
                        epochs=epochs_number,
                        batch_size=batch_size
                        )
    stop = time.time()
    tr_loss_lst = blackbox.history['loss']
    tr_accuracy_lst = blackbox.history['accuracy']
    # Compute the training speed of this CNN architecture
    tr_time = stop - start

    # Compute the accuracy of our training model in the testing dataset
    test_loss, test_acc = model.evaluate(reshaped_x_test,  labels_test, verbose=2)

    # # Return the validation accuracy for the last epoch.
    # accuracy = blackbox.history['val_accuracy'][-1]

    # Compute the metric that captures the accuracy--speed tradeoff
    tradeOff_metric = lamda_acc * test_acc - (1 - lamda_acc) * math.tanh(tr_time/theta_parameter - 1)

    # Print the results.
    print()
    print("Accuracy (on the testing dataset): {0:.2%}".format(test_acc))
    print(f"Training time: ", tr_time)
    print(tradeOff_metric)
    print()

    tmp = "\nAccuracy (on the testing dataset): {0:.2%}".format(
        test_acc) + '\nTraining time:' + tr_time.__str__() + '\nTradeOff Metric:' + tradeOff_metric.__str__() + '\n\n'

    streamlit_live_socket.sendall(pickle.dumps(input_str + short_model_summary + tmp))

    # Delete the Keras model with these hyper-parameters from memory.
    del model
    q.put([test_acc, tr_time, tr_loss_lst, tr_accuracy_lst, tradeOff_metric])
  else:
    q.put([0, 1000000000, 1000000000, 0, 0])


def create_model(layers_lst, layer2add, CONV_NEURONS_CONST, UNITS_CONST, DENSE_NEURONS_CONST, CONV_NEURONS_BOUND,
                 UNITS_BOUND, DENSE_NEURONS_BOUND):

    if layers_lst[0] == 'pool' or len(layers_lst) == 0:
        return -1

    # Initialize a sequential model
    model = tf.keras.models.Sequential()

    # Define the number of neurons for conv and dense layers and the number of units for lstm-gru-rnn
    conv_tmp2 = CONV_NEURONS_CONST
    units_tmp2 = UNITS_CONST
    dense_tmp2 = DENSE_NEURONS_CONST

    # Find the type of the next and the previous layer because you need different configurations
    for count, layer in enumerate(layers_lst):
        if count == 0 and len(layers_lst) > 1:
            previous_layer_tmp = 'no'
            next_layer_tmp = layers_lst[count + 1]
        elif count == 0:
            previous_layer_tmp = 'no'
            next_layer_tmp = 'no'
        elif count == len(layers_lst) - 1:
            next_layer_tmp = 'no'
            previous_layer_tmp = layers_lst[count - 1]
        else:
            previous_layer_tmp = layers_lst[count - 1]
            next_layer_tmp = layers_lst[count + 1]

        # Recreate the so-far-model
        # First layer conv
        if layer == 'conv' and count == 0:
            model.add(TimeDistributed(Conv2D(int(conv_tmp2), (3, 3), padding='same', activation='relu'),
                                      input_shape=(SEQUENCE_LENGTH, DATASET_SHAPE[0], DATASET_SHAPE[1], 3)))
            conv_tmp2 = conv_tmp2 * 2
        # First layer lstm-gru-rnn (change the shape of the input) and next or 2-be-added layer lstm-gru-rnn (should add the 'return conf')
        elif ((layer == 'lstm' or layer == 'gru' or layer == 'rnn') and (((count == 0) and len(
            layers_lst) == 1 and (layer2add == 'lstm' or layer2add == 'gru' or layer2add == 'rnn')) or (
                                                                             (count == 0) and (
                                                                             next_layer_tmp == 'lstm' or next_layer_tmp == 'gru' or next_layer_tmp == 'rnn')))):
            if layer == 'lstm':
                model.add(tf.keras.layers.LSTM(int(units_tmp2), return_sequences=True,
                                               input_shape=(SEQUENCE_LENGTH, DATASET_SHAPE[0] * DATASET_SHAPE[1] * 3)))
            elif layer == 'gru':
                model.add(tf.keras.layers.GRU(int(units_tmp2), return_sequences=True,
                                              input_shape=(SEQUENCE_LENGTH, DATASET_SHAPE[0] * DATASET_SHAPE[1] * 3)))
            else:
                model.add(tf.keras.layers.SimpleRNN(int(units_tmp2), return_sequences=True,
                                                    input_shape=(SEQUENCE_LENGTH, DATASET_SHAPE[0] * DATASET_SHAPE[1] * 3)))
            units_tmp2 = units_tmp2 / 2
        # First layer lstm-gru-rnn (change the shape of the input)
        elif ((layer == 'lstm' or layer == 'gru' or layer == 'rnn') and count == 0):
            if layer == 'lstm':
                model.add(tf.keras.layers.LSTM(int(units_tmp2),
                                               input_shape=(SEQUENCE_LENGTH, DATASET_SHAPE[0] * DATASET_SHAPE[1] * 3)))
            elif layer == 'gru':
                model.add(tf.keras.layers.GRU(int(units_tmp2),
                                              input_shape=(SEQUENCE_LENGTH, DATASET_SHAPE[0] * DATASET_SHAPE[1] * 3)))
            else:
                model.add(tf.keras.layers.SimpleRNN(int(units_tmp2),
                                                    input_shape=(SEQUENCE_LENGTH, DATASET_SHAPE[0] * DATASET_SHAPE[1] * 3)))
            units_tmp2 = units_tmp2 / 2
        # First layer densse (change the shape of the input)
        elif layer == 'dense' and count == 0:
            model.add(tf.keras.layers.Dense(int(dense_tmp2), activation='relu',
                                            input_shape=(SEQUENCE_LENGTH * DATASET_SHAPE[0] * DATASET_SHAPE[1] * 3,)))
            dense_tmp2 = dense_tmp2 / 2
        # For the remaining layers
        else:
            if layer == 'conv':
                # Add a conv layer by doubling its neurons if they do not violate our user-defined bound
                if conv_tmp2 <= CONV_NEURONS_BOUND:
                    model.add(TimeDistributed(Conv2D(int(conv_tmp2), (3, 3), padding='same', activation='relu')))
                    conv_tmp2 = conv_tmp2 * 2
                else:
                    model.add(
                        TimeDistributed(Conv2D(int(CONV_NEURONS_BOUND), (3, 3), padding='same', activation='relu')))
                    conv_tmp2 = CONV_NEURONS_BOUND
            elif layer == 'pool':
                # Add a pool layer
                model.add(TimeDistributed(MaxPooling2D((4, 4))))
            elif layer == 'lstm':
                # If the previous layer is conv or pool add a flatten layer first
                if previous_layer_tmp == 'conv' or previous_layer_tmp == 'pool':
                    model.add(TimeDistributed(Flatten()))
                # Add a lstm layer by reducing (* 0.5) its units if they do not violate our user-defined bound
                if units_tmp2 >= UNITS_BOUND:
                    # If the next layer is dense then do not return sequences
                    if next_layer_tmp == 'dense' or (layer2add == 'dense' and count == len(layers_lst) - 1):
                        model.add(tf.keras.layers.LSTM(int(units_tmp2)))
                    else:
                        model.add(tf.keras.layers.LSTM(int(units_tmp2), return_sequences=True))
                    units_tmp2 = units_tmp2 / 2
                else:
                    # If the next layer is dense then do not return sequences
                    if next_layer_tmp == 'dense' or (layer2add == 'dense' and count == len(layers_lst) - 1):
                        model.add(tf.keras.layers.LSTM(int(UNITS_BOUND)))
                    else:
                        model.add(tf.keras.layers.LSTM(int(UNITS_BOUND), return_sequences=True))
                    units_tmp2 = UNITS_BOUND
            elif layer == 'gru':
                # If the previous layer is conv or pool add a flatten layer first
                if previous_layer_tmp == 'conv' or previous_layer_tmp == 'pool':
                    model.add(TimeDistributed(Flatten()))
                # Add a gru layer by reducing (* 0.5) its units if they do not violate our user-defined bound
                if units_tmp2 >= UNITS_BOUND:
                    # If the next layer is dense then do not return sequences
                    if next_layer_tmp == 'dense' or (layer2add == 'dense' and count == len(layers_lst) - 1):
                        model.add(tf.keras.layers.GRU(int(units_tmp2)))
                    else:
                        model.add(tf.keras.layers.GRU(int(units_tmp2), return_sequences=True))
                    units_tmp2 = units_tmp2 / 2
                else:
                    # If the next layer is dense then do not return sequences
                    if next_layer_tmp == 'dense' or (layer2add == 'dense' and count == len(layers_lst) - 1):
                        model.add(tf.keras.layers.GRU(int(UNITS_BOUND)))
                    else:
                        model.add(tf.keras.layers.GRU(int(UNITS_BOUND), return_sequences=True))
                    units_tmp2 = UNITS_BOUND
            elif layer == 'rnn':
                # If the previous layer is conv or pool add a flatten layer first
                if previous_layer_tmp == 'conv' or previous_layer_tmp == 'pool':
                    model.add(TimeDistributed(Flatten()))
                # Add a rnn layer by reducing (* 0.5) its units if they do not violate our user-defined bound
                if units_tmp2 >= UNITS_BOUND:
                    # If the next layer is dense then do not return sequences
                    if next_layer_tmp == 'dense' or (layer2add == 'dense' and count == len(layers_lst) - 1):
                        model.add(tf.keras.layers.SimpleRNN(int(units_tmp2)))
                    else:
                        model.add(tf.keras.layers.SimpleRNN(int(units_tmp2), return_sequences=True))
                    units_tmp2 = units_tmp2 / 2
                else:
                    # If the next layer is dense then do not return sequences
                    if next_layer_tmp == 'dense' or (layer2add == 'dense' and count == len(layers_lst) - 1):
                        model.add(tf.keras.layers.SimpleRNN(int(UNITS_BOUND)))
                    else:
                        model.add(tf.keras.layers.SimpleRNN(int(UNITS_BOUND), return_sequences=True))
                    units_tmp2 = UNITS_BOUND
            else:
                if previous_layer_tmp == 'conv' or previous_layer_tmp == 'pool':
                    model.add(Flatten())
                # Add a dense layer by reducing (* 0.5) its neurons if they do not violate our user-defined bound
                if dense_tmp2 >= DENSE_NEURONS_BOUND:
                    model.add(tf.keras.layers.Dense(int(dense_tmp2), activation='relu'))
                    dense_tmp2 = dense_tmp2 / 2
                else:
                    model.add(tf.keras.layers.Dense(int(DENSE_NEURONS_BOUND), activation='relu'))
                    dense_tmp2 = DENSE_NEURONS_BOUND

    return model, conv_tmp2, units_tmp2, dense_tmp2


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")

    streamlit_socket, streamlit_live_socket_tmp, config = start_controller()
    # global streamlit_live_socket
    streamlit_live_socket = streamlit_live_socket_tmp

    # Consumer-part for images
    consumer_images = KafkaConsumer("train-topic", group_id = 'group1', bootstrap_servers = ['127.0.0.1:9092'], auto_offset_reset = 'earliest')
    try:
      received_images = []
      received_labels = []
      parts = []
      for partition in consumer_images.partitions_for_topic("train-topic"):
          parts.append(TopicPartition("train-topic", partition))
      end_offsets = consumer_images.end_offsets(parts)
      end_offset = list(end_offsets.values())[0]

      while list(end_offsets.values())[0] != list(end_offsets.values())[1] or end_offset - tmp_filter_train < 0:
        parts = []
        for partition in consumer_images.partitions_for_topic("train-topic"):
            parts.append(TopicPartition("train-topic", partition))
        end_offsets = consumer_images.end_offsets(parts)
        end_offset = list(end_offsets.values())[0]
        continue

      consumer_images.poll(timeout_ms = 1, update_offsets = False)
      
      for partition in consumer_images.assignment():
        consumer_images.seek(partition, end_offset - tmp_filter_train)

      my_print_flag0 = True
      my_print_flag1 = True
      max0 = -1
      max1 = -1
      for message in consumer_images:
        
        if my_print_flag0 and message.partition == 0: 
          print("Partition0")
          print(message.offset)
          my_print_flag0 = False
        if my_print_flag1 and message.partition == 1: 
          print("Partition1")
          print(message.offset)
          my_print_flag1 = False
        
        if message.partition == 0:
          max0 = message.offset
        else:
          max1 = message.offset

        if message.partition == 0 and len(received_images) < tmp_filter_train:
          decode_img = np.frombuffer(message.value, dtype=np.uint8)
          received_images.append(decode_img)
          del decode_img
        elif message.partition == 1 and len(received_labels) < tmp_filter_train:
          received_labels.append(message.value)
        else:
          continue
        if len(received_images) + len(received_labels) >= 2 * tmp_filter_train:
          print("Spoiler:")
          print(len(received_images))
          print(max0)
          print(len(received_labels))
          print(max1)
          break
    except KeyboardInterrupt:
        sys.exit()
    consumer_images.close()

    # Consumer-part for images
    consumer_images_test = KafkaConsumer("test-topic", group_id = 'group2', bootstrap_servers = ['127.0.0.1:9092'], auto_offset_reset = 'earliest')
    try:
      received_images_test = []
      received_labels_test = []
      parts = []
      for partition in consumer_images_test.partitions_for_topic("test-topic"):
          parts.append(TopicPartition("test-topic", partition))
      end_offsets = consumer_images_test.end_offsets(parts)
      end_offset = list(end_offsets.values())[0]

      while list(end_offsets.values())[0] != list(end_offsets.values())[1] or end_offset - tmp_filter_test < 0:
        parts = []
        for partition in consumer_images_test.partitions_for_topic("test-topic"):
            parts.append(TopicPartition("test-topic", partition))    
        end_offsets = consumer_images_test.end_offsets(parts)
        end_offset = list(end_offsets.values())[0]
        continue


      consumer_images_test.poll(timeout_ms = 1, update_offsets = False)
      
      print(consumer_images_test.assignment())
      for partition in consumer_images_test.assignment():
        consumer_images_test.seek(partition, end_offset - tmp_filter_test)
      
      my_print_flag0 = True
      my_print_flag1 = True
      max0 = -1
      max1 = -1
      for message in consumer_images_test:
        
        if my_print_flag0 and message.partition == 0: 
          print("Partition0")
          print(message.offset)
          my_print_flag0 = False
        if my_print_flag1 and message.partition == 1: 
          print("Partition1")
          print(message.offset)
          my_print_flag1 = False
        
        if message.partition == 0:
          max0 = message.offset
        else:
          max1 = message.offset

        # print ("%s:%d:%d: key=%s value=%s" % (message.topic, message.partition, message.offset, message.key, message.value))
        if message.partition == 0 and len(received_images_test) < tmp_filter_test:
          decode_img = np.frombuffer(message.value, dtype=np.uint8)
          received_images_test.append(decode_img)
          del decode_img
        elif message.partition == 1 and len(received_labels_test) < tmp_filter_test:
          received_labels_test.append(message.value)
        else:
          continue
        if len(received_images_test) + len(received_labels_test) >= 2 * tmp_filter_test:
          print("Spoiler:")
          print(len(received_images_test))
          print(max0)
          print(len(received_labels_test))
          print(max1)
          break
    except KeyboardInterrupt:
        sys.exit()

    consumer_images_test.close()
    received_images_reshaped = []
    for i in range(0, len(received_images)):
      r = received_images[i].reshape(DATASET_SHAPE)
      received_images_reshaped.append(r)

    received_images_reshaped_test = []
    for i in range(0, len(received_images_test)):
      r = received_images_test[i].reshape(DATASET_SHAPE)
      received_images_reshaped_test.append(r)

    received_labels_decoded = []
    for i in range(0, len(received_labels)):
      l = int(received_labels[i].decode("utf-8"))
      received_labels_decoded.append(l)

    received_labels_decoded_test = []
    for i in range(0, len(received_labels_test)):
      l = int(received_labels_test[i].decode("utf-8"))
      received_labels_decoded_test.append(l)

    chunks = np.array_split(received_images_reshaped, len(received_images_reshaped) // SEQUENCE_LENGTH)
    received_images_reshaped = np.array(chunks)
    print(received_images_reshaped.shape)
    tmp_lst = []
    for i in range(0, len(received_labels_decoded), SEQUENCE_LENGTH):
        tmp_lst.append(received_labels_decoded[i])
    received_labels_decoded = np.array(tmp_lst)
    print(received_labels_decoded.shape)

    chunks = np.array_split(received_images_reshaped_test, len(received_images_reshaped_test) // SEQUENCE_LENGTH)
    received_images_reshaped_test = np.array(chunks)
    print(received_images_reshaped_test.shape)
    tmp_lst = []
    for i in range(0, len(received_labels_decoded_test), SEQUENCE_LENGTH):
        tmp_lst.append(received_labels_decoded_test[i])
    received_labels_decoded_test = np.array(tmp_lst)
    print(received_labels_decoded_test.shape)

    print("Received Training Data:")
    print("------> # of received videos:", len(received_images_reshaped))
    print("------> # of received labels:", len(received_labels_decoded))
    print("Received Testing Data:")
    print("------> # of received video:", len(received_images_reshaped_test))
    print("------> # of received labels:", len(received_labels_decoded_test))

    start_bo(config, streamlit_socket, UNIQUE_CLASS_LABELS, received_images_reshaped, received_labels_decoded, received_images_reshaped_test, received_labels_decoded_test, DATASET_SHAPE)
