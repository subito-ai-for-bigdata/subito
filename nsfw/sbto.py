import json
import pickle
import socket
import sys
import random
import math
from datetime import datetime
import time
import json
from kafka import KafkaProducer, KafkaConsumer, TopicPartition
from kafka.errors import KafkaError
import pandas as pd
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.datasets import fashion_mnist, mnist, cifar10, cifar100
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from skopt import gbrt_minimize, gp_minimize
from skopt.utils import use_named_args
from skopt.space import Real, Categorical, Integer
from tensorflow.python.keras import backend as K
import matplotlib.pyplot as plt

try:
    with open('config.json') as json_file:
        config = json.load(json_file)
except:
    print("config.json not found")
    exit()

call_counter = 0
tmp_filter_train = config['bo_data_train']
tmp_filter_test = config['bo_data_test']
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
sample_size_low = config['sample_size_low']
sample_size_high = config['sample_size_high']
num_of_epochs_low = config['num_of_epochs_low']
num_of_epochs_high = config['num_of_epochs_high']
sampling_method_id = 1
acquisition_f = config['acquisition_f']
theta_parameter = config['theta_parameter']
lamda_acc = config['lamda_acc']
bo_call_number = config['bo_call_number']
dataset_shape = [50, 50, 3]
unique_class_labels = range(config['num_of_classes'])
# Percentage of the dataset that will be used for testing
perc_test = 1 #@param {type:"number"}


extra_results = []


dim_sample_size = Real(low=sample_size_low, high=sample_size_high, name='sample_size')
dim_epochs_number = Integer(low=num_of_epochs_low, high=num_of_epochs_high, name='epochs_number')
dim_conv_number = Integer(low=num_of_conv_layers_low, high=num_of_conv_layers_high, name='conv_number')
dim_pool_number = Integer(low=num_of_pool_layers_low, high=num_of_pool_layers_high, name='pool_number')
dim_dense_number = Integer(low=num_of_dense_layers_low, high=num_of_dense_layers_high, name='dense_number')
# dim_lr = Real(low=lr_low, high=lr_high, name='learning_rate')
# dim_batch_size = Integer(low=size_of_batch_low, high=size_of_batch_high, name='batch_size')

dimensions = [dim_sample_size,
              dim_epochs_number,
              dim_conv_number,
              dim_pool_number,
              dim_dense_number
             ]
# default_parameters = [0.2, 5, 4, 2, 2]
default_parameters = [(sample_size_low + sample_size_high) / 2,
                     (num_of_epochs_low + num_of_epochs_high) // 2,
                     (num_of_conv_layers_low + num_of_conv_layers_high) // 2,
                     (num_of_pool_layers_low + num_of_pool_layers_high) // 2,
                     (num_of_dense_layers_low + num_of_dense_layers_high) // 2]

#@markdown ##Other Configuration(s)
CONV_PADDING = 'same'
MAX_POOL_PADDING = 'same'
CONV_NEURONS_CONST = 32
CONV_NEURONS_BOUND = 256
DENSE_NEURONS_CONST = 128
DENSE_NEURONS_BOUND = 32
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
    # Close socket on exit
    streamlit_socket.close()
    streamlit_live_socket.close()
    print("Sockets Closed")

@use_named_args(dimensions = dimensions)
def fitness(sample_size, epochs_number, conv_number, pool_number, dense_number):
    print()
    print(f"EPOCHS to: {epochs_number} \n SAMPLE RATE to: {sample_size} \n NUM_OF_CONV_LAYERS to: {conv_number} \n NUM_OF_POOL_LAYERS to: {pool_number} \n NUM_OF_DENSE_LAYERS to: {dense_number}")
    print()
    global call_counter
    call_counter += 1
    input_str = f"CALL: {call_counter}/{bo_call_number} \n EPOCHS to: {epochs_number} \n SAMPLE RATE to: {sample_size} \n NUM_OF_CONV_LAYERS to: {conv_number} \n NUM_OF_POOL_LAYERS to: {pool_number} \n NUM_OF_DENSE_LAYERS to: {dense_number}\n"
    # Call the sampling method
    train_images, train_labels = sampling_method(sampling_method_id, received_images_reshaped, received_labels_decoded, sample_size)

    # Function that creates the model
    model = create_model(conv_number, pool_number, dense_number, CONV_NEURONS_CONST, DENSE_NEURONS_CONST, CONV_NEURONS_BOUND, DENSE_NEURONS_BOUND)
    stringlist = []
    model.summary(print_fn=lambda x: stringlist.append(x))
    short_model_summary = "\n".join(stringlist)
    model.summary()

    start = time.time()

    blackbox = model.fit(x=train_images,
                        y=train_labels,
                        epochs=epochs_number,
                        batch_size=size_of_batch
                        )
    stop = time.time()

    tr_loss_lst = blackbox.history['loss']
    tr_accuracy_lst = blackbox.history['accuracy']

    # Compute the training speed of this CNN architecture
    tr_time = stop - start

    # Transform to numpy arrays
    received_images_reshaped_test_arr = np.asarray(received_images_reshaped_test)
    received_labels_decoded_test_arr = np.asarray(received_labels_decoded_test)

    # Evaluate our model on the test dataset
    test_loss, test_acc = model.evaluate(received_images_reshaped_test_arr, received_labels_decoded_test_arr)

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
    tmp = "\nAccuracy (on the testing dataset): {0:.2%}".format(test_acc)+'\nTraining time:'+tr_time.__str__()+'\nTradeOff Metric:'+tradeOff_metric.__str__()+'\n\n'
    streamlit_live_socket.sendall(pickle.dumps(input_str + short_model_summary + tmp))

    # Store the accuracy and the training speed of the corresponding model in order to be printed in the final cell
    tmp = [test_acc, tr_time, tr_loss_lst, tr_accuracy_lst]
    extra_results.append(tmp)

    # Delete the Keras model with these hyper-parameters from memory.
    del model

    # Clear the Keras session, otherwise it will keep adding new
    # models to the same TensorFlow graph each time we create
    # a model with a different set of hyper-parameters.
    K.clear_session()
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

    pd_tmp = pd.concat([pd.DataFrame(gp_result.x_iters, columns=["Sample Size", "Epochs", "Conv", "Pool", "Dense"]), (pd.Series(gp_result.func_vals * -1, name="Score"))], axis=1)
    final_result = pd.concat([pd_tmp, df_extra], axis=1)

    # print(gp_result.x)
    print(f" NEW EPOCHS to: {gp_result.x[1]} \n NEW SAMPLE RATE to: {gp_result.x[0]} \n NUM_OF_CONV_LAYERS to: {gp_result.x[2]} \n NUM_OF_POOL_LAYERS to: {gp_result.x[3]} \n NUM_OF_DENSE_LAYERS to: {gp_result.x[4]}")
    
    tmp = gp_result.x[1].__str__() +','+ gp_result.x[0].__str__()+','+gp_result.x[2].__str__() +','+ gp_result.x[3].__str__()+','+ gp_result.x[4].__str__()

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

def create_model(conv_number, pool_number, dense_number, CONV_NEURONS_CONST, DENSE_NEURONS_CONST, CONV_NEURONS_BOUND, DENSE_NEURONS_BOUND):
    # STRATEGY FOR NUM_OF_NEURONS
    # Regarding CONV layers we begin the input of user and as we add conv layers we double the number of neurons, e.g., 1st -> 32, 2nd -> 64, 3rd -> 128 until we reach the upper bound (imput of user)
    # Regarding DENSE layers we begin the input of user and as we add conv layers we divide by 2 the number of neurons, e.g., 1st -> 128, 2nd ->64, 3rd -> 32 until we reach the lower bound (imput of user)

    # INTUITION for this strategy
    # The intution behind this is that early layers deal with primitive concepts and thus having a large amount of neurons wouldn't really benifit after some point,
    # but as you go deeper, the heierarchy of abstractions get richer and richer and you'd want to be able to capture as much information as you can and create
    # new-higher-richer abstaractions better. This is why you increase the neurons as you go deeper.
    # On the other hand, when you reach the end of the network, you'd want to choose the best features out of all the features you have so far developed, so you start
    # to gradually decrease the number of neurons so hopefully you'll end up with the most important features that matters to your specific task.
    model = models.Sequential()
    conv_tmp = CONV_NEURONS_CONST
    dense_tmp = DENSE_NEURONS_CONST

    ############################################################################################
    # Part I: Convolutional part of our network, i.e., extraction of (important) features
    ############################################################################################
    if conv_number > pool_number:
      for i in range(0, int(conv_number) - int(pool_number)):
        if i == 0:
          model.add(layers.Conv2D(int(CONV_NEURONS_CONST), (3, 3), activation='relu', input_shape = dataset_shape, padding=CONV_PADDING))
          conv_tmp = conv_tmp * 2
        else:
          if conv_tmp <= CONV_NEURONS_BOUND:
            model.add(layers.Conv2D(conv_tmp, (3, 3), activation='relu', padding=CONV_PADDING))
            conv_tmp = conv_tmp * 2
          else:
            model.add(layers.Conv2D(CONV_NEURONS_BOUND, (3, 3), activation='relu', padding=CONV_PADDING))

      for i in range(int(conv_number) - int(pool_number), int(conv_number)):
        if conv_tmp <= CONV_NEURONS_BOUND:
          model.add(layers.Conv2D(conv_tmp, (3, 3), activation='relu', padding=CONV_PADDING))
          conv_tmp = conv_tmp * 2
        else:
          model.add(layers.Conv2D(CONV_NEURONS_BOUND, (3, 3), activation='relu', padding=CONV_PADDING))
        model.add(layers.MaxPooling2D((2, 2), strides=(2,2), padding=MAX_POOL_PADDING))

    elif conv_number == pool_number:
      for i in range(0, int(conv_number)):
        if i == 0:
          model.add(layers.Conv2D(int(CONV_NEURONS_CONST), (3, 3), activation='relu', input_shape = dataset_shape, padding=CONV_PADDING))
          conv_tmp = conv_tmp * 2
          model.add(layers.MaxPooling2D((2, 2), strides=(2,2), padding=MAX_POOL_PADDING))
        else:
          if conv_tmp <= CONV_NEURONS_BOUND:
            model.add(layers.Conv2D(conv_tmp, (3, 3), activation='relu', padding=CONV_PADDING))
            conv_tmp = conv_tmp * 2
          else:
            model.add(layers.Conv2D(CONV_NEURONS_BOUND, (3, 3), activation='relu', padding=CONV_PADDING))
          model.add(layers.MaxPooling2D((2, 2), strides=(2,2), padding=MAX_POOL_PADDING))
    else:
      for i in range(0, int(conv_number)):
        if i == 0:
          model.add(layers.Conv2D(int(CONV_NEURONS_CONST), (3, 3), activation='relu', input_shape = dataset_shape, padding=CONV_PADDING))
          conv_tmp = conv_tmp * 2
          model.add(layers.MaxPooling2D((2, 2), strides=(2,2), padding=MAX_POOL_PADDING))
        else:
          if conv_tmp <= CONV_NEURONS_BOUND:
            model.add(layers.Conv2D(conv_tmp, (3, 3), activation='relu', padding=CONV_PADDING))
            conv_tmp = conv_tmp * 2
          else:
            model.add(layers.Conv2D(CONV_NEURONS_BOUND, (3, 3), activation='relu', padding=CONV_PADDING))
          model.add(layers.MaxPooling2D((2, 2), strides=(2,2), padding=MAX_POOL_PADDING))

      for i in range(int(conv_number), int(pool_number)):
        model.add(layers.MaxPooling2D((2, 2), strides=(2,2), padding=MAX_POOL_PADDING))



    ############################################################################################
    # Part II: Dense part of our network, i.e., classification of an image in our classes
    ############################################################################################

    # Converts multi-dimensional matrix to single dimensional matrix.
    model.add(layers.Flatten())

    # Dense Layer is simple layer of neurons in which each neuron receives input from all the neurons of previous layer
    for i in range(0, int(dense_number)):
      if i == 0:
        model.add(layers.Dense(int(DENSE_NEURONS_CONST), activation='relu'))
        dense_tmp = dense_tmp // 2
      else:
        if dense_tmp >= DENSE_NEURONS_BOUND:
          model.add(layers.Dense(dense_tmp, activation='relu'))
          dense_tmp = dense_tmp // 2
        else:
          model.add(layers.Dense(DENSE_NEURONS_BOUND, activation='relu'))


    # Softmax is an activation function that is used mainly for classification tasks
    # It normalizes the input vector into a probability distribution  that is proportional to the exponential of the input numbers.
    model.add(layers.Dense(len(unique_class_labels), activation='softmax'))

    model.compile(optimizer=Adam(lr),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

    return model




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
          decode_img = np.frombuffer(message.value)
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
          decode_img = np.frombuffer(message.value)
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
      r = received_images[i].reshape(dataset_shape)
      received_images_reshaped.append(r)

    received_images_reshaped_test = []
    for i in range(0, len(received_images_test)):
      r = received_images_test[i].reshape(dataset_shape)
      received_images_reshaped_test.append(r)

    received_labels_decoded = []
    for i in range(0, len(received_labels)):
      l = int(received_labels[i].decode("utf-8"))
      received_labels_decoded.append(l)

    received_labels_decoded_test = []
    for i in range(0, len(received_labels_test)):
      l = int(received_labels_test[i].decode("utf-8"))
      received_labels_decoded_test.append(l)


    print("Received Training Data:")
    print("------> # of received images:", len(received_images_reshaped))
    print("------> # of received labels:", len(received_labels_decoded))
    print("Received Testing Data:")
    print("------> # of received images:", len(received_images_reshaped_test))
    print("------> # of received labels:", len(received_labels_decoded_test))

    start_bo(config, streamlit_socket, unique_class_labels, received_images_reshaped, received_labels_decoded, received_images_reshaped_test, received_labels_decoded_test, dataset_shape)
