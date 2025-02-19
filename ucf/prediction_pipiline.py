import gc
import json
import pickle
import sys
import torch
from threading import Thread
from torch.nn import ModuleList
from torch.utils.data import SubsetRandomSampler, TensorDataset
from torchinfo import summary
import numpy as np
import socket
from tqdm import tqdm
from kafka import KafkaProducer, KafkaConsumer, TopicPartition
import torch.nn as nn
import torch.nn.functional as F

USE_GPU_TMP = False
BUILD_TRAINSET = True
CONV_PADDING = 'same'
MAX_POOL_PADDING = 'same'
CONV_NEURONS_CONST = 16
CONV_NEURONS_BOUND = 64
DENSE_NEURONS_CONST = 128
DENSE_NEURONS_BOUND = 32
UNITS_CONST = 32
UNITS_BOUND = 32
SEQUENCE_LENGTH = 20
IMAGE_HEIGHT = 64
IMAGE_WIDTH = 64

if MAX_POOL_PADDING == 'same':
    MAX_POOL_PADDING = 0
elif MAX_POOL_PADDING == 'valid':
    MAX_POOL_PADDING = 1
tmp_filter_test = None
current_offset = 0


def reshape_for_pytorch(res_images, dataset_shape):
    """
    Returns res images reshaped according to dataset_shape

    :param res_images: images vector input
    :param dataset_shape: the shape of the dataset
    :return: properly reshapes res_images
    """
    dataset_shape_for_decoding = [64, 64, 3]
    received_images_reshaped = []
    for i in range(0, len(res_images)):
        r = res_images[i].reshape(dataset_shape_for_decoding)
        received_images_reshaped.append(r)
    received_images = []
    for i in range(0, len(received_images_reshaped)):
        r = received_images_reshaped[i].transpose((2, 0, 1))
        received_images.append(r)
    return received_images


# We need a way to calculate the number of neurons on the output of convolutional layers, sourced online
def flatten(w, k=3, s=1, p=0, m=True):
    """
    Returns the right size of the flattened tensor after convolutional transformation

    :param w: width of the image
    :param k: kernel size
    :param s: stride
    :param p: padding
    :param m: max pooling (bool)
    :return: proper shape and params: use x * x * previous_out_channels
    """
    return int((np.floor((w - k + 2 * p) / s) + 1) / 2 if m else 1), k, s, p, m


class TimeDistributed(nn.Module):
    """
    A PyTorch wrapper that applies a given module independently to each time step
    of an input sequence. This is useful for processing sequential data while
    maintaining spatial or feature integrity across time steps.
    """

    def __init__(self, module, batch_first=False):
        """
        Initializes the TimeDistributed wrapper

        :param module: the model to be applied at each time-step
        :param batch_first: whether the input shape has batch as the first dimension
        """
        super(TimeDistributed, self).__init__()
        self.module = module.to(torch.device('cuda' if USE_GPU_TMP else 'cpu'))
        self.batch_first = batch_first

    def forward(self, x):
        """
        Forward pass through the TimeDistributed module

        :param x: input tensor of shape
        :return: output tensor with the same time-step structure but transformed by the given module
        """
        batch_size, time_steps, C, H, W = x.size()
        if not isinstance(self.module, nn.Flatten):
            c_in = x.contiguous().view(batch_size * time_steps, C, H, W)
            c_out = self.module(c_in)
            r_in = c_out.contiguous().view(batch_size, c_out.shape[0] // batch_size, c_out.shape[1], c_out.shape[2],
                                           c_out.shape[3])
            return r_in
        c_out = self.module(x)
        c_out = c_out.permute(1, 0, 2)
        return c_out


# Define the structure of the Neural Network
class Net(nn.Module):
    FLAT_SHAPE_SIZE = -1

    def __init__(self, layers_lst, layer2add, dataset_shape, CONV_NEURONS_CONST, UNITS_CONST, DENSE_NEURONS_CONST,
                 CONV_NEURONS_BOUND,
                 UNITS_BOUND, DENSE_NEURONS_BOUND):
        """
        Initializes the dynamic NN structure based on the given layer list

        :param layers_lst: A list defining the sequence of layers.
        :param layer2add: The layer type to be added next.
        :param dataset_shape: Shape of the dataset.
        :param CONV_NEURONS_CONST: Initial neurons for convolutional layers.
        :param UNITS_CONST: Initial units for recurrent layers.
        :param DENSE_NEURONS_CONST: Initial neurons for dense layers.
        :param CONV_NEURONS_BOUND: Upper limit for convolutional neurons.
        :param UNITS_BOUND: Upper limit for recurrent layer units.
        :param DENSE_NEURONS_BOUND: Upper limit for dense layer neurons.
        """
        super().__init__()
        conv_tmp2 = CONV_NEURONS_CONST
        conv_tmp_old = conv_tmp2
        dense_tmp2 = DENSE_NEURONS_CONST
        dense_tmp_old = dense_tmp2
        units_tmp2 = UNITS_CONST
        units_tmp_old = units_tmp2
        self.layers_lst = layers_lst
        self.layers = []
        kernel_size = (3, 3)
        self.layers_module_list = ModuleList()

        if layers_lst[0] == 'pool' or len(layers_lst) == 0:
            return -1

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
                tmp_l = TimeDistributed(nn.Conv2d(3, int(conv_tmp2), kernel_size=kernel_size, padding=CONV_PADDING))
                self.layers.append(tmp_l)
                self.layers_module_list.append(tmp_l)
                conv_tmp_old = conv_tmp2
                conv_tmp2 = conv_tmp2 * 2
            # First layer lstm-gru-rnn (change the shape of the input) and next or 2-be-added layer lstm-gru-rnn (should add the 'return conf')
            elif ((layer == 'lstm' or layer == 'gru' or layer == 'rnn') and (((count == 0) and len(
                layers_lst) == 1 and (layer2add == 'lstm' or layer2add == 'gru' or layer2add == 'rnn')) or (
                                                                                 (count == 0) and (
                                                                                 next_layer_tmp == 'lstm' or next_layer_tmp == 'gru' or next_layer_tmp == 'rnn')))):
                if layer == 'lstm':
                    tmp_l = nn.LSTM(IMAGE_HEIGHT * IMAGE_WIDTH * 3, int(units_tmp2), batch_first=True)
                    self.layers.append(tmp_l)
                    self.layers_module_list.append(tmp_l)
                elif layer == 'gru':
                    tmp_l = nn.GRU(IMAGE_HEIGHT * IMAGE_WIDTH * 3, int(units_tmp2), batch_first=True)
                    self.layers.append(tmp_l)
                    self.layers_module_list.append(tmp_l)
                else:
                    tmp_l = nn.RNN(IMAGE_HEIGHT * IMAGE_WIDTH * 3, int(units_tmp2), batch_first=True)
                    self.layers.append(tmp_l)
                    self.layers_module_list.append(tmp_l)
                units_tmp2 = units_tmp2 / 2
            # First layer lstm-gru-rnn (change the shape of the input)
            elif ((layer == 'lstm' or layer == 'gru' or layer == 'rnn') and count == 0):
                if layer == 'lstm':
                    tmp_l = nn.LSTM(IMAGE_HEIGHT * IMAGE_WIDTH * 3, int(units_tmp2), batch_first=True)
                    self.layers.append(tmp_l)
                    self.layers_module_list.append(tmp_l)
                    self.layers.append("Forget Sequence")
                elif layer == 'gru':
                    tmp_l = nn.GRU(IMAGE_HEIGHT * IMAGE_WIDTH * 3, int(units_tmp2), batch_first=True)
                    self.layers.append(tmp_l)
                    self.layers_module_list.append(tmp_l)
                    self.layers.append("Forget Sequence")
                else:
                    tmp_l = nn.RNN(IMAGE_HEIGHT * IMAGE_WIDTH * 3, int(units_tmp2), batch_first=True)
                    self.layers.append(tmp_l)
                    self.layers_module_list.append(tmp_l)
                    self.layers.append("Forget Sequence")
                units_tmp2 = units_tmp2 / 2
            # First layer dense (change the shape of the input)
            elif layer == 'dense' and count == 0:
                tmp_l = nn.Linear(int(SEQUENCE_LENGTH * IMAGE_HEIGHT * IMAGE_WIDTH * 3), int(dense_tmp2))
                self.layers.append(tmp_l)
                self.layers_module_list.append(tmp_l)
                dense_tmp2 = dense_tmp2 / 2
            # For the remaining layers
            else:
                if layer == 'conv':
                    # Add a conv layer by doubling its neurons if they do not violate our user-defined bound
                    if conv_tmp2 <= CONV_NEURONS_BOUND:
                        tmp_l = TimeDistributed(
                            nn.Conv2d(conv_tmp_old, int(conv_tmp2), kernel_size=kernel_size, padding=CONV_PADDING))
                        self.layers.append(tmp_l)
                        self.layers_module_list.append(tmp_l)
                        conv_tmp_old = conv_tmp2
                        conv_tmp2 = conv_tmp2 * 2
                    else:
                        tmp_l = TimeDistributed(
                            nn.Conv2d(conv_tmp_old, int(CONV_NEURONS_BOUND), kernel_size=kernel_size,
                                      padding=CONV_PADDING))
                        self.layers.append(tmp_l)
                        self.layers_module_list.append(tmp_l)
                        conv_tmp_old = conv_tmp2
                        conv_tmp2 = CONV_NEURONS_BOUND
                elif layer == 'pool':
                    # Add a pool layer
                    tmp_l = TimeDistributed(nn.MaxPool2d(kernel_size=4, padding=MAX_POOL_PADDING))
                    self.layers.append(tmp_l)
                    self.layers_module_list.append(tmp_l)
                elif layer == 'lstm':
                    # If the previous layer is conv or pool add a flatten layer first
                    if previous_layer_tmp == 'conv' or previous_layer_tmp == 'pool':
                        tmp_l = TimeDistributed(
                            nn.BatchNorm2d(self.calculate_flatten_dim(dataset_shape, called_from_batch_norm=True)))
                        self.layers_module_list.append(tmp_l)
                        self.layers.append(tmp_l)
                        tmp_l = TimeDistributed(nn.Flatten(start_dim=2))
                        self.layers_module_list.append(tmp_l)
                        self.layers.append("Flatten")
                    # Add a lstm layer by reducing (* 0.5) its units if they do not violate our user-defined bound
                    if units_tmp2 >= UNITS_BOUND:
                        # If the next layer is dense then do not return sequences
                        if next_layer_tmp == 'dense' or (layer2add == 'dense' and count == len(layers_lst) - 1):
                            tmp_l = nn.LSTM(self.calculate_flatten_dim(dataset_shape), int(units_tmp2),
                                            batch_first=True)
                            self.layers_module_list.append(tmp_l)
                            self.layers.append(tmp_l)
                            self.layers.append("Forget Sequence")
                        else:
                            tmp_l = nn.LSTM(self.calculate_flatten_dim(dataset_shape), int(units_tmp2),
                                            batch_first=True)
                            self.layers_module_list.append(tmp_l)
                            self.layers.append(tmp_l)
                        units_tmp2 = units_tmp2 / 2
                    else:
                        # If the next layer is dense then do not return sequences
                        if next_layer_tmp == 'dense' or (layer2add == 'dense' and count == len(layers_lst) - 1):
                            tmp_l = nn.LSTM(self.calculate_flatten_dim(dataset_shape), int(UNITS_BOUND),
                                            batch_first=True)
                            self.layers_module_list.append(tmp_l)
                            self.layers.append(tmp_l)
                            self.layers.append("Forget Sequence")
                        else:
                            tmp_l = nn.LSTM(self.calculate_flatten_dim(dataset_shape), int(UNITS_BOUND),
                                            batch_first=True)
                            self.layers_module_list.append(tmp_l)
                            self.layers.append(tmp_l)
                        units_tmp2 = UNITS_BOUND
                elif layer == 'gru':
                    # If the previous layer is conv or pool add a flatten layer first
                    if previous_layer_tmp == 'conv' or previous_layer_tmp == 'pool':
                        tmp_l = TimeDistributed(
                            nn.BatchNorm2d(self.calculate_flatten_dim(dataset_shape, called_from_batch_norm=True)))
                        self.layers_module_list.append(tmp_l)
                        self.layers.append(tmp_l)
                        tmp_l = TimeDistributed(nn.Flatten(start_dim=2))
                        self.layers_module_list.append(tmp_l)
                        self.layers.append("Flatten")
                    # Add a gru layer by reducing (* 0.5) its units if they do not violate our user-defined bound
                    if units_tmp2 >= UNITS_BOUND:
                        # If the next layer is dense then do not return sequences
                        if next_layer_tmp == 'dense' or (layer2add == 'dense' and count == len(layers_lst) - 1):
                            tmp_l = nn.GRU(self.calculate_flatten_dim(dataset_shape), int(units_tmp2), batch_first=True)
                            self.layers_module_list.append(tmp_l)
                            self.layers.append(tmp_l)
                            self.layers.append("Forget Sequence")
                        else:
                            tmp_l = nn.GRU(self.calculate_flatten_dim(dataset_shape), int(units_tmp2), batch_first=True)
                            self.layers_module_list.append(tmp_l)
                            self.layers.append(tmp_l)
                        units_tmp2 = units_tmp2 / 2
                    else:
                        # If the next layer is dense then do not return sequences
                        if next_layer_tmp == 'dense' or (layer2add == 'dense' and count == len(layers_lst) - 1):
                            tmp_l = nn.GRU(self.calculate_flatten_dim(dataset_shape), int(UNITS_BOUND),
                                           batch_first=True)
                            self.layers_module_list.append(tmp_l)
                            self.layers.append(tmp_l)
                            self.layers.append("Forget Sequence")
                        else:
                            tmp_l = nn.GRU(self.calculate_flatten_dim(dataset_shape), int(UNITS_BOUND),
                                           batch_first=True)
                            self.layers_module_list.append(tmp_l)
                            self.layers.append(tmp_l)
                        units_tmp2 = UNITS_BOUND
                elif layer == 'rnn':
                    # If the previous layer is conv or pool add a flatten layer first
                    if previous_layer_tmp == 'conv' or previous_layer_tmp == 'pool':
                        tmp_l = TimeDistributed(
                            nn.BatchNorm2d(self.calculate_flatten_dim(dataset_shape, called_from_batch_norm=True)))
                        self.layers_module_list.append(tmp_l)
                        self.layers.append(tmp_l)
                        tmp_l = TimeDistributed(nn.Flatten(start_dim=2))
                        self.layers_module_list.append(tmp_l)
                        self.layers.append("Flatten")
                    # Add a gru layer by reducing (* 0.5) its units if they do not violate our user-defined bound
                    if units_tmp2 >= UNITS_BOUND:
                        # If the next layer is dense then do not return sequences
                        if next_layer_tmp == 'dense' or (layer2add == 'dense' and count == len(layers_lst) - 1):
                            tmp_l = nn.RNN(self.calculate_flatten_dim(dataset_shape), int(units_tmp2), batch_first=True)
                            self.layers_module_list.append(tmp_l)
                            self.layers.append(tmp_l)
                            self.layers.append("Forget Sequence")
                        else:
                            tmp_l = nn.RNN(self.calculate_flatten_dim(dataset_shape), int(units_tmp2), batch_first=True)
                            self.layers_module_list.append(tmp_l)
                            self.layers.append(tmp_l)
                        units_tmp2 = units_tmp2 / 2
                    else:
                        # If the next layer is dense then do not return sequences
                        if next_layer_tmp == 'dense' or (layer2add == 'dense' and count == len(layers_lst) - 1):
                            tmp_l = nn.RNN(self.calculate_flatten_dim(dataset_shape), int(UNITS_BOUND),
                                           batch_first=True)
                            self.layers_module_list.append(tmp_l)
                            self.layers.append(tmp_l)
                            self.layers.append("Forget Sequence")
                        else:
                            tmp_l = nn.RNN(self.calculate_flatten_dim(dataset_shape), int(UNITS_BOUND),
                                           batch_first=True)
                            self.layers_module_list.append(tmp_l)
                            self.layers.append(tmp_l)
                        units_tmp2 = UNITS_BOUND
                else:
                    if previous_layer_tmp == 'conv' or previous_layer_tmp == 'pool':
                        tmp_l = nn.Flatten()
                        self.layers_module_list.append(tmp_l)
                        self.layers.append("Flatten")
                    # Add a dense layer by reducing (* 0.5) its neurons if they do not violate our user-defined bound
                    if dense_tmp2 >= DENSE_NEURONS_BOUND:
                        tmp_l = nn.Linear(self.calculate_flatten_dim(dataset_shape), int(dense_tmp2))
                        self.layers_module_list.append(tmp_l)
                        self.layers.append(tmp_l)
                        dense_tmp2 = dense_tmp2 / 2
                    else:
                        tmp_l = nn.Linear(self.calculate_flatten_dim(dataset_shape), int(DENSE_NEURONS_BOUND))
                        self.layers_module_list.append(tmp_l)
                        self.layers.append(tmp_l)
                        dense_tmp2 = DENSE_NEURONS_BOUND
        # If the just-added-layer was conv or pool then add manually a flatten layer
        if 'lstm' not in layers_lst and 'gru' not in layers_lst and 'rnn' not in layers_lst and 'dense' not in layers_lst:
            tmp_l = nn.Flatten()
            self.layers_module_list.append(tmp_l)
            self.layers.append("Flatten")

        # Softmax is an activation function that is used mainly for classification tasks
        # It normalizes the input vector into a probability distribution  that is proportional to the exponential of the input numbers.
        tmp_l = nn.Linear(self.calculate_flatten_dim(dataset_shape), len(unique_class_labels))
        self.layers_module_list.append(tmp_l)

    def forward(self, x):
        """
        Forward pass for the dynamic model

        :param x: Input tensor with shape (batch_size, time_steps, height, width, channels)
        :return: output tensor after passing through all layers
        """
        # Here we reshape the input of the network based on the type of the first layer of the network
        # If the first layer is conv
        if self.layers_lst[0] == 'conv':
            reshaped_x = x
        # If the first layer is lstm-gru-rnn
        elif self.layers_lst[0] == 'lstm' or self.layers_lst[0] == 'gru' or self.layers_lst[0] == 'rnn':
            num_samples, num_frames, height, width, channels = x.shape
            reshaped_x = x.reshape(num_samples, num_frames, height * width * channels)
        # If the first layer is dense
        else:
            num_samples, num_frames, height, width, channels = x.shape
            reshaped_x = x.reshape(num_samples, num_frames * height * width * channels)

        x = reshaped_x

        for i, layer in enumerate(self.layers):
            if layer == 'Flatten':
                catch_flat_layer = None
                for mod in self.layers_module_list:
                    if isinstance(mod, nn.Flatten):
                        catch_flat_layer = mod
                    elif isinstance(mod, TimeDistributed):
                        if isinstance(mod.module, nn.Flatten):
                            catch_flat_layer = mod.module
                x = catch_flat_layer(x)
                continue
            elif isinstance(layer, nn.Linear) or (
                isinstance(layer, TimeDistributed) and isinstance(layer.module, nn.Conv2d)):
                x = F.relu(layer(x))
            elif isinstance(layer, TimeDistributed) and isinstance(layer.module, nn.MaxPool2d):
                x = layer(x)
            elif isinstance(layer, TimeDistributed) and isinstance(layer.module, nn.BatchNorm2d):
                x = layer(x)
            elif isinstance(layer, nn.LSTM):
                x, _ = layer(x)
            elif isinstance(layer, nn.GRU) or isinstance(layer, nn.RNN):
                x, _ = layer(x)
            else:
                if layer == 'Forget Sequence':
                    x = x[:, -1, :]
        x = self.layers_module_list[-1](x)
        return x

    def calculate_flatten_dim(self, dataset_shape, called_from_batch_norm=False):
        """
        Calculates the flattened dimension of the dataset after passing through the network layers.

        This method iteratively processes a dummy input tensor through the model layers to determine
        the final shape of the feature map before classification. The function is particularly useful
        for adapting the architecture dynamically when dealing with variable input shapes.

        :param dataset_shape: The shape of the dataset input as (frames, height, width, channels).
        :param called_from_batch_norm (bool, optional): If True, returns the second dimension of the output (useful for BatchNorm layers). If False, returns the last dimension of the output (final flattened feature size).

        :return: The calculated flattened feature dimension.
        """
        x = torch.zeros(1, *dataset_shape)
        h = -1
        c = -1
        with torch.no_grad():
            for i, layer in enumerate(self.layers):
                if i == 0:
                    if isinstance(layer, nn.GRU) or isinstance(layer, nn.RNN) or isinstance(layer, nn.LSTM):
                        x = x.reshape(1, dataset_shape[0], dataset_shape[1] * dataset_shape[2] * dataset_shape[3])
                    elif isinstance(layer, nn.Linear):
                        x = x.reshape(1, dataset_shape[0] * dataset_shape[1] * dataset_shape[2] * dataset_shape[3])
                if isinstance(layer, TimeDistributed) and isinstance(layer.module, nn.Conv2d):
                    x = F.relu(layer(x))
                elif isinstance(layer, TimeDistributed) and isinstance(layer.module, nn.MaxPool2d):
                    x = layer(x)
                elif isinstance(layer, TimeDistributed) and isinstance(layer.module, nn.BatchNorm2d):
                    x = layer(x)
                elif isinstance(layer, nn.GRU) or isinstance(layer, nn.RNN):
                    try:
                        x, h = layer(x, h)
                    except:
                        x, h = layer(x)
                elif isinstance(layer, nn.LSTM):
                    try:
                        x, (h, c) = layer(x, (h, c))
                    except:
                        x, (h, c) = layer(x)
                elif layer == 'Forget Sequence':
                    x = x[:, -1, :]
                elif isinstance(layer, nn.Linear):
                    x = layer(x)
                else:
                    catch_flat_layer = None
                    for mod in self.layers_module_list:
                        if isinstance(mod, nn.Flatten):
                            catch_flat_layer = mod
                            break
                        elif isinstance(mod, TimeDistributed):
                            if isinstance(mod.module, nn.Flatten):
                                catch_flat_layer = mod.module
                                break
                    x = catch_flat_layer(x)
        if called_from_batch_norm:
            return x.shape[2]
        elif len(x.shape) < 2:
            return x.numel()
        else:
            return x.shape[-1]


class Trainer:
    """
    Class to handle training of the neural network and the updating of sample_size and epochs
    """
    epochs, sample_size, trainset, testset, device = [None, None, None, None, None]
    received_images_reshaped = None
    received_labels_decoded = None
    old_sample_size = None
    sample_size_slack = None
    needed_shape = None
    server_socket = None
    server_nas_socket = None
    rebuild_data = None
    net = None
    live_socket = None

    def __init__(self, rebuild_data, num_of_conv_layers, num_of_pool_layers, num_of_dense_layers, num_of_lstm_layers,
                 num_of_gru_layers, num_of_rnn_layers, unique_class_labels, sequence_length, class_stats):
        """
        Initializes the Trainer class with network architecture parameters and dataset configurations.

        :param rebuild_data: If False, only resamples the train set.
        :param num_of_conv_layers: Number of convolutional layers.
        :param num_of_pool_layers: Number of pooling layers.
        :param num_of_dense_layers: Number of dense layers.
        :param num_of_lstm_layers: Number of LSTM layers.
        :param num_of_gru_layers: Number of GRU layers.
        :param num_of_rnn_layers: Number of RNN layers.
        :param unique_class_labels: Unique class labels in the dataset.
        :param sequence_length: Length of the input sequence.
        :param class_stats: Statistics of class distribution.
        """
        self.class_stats = class_stats
        self.rebuild_data = rebuild_data
        self.num_of_conv_layers = num_of_conv_layers
        self.num_of_pool_layers = num_of_pool_layers
        self.num_of_dense_layers = num_of_dense_layers
        self.num_of_lstm_layers = num_of_lstm_layers
        self.num_of_gru_layers = num_of_gru_layers
        self.num_of_rnn_layers = num_of_rnn_layers
        self.sequence_length = sequence_length
        self.layers_lst = []
        if rebuild_data:
            self.unique_class_labels = unique_class_labels
            self.new_weights_flag = False
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
                print("Running on the GPU")
            else:
                self.device = torch.device("cpu")
                print("Running on the CPU")
            if config['USE_KAFKA']:
                consumer_images = KafkaConsumer("train-topic", group_id='group10', bootstrap_servers=['127.0.0.1:9092'],
                                                auto_offset_reset='earliest')
                try:
                    received_images = []
                    received_labels = []
                    tmp_count = 0
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
                        parts = []
                        for partition in consumer_images.partitions_for_topic("train-topic"):
                            parts.append(TopicPartition("train-topic", partition))
                        end_offsets = consumer_images.end_offsets(parts)
                        end_offset = list(end_offsets.values())[0]
                        if message.partition == 0:
                            decode_img = np.frombuffer(message.value, dtype=np.uint8)
                            received_images.append(decode_img)
                            del decode_img
                        else:
                            received_labels.append(message.value)
                        tmp_count = tmp_count + 1
                        if tmp_count >= 2 * (end_offset):
                            consumer_images.poll(timeout_ms=1, update_offsets=False)
                            for partition in consumer_images.assignment():
                                consumer_images.seek(partition, 0)
                            print("Spoiler:")
                            print(len(received_images))
                            print(max0)
                            print(len(received_labels))
                            print(max1)
                            break
                except KeyboardInterrupt:
                    sys.exit()
                consumer_images.close()
                print(f"Receiven Images: {len(received_images)}")
                print(f"Received Labels: {len(received_labels)}")
                self.received_images_reshaped = reshape_for_pytorch(received_images, dataset_shape_torch)
                self.received_labels_decoded = []
                for i in range(0, len(received_labels)):
                    l = int(received_labels[i].decode("utf-8"))
                    self.received_labels_decoded.append(l)
                chunks = np.array_split(self.received_images_reshaped,
                                        len(self.received_images_reshaped) // self.sequence_length)
                self.received_images_reshaped = np.array(chunks)
                print(self.received_images_reshaped.shape)
                tmp_lst = []
                for i in range(0, len(self.received_labels_decoded), self.sequence_length):
                    tmp_lst.append(self.received_labels_decoded[i])
                self.received_labels_decoded = np.array(tmp_lst)
                print(self.received_labels_decoded.shape)

                train_images = self.received_images_reshaped
                train_labels = self.received_labels_decoded
                # Build initial or updated trainset (be careful Tensors to float32 not float64)
                train_images_tensor = torch.tensor(train_images, dtype=torch.float32)
                train_labels_tensor = torch.tensor(train_labels, dtype=torch.long)
                train_dataset = TensorDataset(train_images_tensor, train_labels_tensor)
                self.testset = torch.utils.data.DataLoader(train_dataset, shuffle=True)

        if rebuild_data:
            self.connect_socket()
            self.start_controller()
            rebuild_data = False
        pass
        while True:
            try:
                self.create_layers_lst(self.num_of_conv_layers, self.num_of_pool_layers, self.num_of_lstm_layers,
                                       self.num_of_gru_layers, self.num_of_rnn_layers, self.num_of_dense_layers)
                self.net = Net(self.layers_lst, 'dense', dataset_shape_torch, CONV_NEURONS_CONST, UNITS_CONST,
                               DENSE_NEURONS_CONST, CONV_NEURONS_BOUND, UNITS_BOUND, DENSE_NEURONS_BOUND)
                self.net.load_state_dict(torch.load('weights_only.pth'))
                break
            except:
                print("Trying to read file...")
                continue
        self.new_weights_flag = False
        self.net.to(self.device)
        for i, layer in enumerate(self.net.layers):
            if layer != 'Forget Sequence' and layer != 'Flatten':
                self.net.layers[i] = layer.to(self.device)
        summary(self.net, input_size=(1, *tuple(dataset_shape_torch)))

    def train(self):
        """
        Train the Neural Network with the trainset
        """
        net = self.net
        net.to(self.device)
        flag_restart = True
        # Used for non-definite number of epochs
        net.eval()
        comm_latency_counter = 0
        comm_latency = 50
        try:
            print(net.layers[-1].weight)
        except:
            pass
        with torch.no_grad():
            for X, y in tqdm(self.testset):
                X, y = X.to(self.device), y.to(self.device)
                # Forward pass
                output = net(X)
                _, predicted_y = torch.max(output.data, 1)
                self.class_stats[predicted_y.item()] += 1
                comm_latency_counter += 1
                if comm_latency_counter == comm_latency:
                    comm_latency_counter = 0
                    self.live_socket.sendall(pickle.dumps(self.class_stats))
        del X, y
        if flag_restart:
            return 1
        else:
            try:
                self.disconnect_socket(conn)
                pass
            except:
                pass
            print("Trying to dc")
            return net

    def create_layers_lst(self, conv_number, pool_number, lstm_number, gru_number, rnn_number, dense_number):
        """
        Create a list of layers based on the provided architecture parameters.

        :param conv_number: Number of convolutional layers.
        :param pool_number: Number of pooling layers.
        :param lstm_number: Number of LSTM layers.
        :param gru_number: Number of GRU layers.
        :param rnn_number: Number of RNN layers.
        :param dense_number: Number of dense layers.
        """
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
        self.layers_lst = layers_lst
        print(self.layers_lst)

    def clear_mem(self):
        """
        Clear memory by deleting the model and emptying the CUDA cache.
        """
        for layer in self.net.layers:
            del layer
        del self.net  # Delete the model
        torch.cuda.empty_cache()
        gc.collect()

    def socket_listener_weights(self, conn):
        """
        Listen for updated model weights from a socket connection.

        :param conn: Received from connect_socket().
        """
        while True:
            # Receive data from the socket
            data = conn.recv(8192)
            if not data:
                break
            try:
                self.new_weights_flag = True
                print("Updated Model received")
            except ValueError:
                print("Invalid input. Please enter a valid model.")

    def socket_listener_nas(self, conn):
        """
        Listen for NAS (Neural Architecture Search) updates from a socket connection.

        :param conn: Received from connect_socket().
        """
        try:
            with open('config_video.json') as json_file:
                config = json.load(json_file)
        except:
            print("config_video.json not found")
            exit()
            args = sys.argv[1:]
        unique_class_labels = range(len(config['classes_list']))
        while True:
            # Receive data from the socket
            data = conn.recv(8192)
            if not data:
                break
            print(pickle.loads(data))
            (self.num_of_conv_layers, self.num_of_pool_layers, self.num_of_dense_layers, self.num_of_lstm_layers,
             self.num_of_gru_layers, self.num_of_rnn_layers) = pickle.loads(data)
            self.class_stats = [0] * len(unique_class_labels)
            print("Updated Model NAS received")

    def connect_socket(self):
        """
        Initialize a server socket as a new thread and wait for connections

        :return: [socket, listening_thread] instances
        """
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind((config['host_address'], config['prediction_port']))
        self.server_socket.listen(1)  # Allow 1 failed connection
        print("Waiting for production pipeline to connect...")
        conn, addr = self.server_socket.accept()
        print("Connected by", addr)
        # Spawn and start a thread to listen for new data
        listen_thread = Thread(target=self.socket_listener_weights, args=(conn,))
        listen_thread.daemon = True
        listen_thread.start()
        self.server_nas_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_nas_socket.bind((config['host_address'], config['prediction_nas_port']))
        self.server_nas_socket.listen(1)  # Allow 1 failed connection
        conn, addr = self.server_nas_socket.accept()
        print("Connected by", addr)
        # Spawn and start a thread to listen for new data
        listen_thread = Thread(target=self.socket_listener_nas, args=(conn,))
        listen_thread.daemon = True
        listen_thread.start()
        return conn, listen_thread

    def disconnect_socket(self, conn):
        """
        Just close the connection

        :param conn: connection instance
        """
        conn.close()

    def start_controller(self):
        """
        Initialize the socket and listen for keyboard input
        """
        # Open config file and get the desired port for socket communication
        try:
            with open('config_video.json') as json_file:
                config = json.load(json_file)
        except:
            print("config_video.json not found")
            exit()
        # Initialize an IPv4 socket with TCP (default) and try to connect to the nn
        self.live_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.live_socket.connect((config["host_address"], config['prediction_live_port']))
        return


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")
    try:
        with open('config_video.json') as json_file:
            config = json.load(json_file)
    except:
        print("config_video.json not found")
        exit()
        args = sys.argv[1:]
    sequence_length = config['sequence_length']
    tmp_filter_test = config['stream_batch_test'] * sequence_length
    sampling_method_id = 1
    unique_class_labels = range(len(config['classes_list']))
    current_offset = 0
    dataset_shape_torch = [20, 3, 64, 64]
    class_stats = [0] * len(unique_class_labels)
    tr = Trainer(True, config['num_of_conv_layers'], config['num_of_pool_layers'], config['num_of_dense_layers'],
                 config['num_of_lstm_layers'], config['num_of_gru_layers'], config['num_of_rnn_layers'],
                 unique_class_labels, sequence_length, class_stats)
    net = tr.train()
    while net == 1:
        tr.__init__(False, tr.num_of_conv_layers, tr.num_of_pool_layers, tr.num_of_dense_layers,
                    tr.num_of_lstm_layers, tr.num_of_gru_layers, tr.num_of_rnn_layers, unique_class_labels,
                    sequence_length, tr.class_stats)
        net = tr.train()
        tr.clear_mem()
