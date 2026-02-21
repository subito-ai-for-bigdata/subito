import gc
import json
import multiprocessing
import pickle
import sys
import time
import torch
from threading import Thread
from kafka import KafkaConsumer, TopicPartition
from torch.utils.data import TensorDataset
import numpy as np
import socket
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary


BUILD_TRAINSET = True
USE_GPU_TMP = False
CONV_PADDING = 'same'
MAX_POOL_PADDING = 'same'
CONV_NEURONS_CONST = 32
CONV_NEURONS_BOUND = 256
DENSE_NEURONS_CONST = 128
DENSE_NEURONS_BOUND = 32
UNITS_CONST = 32
UNITS_BOUND = 32
try:
    with open('config_video.json') as json_file:
        config = json.load(json_file)
except:
    print("config_video.json not found")
    exit()
dataset_shape_torch = config['dataset_shape_torch']
IMAGE_HEIGHT = dataset_shape_torch[3]
IMAGE_WIDTH = dataset_shape_torch[2]
SEQUENCE_LENGTH = dataset_shape_torch[0]
if MAX_POOL_PADDING == 'same':
    MAX_POOL_PADDING = 0
elif MAX_POOL_PADDING == 'valid':
    MAX_POOL_PADDING = 1
current_offset = 0


def reshape_for_pytorch(res_images, dataset_shape):
    """
    Returns res images reshaped according to dataset_shape

    :param res_images: images vector input
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

    :param w: width of image
    :param k: kernel size
    :param s: stride
    :param p: padding
    :param m: max pooling (bool)
    :return: proper shape and params: use x * x * previous_out_channels
    """
    return int((np.floor((w - k + 2 * p) / s) + 1) / 2 if m else 1), k, s, p, m

class TimeDistributed(nn.Module):
    def __init__(self, module, batch_first=False):
        super(TimeDistributed, self).__init__()
        self.module = module.to(torch.device('cuda' if USE_GPU_TMP else 'cpu'))
        self.batch_first = batch_first

    def forward(self, x):
        """
        x size: (batch_size, time_steps, in_channels, height, width)
        """
        batch_size, time_steps, C, H, W = x.size()
        if not isinstance(self.module, nn.Flatten):
            c_in = x.contiguous().view(batch_size * time_steps, C, H, W)
            c_out = self.module(c_in)
            r_in = c_out.contiguous().view(batch_size, c_out.shape[0] // batch_size, c_out.shape[1], c_out.shape[2],
                                         c_out.shape[3])
            # if self.batch_first is False:
            #     r_in = r_in.permute(1, 0, 2)
            return r_in
        c_out = self.module(x)
        c_out = c_out.permute(1, 0, 2)
        return c_out

# Define the structure of the Neural Network
class Net(nn.Module):
    FLAT_SHAPE_SIZE = -1

    def __init__(self, layers_lst, layer2add, dataset_shape):
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
        self.layers_module_list = nn.ModuleList()

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
                        # self.flat = TimeDistributed(nn.Flatten(start_dim=2))
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
                        # self.flat = TimeDistributed(nn.Flatten(start_dim=2))
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
                        # self.flat = TimeDistributed(nn.Flatten(start_dim=2))
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
                        # self.flat = nn.Flatten()
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
            # self.flat = nn.Flatten()
            tmp_l = nn.Flatten()
            self.layers_module_list.append(tmp_l)
            self.layers.append("Flatten")

        # Softmax is an activation function that is used mainly for classification tasks
        # It normalizes the input vector into a probability distribution  that is proportional to the exponential of the input numbers.
        # self.output_layer = nn.Linear(self.calculate_flatten_dim(dataset_shape), len(unique_class_labels))
        tmp_l = nn.Linear(self.calculate_flatten_dim(dataset_shape), len(unique_class_labels))
        self.layers_module_list.append(tmp_l)

    def forward(self, x):
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
               num_of_gru_layers, num_of_rnn_layers, unique_class_labels, sequence_length, batch_size, rank):
        """
        Initialize a Trainer object

        :param rebuild_data: if False we only resample the train set
        :param num_of_conv_layers: Number of convolutional layers
        :param num_of_pool_layers: Number of pooling layers
        :param num_of_dense_layers: Number of dense layers
        :param num_of_lstm_layers: Number of LSTM layers
        :param num_of_gru_layers: Number of GRU layers
        :param num_of_rnn_layers: Number of RNN layers
        :param unique_class_labels: unique class labels
        :param sequence_length: sequence length
        :param batch_size: batch size
        :param rank: predictor_id
        """
        self.layers_lst = []
        self.rebuild_data = rebuild_data
        self.num_of_conv_layers = num_of_conv_layers
        self.num_of_pool_layers = num_of_pool_layers
        self.num_of_dense_layers = num_of_dense_layers
        self.num_of_lstm_layers = num_of_lstm_layers
        self.num_of_gru_layers = num_of_gru_layers
        self.num_of_rnn_layers = num_of_rnn_layers
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.unique_class_labels = unique_class_labels
        self.rank = rank
        self.class_stats = [0] * (len(unique_class_labels) + 1)
        if rebuild_data:
            self.updated_NAS = False
            self.new_weights_flag = False
            self.state_dict = None
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
                print("Running on the GPU")
            else:
                self.device = torch.device("cpu")
                print("Running on the CPU")
            if config['USE_KAFKA']:
                consumer_images = KafkaConsumer("train-topic", group_id='pr_client'+self.rank.__str__(),
                                                bootstrap_servers=['127.0.0.1:9092'], auto_offset_reset='earliest')
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
                print(f"Received Images: {len(received_images)}")
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
                train_images_tensor = train_images_tensor / 255.0
                train_labels_tensor = torch.tensor(train_labels, dtype=torch.long)
                train_dataset = TensorDataset(train_images_tensor, train_labels_tensor)
                self.testset = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=self.batch_size)
                if self.state_dict is None:
                    self.state_dict = torch.load('weights_only.pth')
                self.create_layers_lst(self.num_of_conv_layers, self.num_of_pool_layers, self.num_of_lstm_layers,
                                       self.num_of_gru_layers, self.num_of_rnn_layers, self.num_of_dense_layers)
                self.net = Net(self.layers_lst, 'dense', dataset_shape_torch)
                summary(self.net, input_size=(1, *tuple(dataset_shape_torch)))
        if self.updated_NAS:
            self.create_layers_lst(self.num_of_conv_layers, self.num_of_pool_layers, self.num_of_lstm_layers,
                                   self.num_of_gru_layers, self.num_of_rnn_layers, self.num_of_dense_layers)
            self.net = Net(self.layers_lst, 'dense', dataset_shape_torch)
            self.updated_NAS = False
            while(True):
                try:
                    self.net.load_state_dict(self.state_dict)
                    print("loading weights from socket")
                    break
                except:
                    print('waiting for weights loading')
                    continue
        elif self.new_weights_flag:
            try:
                self.net.load_state_dict(self.state_dict)
                print("loading weights from socket")
            except:
                return
        self.new_weights_flag = False
        self.net.to(self.device)
        for i, layer in enumerate(self.net.layers):
            if layer != 'Forget Sequence' and layer != 'Flatten':
                self.net.layers[i] = layer.to(self.device)

    def train(self):
        """
        Train the Neural Network with the trainset
        """
        self.net.to(self.device)
        flag_restart = True
        self.net.eval()
        inf_times = []
        num_of_elements_inferred = 0
        sta = time.time()
        with torch.no_grad():
            for x, _ in tqdm(self.testset):
                st = time.time()
                x = x.to(self.device)
                output = self.net(x)
                _, predicted = torch.max(output.data, 1)
                counts = torch.bincount(predicted, minlength=len(self.unique_class_labels))
                for i, count in enumerate(counts):
                    self.class_stats[i] += count.item()
                if len(inf_times) > 50:
                    inf_times.pop(0)
                inf_times.append(time.time() - st)
                num_of_elements_inferred += self.batch_size
                if self.updated_NAS is True:
                    return
        self.class_stats[-1] = time.time() - sta
        self.client_socket.sendall(pickle.dumps(self.class_stats))
        if flag_restart:
            return 1

    def create_layers_lst(self, conv_number, pool_number, lstm_number, gru_number, rnn_number, dense_number):
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
        torch.cuda.empty_cache()
        gc.collect()

    def connect_socket(self):
        """
        Handle the connection between the prediction client and the prediction server
        """
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client_socket.connect((config["host_predictors"], config["prediction_server_port"]))
        while True:
            print('Waiting for prediction Server signal')
            data = self.client_socket.recv(128)
            print(data.decode())
            if data.decode() == 'start':
                break
        Thread(target=self.receive_weights_and_NAS, args=(self.client_socket,), daemon=True).start()


    def receive_weights_and_NAS(self, conn):
        """
        Handle socket communication and update Trainer object values

        :param conn: received from connect_socket()
        """
        while True:
            length_bytes = conn.recv(8)
            total_length = int.from_bytes(length_bytes, 'big')
            buffer = b''
            while len(buffer) < total_length:
                packet = conn.recv(262144)
                if not packet:
                    break
                buffer += packet
            if buffer[0:3] == b'nas':
                buffer = buffer[3:]
                layers = pickle.loads(buffer)
                self.num_of_conv_layers, self.num_of_pool_layers, self.num_of_dense_layers, self.num_of_lstm_layers, self.num_of_gru_layers, self.num_of_rnn_layers = layers
                self.class_stats = [0] * (len(self.unique_class_labels) + 1)
                self.updated_NAS = True
                print("Updated NAS Model Received")
                pass
            elif buffer[0:3] == b'wei':
                buffer = buffer[3:]
                self.state_dict = pickle.loads(buffer)
                self.new_weights_flag = True
                print("Updated Model Weights Received")


if __name__ == "__main__":
    import warnings
    multiprocessing.set_start_method('spawn', force=True)
    rank = int(sys.argv[1])
    warnings.filterwarnings("ignore")
    try:
        with open('config_video.json') as json_file:
            config = json.load(json_file)
    except:
        print("config_video.json not found")
        exit()
    dataset_shape_torch = config['dataset_shape_torch']
    sequence_length = config['sequence_length']
    unique_class_labels = range(len(config['classes_list']))
    tr = Trainer(True, config['num_of_conv_layers'], config['num_of_pool_layers'], config['num_of_dense_layers'],
                 config['num_of_lstm_layers'], config['num_of_gru_layers'], config['num_of_rnn_layers'],
                 unique_class_labels, sequence_length, config['predictor_batch_size'], rank)
    tr.connect_socket()
    while True:
        st = time.time()
        tr.__init__(False, tr.num_of_conv_layers, tr.num_of_pool_layers, tr.num_of_dense_layers,
                    tr.num_of_lstm_layers, tr.num_of_gru_layers, tr.num_of_rnn_layers, tr.unique_class_labels,
                    tr.sequence_length, tr.batch_size, tr.rank)
        try:
            net = tr.train()
        except:
            continue
        tr_time = time.time() - st
        print(tr_time)
        tr.clear_mem()
        if torch.cuda.is_available():
            print(f"CUDA memory allocated: {torch.cuda.memory_allocated() / 1e6:.2f} MB")
            print(f"CUDA memory reserved: {torch.cuda.memory_reserved() / 1e6:.2f} MB")