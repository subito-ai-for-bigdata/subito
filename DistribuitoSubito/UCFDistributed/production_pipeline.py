import gc
import json
import os
import pickle
import sys
import time
from typing import Optional
import cv2
import ray
import torch
from threading import Thread
from sphere_exploration.gm_utils import GMUtils
from torchsummary import summary
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import socket
from kafka import KafkaConsumer, TopicPartition
import dask
from dask.distributed import Client, LocalCluster
import sampling_lib

BUILD_TRAINSET = True
USE_GPU_TMP = False
CONV_PADDING = 'same'
MAX_POOL_PADDING = 'same'
CONV_NEURONS_CONST = 32
CONV_NEURONS_BOUND = 256
DENSE_NEURONS_CONST = 128
DENSE_NEURONS_BOUND = 32
UNITS_BOUND = 32
UNITS_CONST = 32
if MAX_POOL_PADDING == 'same':
    MAX_POOL_PADDING = 0
elif MAX_POOL_PADDING == 'valid':
    MAX_POOL_PADDING = 1
tmp_filter_test = None
current_offset = 0


def evaluate(model, test_loader):
    """Evaluates the accuracy of the model on a validation dataset."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Running on the GPU")
    else:
        device = torch.device("cpu")
    model = model.to(device)
    loss_function = nn.CrossEntropyLoss()
    model.eval()
    correct = 0
    total = 0
    running_loss = 0.0
    predictions = [0]*5
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            data = batch["image"].to(device)
            target = batch["label"].to(device)
            outputs = model(data)
            loss = loss_function(outputs, target)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            running_loss += loss.item()
            for label in predicted:
                predictions[label] += 1
    print(predictions)
    return 100.0*correct/total, running_loss/total


def reshape_for_pytorch(res_images):
    """
    Returns res images reshaped according to dataset_shape

    :param res_images: images vector input
    :return: properly reshapes res_images
    """
    dataset_shape_for_decoding = [IMAGE_WIDTH, IMAGE_HEIGHT, 3]
    received_images_reshaped = []
    for i in range(0, len(res_images)):
        r = res_images[i].reshape(dataset_shape_for_decoding)
        received_images_reshaped.append(r)
    received_images = []
    for i in range(0, len(received_images_reshaped)):
        r = received_images_reshaped[i].transpose((2, 0, 1))
        received_images.append(r)
    return received_images


def flatten(w, k=3, s=1, p=0, m=True):
    """
    Returns the right size of the flattened tensor after convolutional transformation. We need a way to calculate
    the number of neurons on the output of convolutional layers, sourced online

    :param w: width of image
    :param k: kernel size
    :param s: stride
    :param p: padding
    :param m: max pooling (bool)
    :return: proper shape and params: use x * x * previous_out_channels
    """
    return int((np.floor((w - k + 2 * p) / s) + 1) / 2 if m else 1), k, s, p, m


@ray.remote
class ShutdownController:
    def __init__(self):
        self.shutdown_flag = False

    def should_shutdown(self):
        return self.shutdown_flag

    def trigger_shutdown(self):
        self.shutdown_flag = True

    def set_shutdown_flag(self, flag):
        self.shutdown_flag = flag


class TimeDistributed(nn.Module):

    def __init__(self, module, batch_first=False):
        super(TimeDistributed, self).__init__()
        self.module = module.to(torch.device('cuda' if USE_GPU_TMP else 'cpu'))
        self.batch_first = batch_first

    def forward(self, x):
        ''' x size: (batch_size, time_steps, in_channels, height, width) '''
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

    def free_module(self):
        del self.module


# Define the structure of the Neural Network
class Net(nn.Module):
    FLAT_SHAPE_SIZE = -1

    def __init__(self, layers_lst, layer2add, dataset_shape, CONV_NEURONS_CONST, UNITS_CONST, DENSE_NEURONS_CONST,
                 CONV_NEURONS_BOUND,
                 UNITS_BOUND, DENSE_NEURONS_BOUND):
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
        # x = x.to(torch.device('cuda'))
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

    @staticmethod
    def create_layers_lst(conv_number, pool_number, lstm_number, gru_number, rnn_number, dense_number):
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
        print(layers_lst)
        return layers_lst

    def get_gradients(self):
        grads = []
        for p in self.parameters():
            grad = None if p.grad is None else p.grad.data.cpu().numpy()
            grads.append(grad)
        return grads

    def set_gradients(self, gradients, device):
        for g, p in zip(gradients, self.parameters()):
            if g is not None:
                p.grad = torch.from_numpy(g).to(device)

    def get_weights(self):
        return {k: v.clone().detach().cpu() for k, v in self.state_dict().items()}

    def set_weights(self, weights):
        self.load_state_dict(weights)


@ray.remote(num_gpus=0.5)
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
    rebuild_data = None
    net = None
    _ray_dataset = None
    train_iter = None
    old_grads = None
    server_estimate = None
    last_transmitted_v_i = None

    def __init__(self, i):
        self.rank = i

    def load_actor(self, rebuild_data, epochs, sample_size, batch_size, lr, num_of_conv_layers, num_of_pool_layers,
                   num_of_dense_layers, num_of_lstm_layers, num_of_gru_layers, num_of_rnn_layers, sequence_length, n_workers, sample_size_slack, unique_class_labels, controller):
        """
        :param rebuild_data: if False we only resample the train set
        :param epochs: nmber of epochs
        :param sample_size: sample_size
        :param batch_size: batch size of the neural network
        :param lr: learning rate
        :param num_of_conv_layers: number of convolutional layers
        :param num_of_pool_layers: number of pooling layers
        :param num_of_dense_layers: number of dense layers
        :param num_of_lstm_layers: number of lstm layers
        :param num_of_gru_layers: number of gru layers
        :param num_of_rnn_layers: number of rnn layers
        :param sequence_length: sequence length
        :param n_workers: number of workers
        :param sample_size_slack: slack
        :param unique_class_labels: unique class labels
        :param controller: controller object for shutdown
        """
        print(self.rank, ' started initializing')
        self.rebuild_data = rebuild_data
        self.epochs = epochs
        self.sample_size = sample_size
        self.old_sample_size = self.sample_size
        self.sequence_length = sequence_length
        self.num_of_conv_layers = num_of_conv_layers
        self.num_of_pool_layers = num_of_pool_layers
        self.num_of_dense_layers = num_of_dense_layers
        self.num_of_lstm_layers = num_of_lstm_layers
        self.num_of_gru_layers = num_of_gru_layers
        self.num_of_rnn_layers = num_of_rnn_layers
        self.batch_size = batch_size
        self.n_workers = n_workers
        self.lr = lr
        self.sample_size_slack = sample_size_slack
        self.controller = controller
        self.need_to_stop_round = False
        self.gm_threshold = -1
        # self.gm_threshold_base = (self.batch_size/64) * self.epochs * 2
        self.gm_threshold_base = 4
        self.threshold_decay = 0.4
        if rebuild_data:
            self.cluster = LocalCluster(n_workers=n_workers, threads_per_worker=thread_per_worker)
            self.client = Client(self.cluster)
            print(f"Dashboard link: {self.client.dashboard_link}")
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
                print("Running on the GPU")
            else:
                self.device = torch.device("cpu")
                print("Running on the CPU")
            if config['USE_KAFKA']:
                consumer_images = KafkaConsumer("train-topic", group_id='group_learner_'+self.rank.__str__(), bootstrap_servers=['127.0.0.1:9092'],
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
                        # print(f"End offset is: {end_offset}")
                        # print ("%s:%d:%d: key=%s value=%s" % (message.topic, message.partition, message.offset, message.key, message.value))
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
                received_images = received_images[self.rank::2]
                received_labels = received_labels[self.rank::2]
                print(f"Receiven Images: {len(received_images)}")
                print(f"Received Labels: {len(received_labels)}")
                self.received_images_reshaped = reshape_for_pytorch(received_images)
                self.received_labels_decoded = []
                for i in range(0, len(received_labels)):
                    l = int(received_labels[i].decode("utf-8"))
                    self.received_labels_decoded.append(l)
                chunks = np.array_split(self.received_images_reshaped,
                                        len(self.received_images_reshaped) // self.sequence_length)
                self.received_images_reshaped = np.array(chunks)
                self.received_images_reshaped = [np.float32(img / 255.0) for img in self.received_images_reshaped]

                tmp_lst = []
                for i in range(0, len(self.received_labels_decoded), self.sequence_length):
                    tmp_lst.append(self.received_labels_decoded[i])
                self.received_labels_decoded = np.array(tmp_lst)
                print(self.received_labels_decoded.shape)
                # print("Loading data from local disk (UCF50)...")
                # received_images = []
                # received_labels = []
                #
                # with open("config_video.json", "r") as f:
                #     config_json = json.load(f)
                # classes_list = config_json["classes_list"]
                # class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes_list)}
                #
                # NUM_FRAMES_PER_VIDEO = 20
                #
                # for class_name in classes_list:
                #     class_path = os.path.join("UCF50", class_name)
                #     label = class_to_idx[class_name]
                #
                #     for video_file in os.listdir(class_path):
                #         video_path = os.path.join(class_path, video_file)
                #         cap = cv2.VideoCapture(video_path)
                #
                #         total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                #         if total_frames < NUM_FRAMES_PER_VIDEO:
                #             cap.release()
                #             continue  # Skip videos that are too short
                #
                #         # Calculate frame indices to sample uniformly
                #         frame_indices = np.linspace(0, total_frames - 1, NUM_FRAMES_PER_VIDEO, dtype=int)
                #         sampled_frames = []
                #         frame_id = 0
                #         idx = 0
                #
                #         while cap.isOpened() and idx < len(frame_indices):
                #             ret, frame = cap.read()
                #             if not ret:
                #                 break
                #             if frame_id == frame_indices[idx]:
                #                 frame = cv2.resize(frame, (64, 64))
                #                 frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                #                 encoded_frame = frame.flatten()
                #                 sampled_frames.append(encoded_frame)
                #                 idx += 1
                #             frame_id += 1
                #
                #         cap.release()
                #
                #         if len(sampled_frames) == NUM_FRAMES_PER_VIDEO:
                #             received_images.extend(sampled_frames)
                #             received_labels.extend([label] * NUM_FRAMES_PER_VIDEO)
                #
                # print(f"Receiven Images: {len(received_images)}")
                # print(f"Received Labels: {len(received_labels)}")
                #
                # # === Match structure of Kafka version ===
                # self.received_images_reshaped = reshape_for_pytorch(received_images)
                #
                # self.received_labels_decoded = received_labels
                #
                # chunks = np.array_split(self.received_images_reshaped,
                #                         len(self.received_images_reshaped) // self.sequence_length)
                # self.received_images_reshaped = np.array(chunks)
                # self.received_images_reshaped = [np.float32(img / 255.0) for img in self.received_images_reshaped]
                #
                # tmp_lst = []
                # for i in range(0, len(self.received_labels_decoded), self.sequence_length):
                #     tmp_lst.append(self.received_labels_decoded[i])
                # self.received_labels_decoded = np.array(tmp_lst)
                # print(self.received_labels_decoded.shape)
        else:
            self.cluster.scale(n_workers)
        layers_lst = Net.create_layers_lst(self.num_of_conv_layers, self.num_of_pool_layers,
                                            self.num_of_lstm_layers,
                                            self.num_of_gru_layers, self.num_of_rnn_layers,
                                            self.num_of_dense_layers)
        if hasattr(self, 'net') and self.net is not None:
            del self.net
        gc.collect()
        self.net = Net(layers_lst, 'dense', dataset_shape_torch, CONV_NEURONS_CONST, UNITS_CONST,
                       DENSE_NEURONS_CONST,
                       CONV_NEURONS_BOUND, UNITS_BOUND, DENSE_NEURONS_BOUND)
        self.net = self.net.to(self.device)
        for i, layer in enumerate(self.net.layers):
            if layer != 'Forget Sequence' and layer != 'Flatten':
                self.net.layers[i] = layer.to(self.device)
        # summary(self.net, (self.batch_size, *tuple(dataset_shape_torch)))
        # self.net.half()
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
        summary(self.net, input_size=tuple(dataset_shape_torch), batch_size=self.batch_size)
        # Sample or ReSample the input
        sa_start = time.time()
        train_images, train_labels = sampling_lib.sampling_method(sampling_method_id, self.received_images_reshaped,
                                                                  self.received_labels_decoded, self.sample_size,
                                                                  self.n_workers, self.cluster)
        train_images = np.array(train_images)
        train_labels = np.array(train_labels, dtype=np.long)
        train_labels = train_labels.reshape((len(train_labels)))
        print(train_images.shape, train_labels.shape)
        data = [{"image": img, "label": int(lbl)} for img, lbl in zip(train_images, train_labels)]
        train_dataset = ray.data.from_items(data)
        self._ray_dataset = train_dataset
        self.trainset = self._ray_dataset.iter_torch_batches(batch_size=self.batch_size)
        self.train_iter = iter(self.trainset)
        del train_images
        del train_labels
        self.sa_end = time.time() - sa_start
        print(self.rank, ' finished initializing')
        return 0

    def play_video_from_array(self, video_array):
        for idx, frame in enumerate(video_array):
            # Convert (3, 64, 64) → (64, 64, 3)
            frame = np.transpose(frame, (1, 2, 0))  # CHW → HWC
            # If frame is float, convert to uint8
            if frame.dtype != np.uint8:
                frame = np.clip(frame * 255, 0, 255).astype(np.uint8)
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imshow('Video Playback', frame_bgr)
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()

    def train(self, weights, round):
        """
        Train the Neural Network with the trainset
        """

        layers_lst = Net.create_layers_lst(self.num_of_conv_layers, self.num_of_pool_layers, self.num_of_lstm_layers,
                                            self.num_of_gru_layers, self.num_of_rnn_layers, self.num_of_dense_layers)
        self.net = Net(layers_lst, 'dense', dataset_shape_torch, CONV_NEURONS_CONST, UNITS_CONST, DENSE_NEURONS_CONST,
                       CONV_NEURONS_BOUND, UNITS_BOUND, DENSE_NEURONS_BOUND)
        # self.net.half()
        self.net.to(self.device)
        for i, layer in enumerate(self.net.layers):
            if layer != 'Forget Sequence' and layer != 'Flatten':
                self.net.layers[i] = layer.to(self.device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
        # summary(self.net, input_size=tuple(dataset_shape_torch), batch_size=self.batch_size)
        # f: y = 4 e^(-0.4x) + 0.05
        self.gm_threshold = self.gm_threshold_base * np.exp(-self.threshold_decay * (round-1)) + 0.15
        try:
            self.net.load_state_dict(weights)
        except:
            print("Cant load state dict")
            return
        self.server_estimate = GMUtils.compute_local_vector(list(weights.values()))
        server_estimate_fft = GMUtils._fft(self.server_estimate, 10)
        loss_function = nn.CrossEntropyLoss()
        self.net.train()
        self.optimizer.zero_grad()
        self.need_to_stop_round = False
        total, correct, running_loss, steps = 1, 1, 0, 0
        while True:
            steps += 1
            # if steps > len(self.trainset) / self.batch_size:
            #     self.need_to_stop_round = True
            try:
                batch = next(self.train_iter)
            except StopIteration:  # When the epoch ends, start a new epoch.
                self.train_iter = iter(self.trainset)
                # self.controller.trigger_shutdown.remote()
                return self.net.get_weights(), steps, 100.0*correct/total, running_loss/total
                # batch = next(self.train_iter)
            self.optimizer.zero_grad()
            X = batch["image"].to(self.device)
            y = batch["label"].to(self.device)

            output = self.net(X)
            loss = loss_function(output, y)
            predicted = output.argmax(dim=1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
            running_loss += loss.item()
            # print(loss)
            loss.backward()
            self.optimizer.step()
            # current_weight_grads = [p.clone().detach().cpu() for p in self.net.parameters()]
            current_weight_grads = [v.clone().detach().cpu() for v in self.net.state_dict().values()]
            self.transform_weights(round, current_weight_grads, server_estimate_fft)
            # self.need_to_stop_round = True
            if steps % 100 == 0 and self.rank == 0:
                print("Reached ", steps, " steps")
            if self.need_to_stop_round:  # check if we should stop other workers and sync
                print("Worker ", self.rank, " decides to shut down all on round ", round, ". Run:", steps, " steps")
                self.controller.trigger_shutdown.remote()
                return self.net.get_weights(), steps, 100.0*correct/total, running_loss/total
            if ray.get(self.controller.should_shutdown.remote()):
                print(f"Worker {self.rank} received shutdown signal on round ", round, ". Run:", steps, ".Exiting...")
                return self.net.get_weights(), steps, 100.0*correct/total, running_loss/total

    def transform_weights(self, cur_round: int, initial_grads: list[Optional[torch.Tensor]], server_estimate_fft):
        v_i: torch.Tensor = GMUtils.compute_local_vector(initial_grads)
        delta_v_i: torch.Tensor = GMUtils.compute_delta_local_vectors(v_i, self.server_estimate)
        u_i_fft = GMUtils._fft(delta_v_i, 10)
        center: torch.Tensor = GMUtils.compute_center(server_estimate_fft, u_i_fft, self.lr)
        radius = GMUtils.compute_radius(server_estimate_fft, u_i_fft, self.lr,
                                        lambda x: torch.norm(x, p=2))  # Check if we need to stop the current round
        local_violation = cur_round == 1 or self.check_threshold_crossing(server_estimate_fft, center, radius,
                                                                          lambda x: torch.norm(x, p=2))
        if local_violation:
            self.need_to_stop_round = True
        return

    def check_threshold_crossing(self, server_estimate_fft: torch.Tensor, center: torch.Tensor, radius: torch.Tensor,
                                 myfunc: callable) -> torch.Tensor:
        dim = center.numel()
        points = 3
        # Generate a grid of points in each dimension from -radius to +radius
        linspaces = [torch.linspace(-radius, radius, points) for _ in range(dim)]
        grid = torch.cartesian_prod(*linspaces)  # shape: (points**dim, dim)
        # Shift grid to be centered around `center`
        coords = grid + center
        # Filter points inside the sphere
        dists = torch.sum((coords - center) ** 2, dim=1)
        inside_sphere = dists <= radius ** 2
        coords = coords[inside_sphere]
        # Evaluate function at each coordinate
        values = torch.tensor([myfunc(pt - server_estimate_fft) for pt in coords])
        tmp_max = values.max()
        tmp_min = values.min()
        # print(f"Max={round(tmp_max.item(), 1)} Min={round(tmp_min.item(), 1)} T={round(self.gm_threshold, 1)}")
        threshold_crossed = (tmp_max > self.gm_threshold)
        return threshold_crossed

    def get_data_loader(self):
        return self._ray_dataset

    def clear_mem(self):
        for layer in self.net.layers:
            del layer
        del self.net  # Delete the model
        del self.optimizer  # Delete the optimizer
        torch.cuda.empty_cache()
        gc.collect()

    def close_dask_cluster(self):
        self.client.close()
        self.cluster.close()


class ParameterServer(object):
    server_socket = None
    rebuild_data = None
    live_socket = None
    prediction_socket = None
    prediction_socket_nas = None
    trainers = None

    def __init__(self, num_trainers, num_of_conv_layers, num_of_pool_layers, num_of_dense_layers, num_of_lstm_layers, num_of_gru_layers, num_of_rnn_layers, sequence_length, n_workers, epochs,
                 sample_size, lr, batch_size):
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            print("Running on the GPU")
        else:
            self.device = torch.device("cpu")
            print("Running on the CPU")
        self.num_trainers = num_trainers
        self.num_of_conv_layers = num_of_conv_layers
        self.num_of_pool_layers = num_of_pool_layers
        self.num_of_dense_layers = num_of_dense_layers
        self.num_of_lstm_layers = num_of_lstm_layers
        self.num_of_gru_layers = num_of_gru_layers
        self.num_of_rnn_layers = num_of_rnn_layers
        self.sequence_length = sequence_length
        self.nas_changed = False
        self.n_workers = n_workers
        self.epochs = epochs
        self.sample_size = sample_size
        self.old_sample_size = sample_size
        self.batch_size = batch_size
        self.lr = lr
        # self.net.half()
        self.controller = ShutdownController.remote()

    def train(self):
        """
                Train the Neural Network with the trainset
                """
        flag_restart = False
        rounds_to_run = 1000
        layers_lst = Net.create_layers_lst(self.num_of_conv_layers, self.num_of_pool_layers, self.num_of_lstm_layers,
                                            self.num_of_gru_layers, self.num_of_rnn_layers, self.num_of_dense_layers)
        self.net = Net(layers_lst, 'dense', dataset_shape_torch, CONV_NEURONS_CONST, UNITS_CONST, DENSE_NEURONS_CONST,
                       CONV_NEURONS_BOUND, UNITS_BOUND, DENSE_NEURONS_BOUND)
        self.net.to(self.device)
        for i, layer in enumerate(self.net.layers):
            if layer != 'Forget Sequence' and layer != 'Flatten':
                self.net.layers[i] = layer.to(self.device)
        avg_weights = self.get_weights()
        self.net.train()
        self.old_sample_size = self.sample_size
        round = 1
        steps_total = 0
        time_total = 0
        curr_epoch_time = 0
        self.print_training_setup()
        next_epoch_target_steps = self.sample_size * 1002 / self.batch_size
        epochs_completed = 0
        self.nas_changed = False
        fed_opt = FederatedOptimizer()

        while round < rounds_to_run:
            start = time.time()
            print(f"Round: {round} starting. Steps {steps_total}")
            if self.nas_changed:
                print("NAS changed--Loading new net to Actors")
                return
            while True:
                weights_futures = [trainer.train.remote(avg_weights, round) for trainer in self.trainers]
                actor_rets = ray.get(weights_futures)
                worker_weights, steps_completed, train_accuracies, train_losses = [], [], [], []
                try:
                    for i, _ in enumerate(self.trainers):
                        worker_weights.append(actor_rets[i][0])
                        steps_completed.append(actor_rets[i][1])
                        train_accuracies.append(actor_rets[i][2])
                        train_losses.append(actor_rets[i][3])
                        steps_total += actor_rets[i][1]
                    break
                except:
                    return
            self.controller.set_shutdown_flag.remote(False)
            train_loss = np.mean(train_losses)
            train_accuracy = np.mean(train_accuracies)
            avg_weights = self.average_weights(worker_weights)
            # avg_weights = fed_opt.fed_adam(avg_weights, worker_weights)
            round += 1
            self.net.set_weights(avg_weights)
            curr_epoch_time += time.time() - start
            time_total += time.time() - start

            # ray_dataset = ray.get(self.trainers[0].get_data_loader.remote())
            # train_loader = ray_dataset.iter_torch_batches(batch_size=self.batch_size)
            # train_accuracy, train_loss = evaluate(self.net, train_loader)
            if steps_total > next_epoch_target_steps:
                epochs_completed += 1
                next_epoch_target_steps = (epochs_completed+1) * self.sample_size * 1002 / self.batch_size
                epoch_duration_estimation = time_total + (self.epochs-epochs_completed) * (time_total / epochs_completed)
                serialized_df = pickle.dumps([train_accuracy, train_loss, curr_epoch_time, epoch_duration_estimation])
                print("Accuracy:", train_accuracy)
                self.live_socket.sendall(serialized_df)
                curr_epoch_time = 0
            if self.nas_changed or self.old_sample_size != self.sample_size:
                if self.nas_changed:
                    print("NAS changed--Loading new net to Actors")
                    return
                else:
                    print("Loading new data to Actors")
                    load_res = [trainer.load_actor.remote(False, self.epochs, self.sample_size, self.batch_size,
                                                          self.lr, self.num_of_conv_layers, self.num_of_pool_layers,
                                                          self.num_of_dense_layers, self.num_of_lstm_layers,
                                                          self.num_of_gru_layers, self.num_of_rnn_layers,
                                                          self.sequence_length, self.n_workers,
                                                          config['sample_rate_slack'], unique_class_labels,
                                                          self.controller)
                                for trainer in self.trainers]
                    res = ray.get(load_res)
                    self.old_sample_size = self.sample_size
                    if res != [0, 0]:
                        print("Loading to Actors failed")
                    self.print_training_setup()
                    print("Loaded new data to Actors")
            if steps_total > self.epochs * self.sample_size * 1002 / self.batch_size:
                print("Training time: ", time_total)
                return

    def average_weights(self, weight_list, weights=None):
        avg = {}
        num_models = len(weight_list)
        if weights is None:
            weights = [1.0] * num_models
        total_weight = sum(weights)
        for k in weight_list[0].keys():
            avg[k] = sum(w * weight_list[i][k] for i, w in enumerate(weights)) / total_weight
        return avg

    def get_weights(self):
        return self.net.get_weights()

    def set_trainers(self, trainers):
        self.trainers = trainers

    def print_training_setup(self):
        print('EPOCHS:\t\t', self.epochs)
        print('SAMPLE_SIZE:\t', self.sample_size * 100, '%')
        print('NUM_OF_CONV:\t', self.num_of_conv_layers)
        print('NUM_OF_POOL:\t', self.num_of_pool_layers)
        print('NUM_OF_DENSE:\t', self.num_of_dense_layers)
        print('NUM_OF_CONV:\t', self.num_of_lstm_layers)
        print('NUM_OF_POOL:\t', self.num_of_gru_layers)
        print('NUM_OF_DENSE:\t', self.num_of_rnn_layers)
        print('LR:\t', self.lr)
        print('BATCH_SIZE:\t', self.batch_size)

    def socket_listener(self, conn):
        """
        Handle socket communication and update Trainer object values

        :param conn: received from connect_socket()
        """
        while True:
            # Receive data from the socket
            try:
                data = conn.recv(1024).decode()
                if not data:
                    break
                try:
                    # Attempt to update the epochs and sampling rate
                    data = data.split(',')
                    self.epochs = int(float(data[0]))
                    self.sample_size = float(data[1])
                    if int(float(data[2])) != -1:
                        changed = False
                        if int(float(data[2])) != self.num_of_conv_layers:
                            self.num_of_conv_layers = int(float(data[2]))
                            changed = True
                        if int(float(data[3])) != self.num_of_pool_layers:
                            self.num_of_pool_layers = int(float(data[3]))
                            changed = True
                        if int(float(data[4])) != self.num_of_dense_layers:
                            self.num_of_dense_layers = int(float(data[4]))
                            changed = True
                        if int(float(data[5])) != self.num_of_lstm_layers:
                            self.num_of_lstm_layers = int(float(data[5]))
                            changed = True
                        if int(float(data[6])) != self.num_of_gru_layers:
                            self.num_of_gru_layers = int(float(data[6]))
                            changed = True
                        if int(float(data[7])) != self.num_of_rnn_layers:
                            self.num_of_rnn_layers = int(float(data[7]))
                            changed = True
                        if changed:
                            self.nas_changed = True
                    # try:
                    #     self.lr = float(data[5])
                    #     self.batch_size = int(data[6])
                    # except:
                    #     pass
                    print(
                        f"Updated EPOCHS to: {self.epochs} \n SAMPLE RATE to: {self.sample_size} \n NUM_OF_CONV_LAYERS to: {self.num_of_conv_layers} \n NUM_OF_POOL_LAYERS to: {self.num_of_pool_layers} \n NUM_OF_DENSE_LAYERS to: {self.num_of_dense_layers} \n NUM_OF_LSTM_LAYERS to: {self.num_of_lstm_layers} \n NUM_OF_GRU_LAYERS to: {self.num_of_gru_layers} \n NUM_OF_RNN_LAYERS to: {self.num_of_rnn_layers} \n N_WORKERS to: {self.n_workers} \n LR to: {self.lr} \n BATCH_SIZE to: {self.batch_size}")
                except ValueError:
                    print("Invalid input. Please enter a valid integer for the new number of epochs.")
            except:
                self.disconnect_socket(conn)
                conn, addr = self.server_socket.accept()

    def connect_socket(self):
        """
        Initialize a server socket as a new thread and wait for connections

        :return: [socket, listening_thread] instances
        """
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind((config['host_address'], config['production_port']))
        self.server_socket.listen(1)  # Allow 1 failed connection
        print("Waiting for synopsis based training optimizer connection...")
        conn, addr = self.server_socket.accept()
        print("Connected by", addr)
        # Spawn and start a thread to listen for new data
        listen_thread = Thread(target=self.socket_listener, args=(conn,))
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
        self.live_socket.connect((config["host_address"], config['production_live_port']))
        self.prediction_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.prediction_socket.connect((config["host_address"], config['prediction_port']))
        self.prediction_socket_nas = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.prediction_socket_nas.connect((config["host_address"], config['prediction_nas_port']))
        return


class FederatedOptimizer:
    def __init__(self, beta1=0.9, beta2=0.999, epsilon=1e-8, lr=0.001):
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.lr = lr
        self.m = None
        self.v = None
        self.t = 0

    def fed_adam(self, global_weights, weight_list, weights=None):
        num_models = len(weight_list)
        if weights is None:
            weights = [1.0] * num_models
        total_weight = sum(weights)

        # Compute weighted average of client updates
        avg_update = {}
        for k in global_weights.keys():
            avg_update[k] = sum(
                weights[i] * (weight_list[i][k] - global_weights[k])
                for i in range(num_models)
            ) / total_weight

        # Initialize m and v
        if self.m is None or self.v is None:
            self.m = {k: 0.0 for k in global_weights}
            self.v = {k: 0.0 for k in global_weights}

        # Time step
        self.t += 1

        # Update m and v
        for k in global_weights.keys():
            self.m[k] = self.beta1 * self.m[k] + (1 - self.beta1) * avg_update[k]
            self.v[k] = self.beta2 * self.v[k] + (1 - self.beta2) * (avg_update[k] ** 2)

            # Bias correction
            m_hat = self.m[k] / (1 - self.beta1 ** self.t)
            v_hat = self.v[k] / (1 - self.beta2 ** self.t)

            # Update global weights using Adam rule
            global_weights[k] += self.lr * m_hat / (v_hat ** 0.5 + self.epsilon)

        return global_weights


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    torch.cuda.set_per_process_memory_fraction(0.5, 0)
    try:
        with open('config_video.json') as json_file:
            config = json.load(json_file)
    except:
        print("config_video.json not found")
        exit()
    thread_per_worker = 1
    dask.config.set(scheduler='threads', num_of_workers=config['n_workers'], threads_per_worker=thread_per_worker)
    # cluster = LocalCluster(n_workers=config['n_workers'], threads_per_worker=thread_per_worker,
    #                        dashboard_address=':8887')
    # client = Client(cluster)
    # print(f"Dashboard link: {client.dashboard_link}")
    sampling_method_id = 2
    tmp_filter_test = config['stream_batch_test']
    unique_class_labels = range(len(config['classes_list']))
    current_offset = 0
    num_trainers = config['num_of_trainers']
    dataset_shape_torch = config['dataset_shape_torch']
    IMAGE_HEIGHT = dataset_shape_torch[3]
    IMAGE_WIDTH = dataset_shape_torch[2]
    SEQUENCE_LENGTH = dataset_shape_torch[0]
    context = ray.init(ignore_reinit_error=True)
    print(context.dashboard_url)
    ps = ParameterServer(num_trainers, config['num_of_conv_layers'],
                         config['num_of_pool_layers'], config['num_of_dense_layers'], config['num_of_lstm_layers'],
                         config['num_of_gru_layers'], config['num_of_rnn_layers'], config['sequence_length'],
                         config['n_workers'], config['initial_epochs'], config['initial_sampling_rate'], config["lr"],
                         config['size_of_batch'])

    trainers = [Trainer.remote(i) for i in range(num_trainers)]
    load_res = [trainer.load_actor.remote(True, ps.epochs, ps.sample_size, ps.batch_size,
                                          ps.lr, ps.num_of_conv_layers, ps.num_of_pool_layers, ps.num_of_dense_layers,
                                          ps.num_of_lstm_layers, ps.num_of_gru_layers, ps.num_of_rnn_layers,
                                          ps.sequence_length, ps.n_workers, config['sample_rate_slack'],
                                          unique_class_labels, ps.controller)
                for trainer in trainers]
    res = ray.get(load_res)
    conn, listen_thread = ps.connect_socket()
    ps.start_controller()
    ps.set_trainers(trainers)
    nas_to_change = False
    ps.train()
    while True:
        if nas_to_change:
            packet = pickle.dumps([ps.num_of_conv_layers, ps.num_of_pool_layers, ps.num_of_dense_layers,
                                   ps.num_of_lstm_layers, ps.num_of_gru_layers, ps.num_of_rnn_layers])
            ps.prediction_socket_nas.sendall(len(packet).to_bytes(8, 'big'))
            ps.prediction_socket_nas.sendall(packet)
            nas_to_change = False
        if ps.nas_changed:
            nas_to_change = True
        else:
            packet = pickle.dumps(ps.net.state_dict())
            print('Sending', len(packet).to_bytes(8, 'big'), ' bytes of weights')
            ps.prediction_socket.sendall(len(packet).to_bytes(8, 'big'))  # send size first
            ps.prediction_socket.sendall(packet)
        load_res = [trainer.load_actor.remote(False, ps.epochs, ps.sample_size, ps.batch_size,
                                              ps.lr, ps.num_of_conv_layers, ps.num_of_pool_layers,
                                              ps.num_of_dense_layers, ps.num_of_lstm_layers, ps.num_of_gru_layers,
                                              ps.num_of_rnn_layers,  ps.sequence_length, ps.n_workers,
                                              config['sample_rate_slack'], unique_class_labels,
                                              ps.controller)
                    for trainer in trainers]
        res = ray.get(load_res)
        ps.train()
        if torch.cuda.is_available():
            print(f"CUDA memory allocated: {torch.cuda.memory_allocated() / 1e6:.2f} MB")
            print(f"CUDA memory reserved: {torch.cuda.memory_reserved() / 1e6:.2f} MB")
