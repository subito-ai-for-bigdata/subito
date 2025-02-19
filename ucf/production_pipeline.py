import gc
import json
import pickle
import sys
import time
import numpy
import torch
from threading import Thread
from torch.nn import ModuleList
from torch.utils.data import SubsetRandomSampler, TensorDataset
from torchinfo import summary
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import socket
from tqdm import tqdm
from kafka import KafkaConsumer, TopicPartition
import sampling_lib

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


def print_model_summary(model, input_size, batch_size):
    """
    Prints a summary of the model including input shape, output shape, and number of parameters for each layer.

    :param model: The PyTorch model to summarize.
    :param input_size: A tuple representing the shape of a single input sample (excluding batch size).
    :param batch_size: The batch size to use for inference during summary calculation.
    """
    def register_hook(module):
        """
        Registers a forward hook for each layer in the model to capture input/output shapes and parameter count.

        :param module: The layer/module of the model to register the hook on.
        """
        def hook(module, input, output):
            """
            A hook function that records input shape, output shape, and parameter count for the module.

            :param module: The layer/module being hooked.
            :param input: The input to the layer.
            :param output: The output of the layer.
            """
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)
            m_key = f"{class_name}-{module_idx + 1}"
            summary[m_key] = {
                "input_shape": list(input[0].size()),
                "output_shape": list(output.size()),
                "nb_params": sum(p.numel() for p in module.parameters())
            }

        if not isinstance(module, nn.Sequential) and not isinstance(module, nn.ModuleList) and module != model:
            hooks.append(module.register_forward_hook(hook))

    summary = {}
    hooks = []
    model.apply(register_hook)
    # Dummy forward pass to trigger the hooks
    with torch.no_grad():
        tmp_data = torch.zeros(batch_size, *input_size)
        tmp_data = tmp_data.to(torch.device('cuda'))
        model(tmp_data)

    # Remove hooks after gathering required information
    for h in hooks:
        h.remove()

    # Print the summary
    print("----------------------------------------------------------------")
    line_new = "{:>20}  {:>25} {:>15}".format("Layer (type)", "Output Shape", "Param #")
    print(line_new)
    print("================================================================")
    total_params = 0
    for layer in summary:
        line_new = "{:>20}  {:>25} {:>15}".format(
            layer,
            str(summary[layer]["output_shape"]),
            "{0:,}".format(summary[layer]["nb_params"])
        )
        total_params += summary[layer]["nb_params"]
        print(line_new)
    print("================================================================")
    print(f"Total params: {total_params:,}")
    print("----------------------------------------------------------------")


# Example usage

def reshape_for_pytorch(res_images, dataset_shape):
    """
    Reshapes and transposes the input images to match the expected input format for PyTorch models.

    :param res_images: A list or array of images to reshape.
    :param dataset_shape: The shape to reshape each image to.
    :return: A list of reshaped and transposed images in the format (C, H, W).
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
    Calculates the appropriate size of the flattened tensor after a convolutional operation.

    :param w: The width (or height) of the input image.
    :param k: The kernel size of the convolution.
    :param s: The stride of the convolution.
    :param p: The padding applied to the convolution.
    :param m: Boolean flag indicating if max pooling is applied after the convolution (default is True).
    :return: The flattened output shape as a tuple representing the size of the flattened tensor.
    """
    return int((np.floor((w - k + 2 * p) / s) + 1) / 2 if m else 1), k, s, p, m


class TimeDistributed(nn.Module):
    """
    A wrapper module that applies a given submodule (e.g., a layer or a set of layers)
    to each time step of the input independently. This class is useful for processing 
    sequential data where a module is applied to each time step in the sequence.
    """
    
    def __init__(self, module, batch_first=False):
        """
        Initializes the TimeDistributed wrapper.

        :param module: The submodule (e.g., a layer or neural network) to apply at each time step.
        :param batch_first: Boolean flag indicating whether the input tensors have the batch dimension first.
        """
        super(TimeDistributed, self).__init__()
        self.module = module.to(torch.device('cuda' if USE_GPU_TMP else 'cpu'))
        self.batch_first = batch_first

    def forward(self, x):
        """
        Forward pass through the TimeDistributed module. Applies the wrapped module to each time step
        of the input tensor.

        :param x: Input tensor of shape (batch_size, time_steps, in_channels, height, width) 
                  when `batch_first=True`, or (time_steps, batch_size, in_channels, height, width) 
                  when `batch_first=False`.
        :return: Output tensor with the module applied to each time step, with the shape
                 (batch_size, time_steps, output_channels, height, width) when `batch_first=True`,
                 or (time_steps, batch_size, output_channels, height, width) when `batch_first=False`.
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

    def free_module(self):
        """
        Frees the resources associated with the module by deleting the wrapped submodule.

        This function can be called to remove the module from memory when no longer needed.
        """
        del self.module


# Define the structure of the Neural Network
class Net(nn.Module):
    """
    A custom deep neural network architecture that allows dynamic construction
    based on a given list of layers. The network supports various layer types including
    convolutional (conv), recurrent (LSTM, GRU, RNN), dense layers, pooling, 
    and time-distributed layers. The model adapts the architecture by adjusting 
    the number of units in the layers based on user-defined bounds.
    """

    FLAT_SHAPE_SIZE = -1

    def __init__(self, layers_lst, layer2add, dataset_shape, CONV_NEURONS_CONST, UNITS_CONST, DENSE_NEURONS_CONST,
                 CONV_NEURONS_BOUND,
                 UNITS_BOUND, DENSE_NEURONS_BOUND):
        """
        Initialize the neural network model with a given configuration of layers.
        
        :param layers_lst: A list defining the layers and their order (e.g., ['conv', 'lstm', 'dense']).
        :param layer2add: Type of the additional layer to be added (e.g., 'lstm', 'dense').
        :param dataset_shape: Shape of the input dataset (e.g., (batch_size, height, width, channels)).
        :param CONV_NEURONS_CONST: Initial number of neurons for the convolutional layers.
        :param UNITS_CONST: Initial number of units for the recurrent layers.
        :param DENSE_NEURONS_CONST: Initial number of neurons for the dense layers.
        :param CONV_NEURONS_BOUND: Maximum allowed neurons for the convolutional layers.
        :param UNITS_BOUND: Maximum allowed units for the recurrent layers.
        :param DENSE_NEURONS_BOUND: Maximum allowed neurons for the dense layers.
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
        Forward pass of the network. Takes an input tensor and processes it through the layers.
        
        :param x: Input tensor (e.g., image or sequence data).
        
        :return: The output of the network after passing through all layers.
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
        Calculates the flattened dimension size for the given dataset shape. This is necessary for layers 
        like dense layers or recurrent layers that expect a 1D vector as input.
        
        :param dataset_shape: Shape of the dataset (e.g., (batch_size, height, width, channels)).
        :param called_from_batch_norm: Boolean indicating if the method was called from a batch normalization layer.
        
        :return: The flattened dimension size (integer).
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
    Class to handle training of the neural network and the updating of sample_size and epochs.
    
    This class manages the setup of training parameters, the construction of neural network layers,
    data handling (including Kafka data streams for training and testing), and resampling of datasets.
    """
    epochs, sample_size, trainset, testset, device = [None, None, None, None, None]
    received_images_reshaped = None
    received_labels_decoded = None
    old_sample_size = None
    sample_size_slack = None
    needed_shape = None
    server_socket = None
    rebuild_data = None
    net = None
    live_socket = None
    prediction_socket = None
    prediction_socket_nas = None

    def __init__(self, rebuild_data, epochs, sample_size, batch_size, lr, num_of_conv_layers, num_of_pool_layers,
                 num_of_dense_layers, num_of_lstm_layers, num_of_gru_layers, num_of_rnn_layers, sample_size_slack,
                 unique_class_labels, sequence_length):
        """
        Initializes the Trainer class with the given configuration parameters. 
        
        :param rebuild_data: Boolean flag indicating whether to rebuild the training data.
        :param epochs: Number of training epochs.
        :param sample_size: Sample size for training.
        :param batch_size: Batch size for training.
        :param lr: Learning rate for training.
        :param num_of_conv_layers: Number of convolution layers in the network.
        :param num_of_pool_layers: Number of pooling layers in the network.
        :param num_of_dense_layers: Number of dense layers in the network.
        :param num_of_lstm_layers: Number of LSTM layers in the network.
        :param num_of_gru_layers: Number of GRU layers in the network.
        :param num_of_rnn_layers: Number of RNN layers in the network.
        :param sample_size_slack: Slack value for sample size.
        :param unique_class_labels: Number of unique class labels in the dataset.
        :param sequence_length: The sequence length of input data for sequence-based models.
        """
        self.nas_changed = False
        self.rebuild_data = rebuild_data
        self.optimizer = None
        if rebuild_data:
            self.epochs = epochs
            self.sample_size = sample_size
            self.old_sample_size = self.sample_size
            self.batch_size = batch_size
            self.lr = lr
            self.sequence_length = sequence_length
            self.num_of_conv_layers = num_of_conv_layers
            self.num_of_pool_layers = num_of_pool_layers
            self.num_of_dense_layers = num_of_dense_layers
            self.num_of_lstm_layers = num_of_lstm_layers
            self.num_of_gru_layers = num_of_gru_layers
            self.num_of_rnn_layers = num_of_rnn_layers
            self.sample_size_slack = sample_size_slack
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
                print("Running on the GPU")
            else:
                self.device = torch.device("cpu")
                print("Running on the CPU")
            if config['USE_KAFKA']:
                # Consumer-part for images
                consumer_images_test = KafkaConsumer("test-topic", group_id='group4',
                                                     bootstrap_servers=['127.0.0.1:9092'], auto_offset_reset='earliest')
                try:
                    tmp_count = 0
                    received_images_test = []
                    received_labels_test = []
                    for message in consumer_images_test:
                        if message.partition == 0:
                            decode_img = np.frombuffer(message.value, dtype=np.uint8)
                            received_images_test.append(decode_img)
                            del decode_img
                        else:
                            received_labels_test.append(message.value)
                        tmp_count = tmp_count + 1
                        if tmp_count >= 2 * tmp_filter_test:
                            break
                except KeyboardInterrupt:
                    sys.exit()
                consumer_images_test.close()
                received_labels_decoded_test = []
                for i in range(0, len(received_labels_test)):
                    l = int(received_labels_test[i].decode("utf-8"))
                    received_labels_decoded_test.append(l)
                received_images_reshaped_test = reshape_for_pytorch(received_images_test, dataset_shape_torch)
                print(len(received_images_reshaped_test))
                print(len(received_labels_decoded_test))
                chunks = np.array_split(received_images_reshaped_test,
                                        len(received_images_reshaped_test) // self.sequence_length)
                received_images_reshaped_test = np.array(chunks)
                print(received_images_reshaped_test.shape)
                tmp_lst = []
                for i in range(0, len(received_labels_decoded_test), self.sequence_length):
                    tmp_lst.append(received_labels_decoded_test[i])
                received_labels_decoded_test = np.array(tmp_lst)
                print(received_labels_decoded_test.shape)
                # We can build the testset here as it will not be resampled
                test_images_tensor = torch.tensor(numpy.array(received_images_reshaped_test), dtype=torch.float32)
                test_labels_tensor = torch.tensor(numpy.array(received_labels_decoded_test), dtype=torch.long)
                test_dataset = TensorDataset(test_images_tensor, test_labels_tensor)
                self.testset = torch.utils.data.DataLoader(test_dataset)

                consumer_images = KafkaConsumer("train-topic", group_id='group3', bootstrap_servers=['127.0.0.1:9092'],
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
        layers_lst = self.create_layers_lst(self.num_of_conv_layers, self.num_of_pool_layers, self.num_of_lstm_layers,
                                            self.num_of_gru_layers, self.num_of_rnn_layers, self.num_of_dense_layers)
        self.net = Net(layers_lst, 'dense', dataset_shape_torch, CONV_NEURONS_CONST, UNITS_CONST, DENSE_NEURONS_CONST,
                       CONV_NEURONS_BOUND, UNITS_BOUND, DENSE_NEURONS_BOUND)
        self.net.calculate_flatten_dim(dataset_shape_torch)
        self.net = self.net.to(self.device)
        for i, layer in enumerate(self.net.layers):
            if layer != 'Forget Sequence' and layer != 'Flatten':
                self.net.layers[i] = layer.to(self.device)
        summary(self.net, (self.batch_size, *tuple(dataset_shape_torch)))
        if config['USE_KAFKA']:
            # Consumer-part for images

            # Sample or ReSample the input
            train_images, train_labels = sampling_lib.sampling_method(sampling_method_id, self.received_images_reshaped,
                                                                      self.received_labels_decoded, self.sample_size)
            # Build initial or updated trainset (be careful Tensors to float32 not float64)
            train_images_tensor = torch.tensor(train_images, dtype=torch.float32)
            train_labels_tensor = torch.tensor(train_labels, dtype=torch.long)
            train_dataset = TensorDataset(train_images_tensor, train_labels_tensor)
            self.trainset = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        print('EPOCHS:\t\t', self.epochs)
        print('SAMPLE_SIZE:\t', self.sample_size * 100, '%')
        print('NUM_OF_CONV:\t', self.num_of_conv_layers)
        print('NUM_OF_POOL:\t', self.num_of_pool_layers)
        print('NUM_OF_DENSE:\t', self.num_of_dense_layers)
        print('NUM_OF_LSTM:\t', self.num_of_lstm_layers)
        print('NUM_OF_GRU:\t', self.num_of_gru_layers)
        print('NUM_OF_RNN:\t', self.num_of_rnn_layers)
        print('LR:\t', self.lr)
        print('BATCH_SIZE:\t', self.batch_size)

    def train(self):
        """
        Train the Neural Network with the trainset
        """
        net = self.net
        net.to(self.device)
        loss_function = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(net.parameters(), lr=self.lr)
        epoch = 0
        flag_restart = False
        # Used for non-definite number of epochs
        epoch_duration_estimation = -1
        current_duration = 0
        epochs_to_run = self.epochs
        while epoch < epochs_to_run:
            start = time.time()
            # Initialize metrics
            running_accuracy = 0.0
            running_loss = 0.0
            net.train()
            print(f"Epoch: {epoch + 1} starting out of {self.epochs}")
            # Train the neural network
            for X, y in tqdm(self.trainset):
                X, y = X.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                # Forward pass
                output = net(X)
                # calculate the batch loss
                loss = loss_function(output, y)
                loss.backward()
                self.optimizer.step()
                # Average train loss
                _, predicted_y = torch.max(output.data, 1)
                running_accuracy += (predicted_y == y).sum().item()
                running_loss += loss.item()
            running_accuracy = running_accuracy / len(self.trainset.dataset)
            print('Training Accuracy: {:.6f}'.format(running_accuracy))
            epoch += 1
            # If sample size changed resample input
            if (config['USE_KAFKA']) and (abs(self.old_sample_size - self.sample_size) > self.sample_size_slack):
                flag_restart = True
                self.old_sample_size = self.sample_size
            flag_restart = True
            end = time.time()
            epoch_t = end - start
            train_loss = running_loss / len(self.trainset)
            current_duration += epoch_t
            remaining_epochs = self.epochs - epoch
            epoch_duration_estimation = current_duration + remaining_epochs * (current_duration / epoch)
            serialized_df = pickle.dumps([running_accuracy, train_loss, epoch_t, epoch_duration_estimation])

            self.live_socket.sendall(serialized_df)
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
        Create a list of layer types based on the specified numbers of each layer type (Convolution, Pooling, LSTM, GRU, RNN, Dense).

        This method constructs a list of strings representing the types of layers (e.g., 'conv', 'pool', 'lstm', etc.) based on the
        provided arguments. It ensures the correct combination of layers based on the given numbers of each layer type.

        :param conv_number: The number of convolution layers to add.
        :param pool_number: The number of pooling layers to add.
        :param lstm_number: The number of LSTM layers to add.
        :param gru_number: The number of GRU layers to add.
        :param rnn_number: The number of RNN layers to add.
        :param dense_number: The number of dense layers to add.
        :return: A list of strings representing the layers.
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
        print(layers_lst)
        return layers_lst

    def clear_mem(self):
        """
        Clear memory by deleting the model, optimizer, and clearing the CUDA cache.

        This method is used to delete the model's layers, the optimizer, and free up memory used by the model.
        It also clears any cached memory from CUDA and performs garbage collection.

        :return: None
        """
        for layer in self.net.layers:
            del layer
        del self.net  # Delete the model
        del self.optimizer  # Delete the optimizer
        torch.cuda.empty_cache()
        gc.collect()

    def socket_listener(self, conn):
        """
        Listen for incoming socket data and update Trainer object values accordingly.

        This method listens for data sent through a socket connection, processes the incoming data,
        and updates the model's hyperparameters, including epochs, sample size, and layer configurations.

        :param conn: The connection object received from the connect_socket() method.
        :return: None
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
                    print(
                        f"Updated EPOCHS to: {self.epochs} \n SAMPLE RATE to: {self.sample_size} \n NUM_OF_CONV_LAYERS to: {self.num_of_conv_layers} \n NUM_OF_POOL_LAYERS to: {self.num_of_pool_layers} \n NUM_OF_DENSE_LAYERS to: {self.num_of_dense_layers} \n NUM_OF_LSTM_LAYERS to: {self.num_of_lstm_layers} \n NUM_OF_GRU_LAYERS to: {self.num_of_gru_layers} \n NUM_OF_RNN_LAYERS to: {self.num_of_rnn_layers} \n LR to: {self.lr} \n BATCH_SIZE to: {self.batch_size}")
                except ValueError:
                    print("Invalid input. Please enter a valid integer for the new number of epochs.")
            except:
                self.disconnect_socket(conn)
                conn, addr = self.server_socket.accept()

    def connect_socket(self):
        """
        Initialize a server socket, bind it to a specific address, and wait for incoming connections.

        This method sets up the server socket to listen for connections and returns the connection instance and
        a listener thread that will handle incoming data from the socket.

        :return: conn (socket instance), listen_thread (Thread instance)
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
        Close the socket connection.

        This method simply closes the provided socket connection.

        :param conn: The socket connection object to close.
        :return: None
        """
        conn.close()

    def start_controller(self):
        """
        Initialize socket connections for live updates and predictions.

        This method loads configuration parameters from a JSON file, connects to the necessary sockets, and
        returns the socket instances used for live communication and predictions.

        :return: None
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
    size_of_batch = config['size_of_batch']
    tr = Trainer(True, config['initial_epochs'], config['initial_sampling_rate'], config['size_of_batch'], config["lr"],
                 config['num_of_conv_layers'], config['num_of_pool_layers'], config['num_of_dense_layers'],
                 config['num_of_lstm_layers'], config['num_of_gru_layers'], config['num_of_rnn_layers'],
                 config['sample_rate_slack'], unique_class_labels, sequence_length)

    conn, listen_thread = tr.connect_socket()
    tr.start_controller()
    net = tr.train()
    nas_to_change = False
    while net == 1:
        torch.save(tr.net.state_dict(), 'weights_only.pth')
        tr.clear_mem()
        if nas_to_change:
            tr.prediction_socket_nas.sendall(pickle.dumps(
                [tr.num_of_conv_layers, tr.num_of_pool_layers, tr.num_of_dense_layers, tr.num_of_lstm_layers,
                 tr.num_of_gru_layers, tr.num_of_rnn_layers]))
            nas_to_change = False
        if tr.nas_changed:
            nas_to_change = True
        else:
            tr.prediction_socket.sendall(pickle.dumps(['2']))
        tr.__init__(False, tr.epochs, tr.sample_size, tr.batch_size, tr.lr,
                    tr.num_of_conv_layers, tr.num_of_pool_layers,
                    tr.num_of_dense_layers, tr.num_of_lstm_layers, tr.num_of_gru_layers, tr.num_of_rnn_layers,
                    config['sample_rate_slack'], unique_class_labels, sequence_length)
        net = tr.train()
