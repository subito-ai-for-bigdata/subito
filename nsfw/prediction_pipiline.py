import json
import pickle
import sys
import torch
from threading import Thread
from torch.utils.data import SubsetRandomSampler, TensorDataset
from torchsummary import summary
import numpy as np
import socket
from tqdm import tqdm
from kafka import KafkaConsumer, TopicPartition
import torch.nn as nn
import torch.nn.functional as F

BUILD_TRAINSET = True

CONV_PADDING = 'same'
MAX_POOL_PADDING = 'same'
CONV_NEURONS_CONST = 32
CONV_NEURONS_BOUND = 256
DENSE_NEURONS_CONST = 128
DENSE_NEURONS_BOUND = 32

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
    :return: properly reshapes res_images
    """
    dataset_shape_for_decoding = [50, 50, 3]
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


class Net(nn.Module):
    FLAT_SHAPE_SIZE = -1

    def __init__(self, num_of_conv_layers, num_of_pool_layers, num_of_dense_layers, dataset_shape, unique_class_labels):
        """
        Initializes the dynamic NN structure based on the given layers

        :param num_of_conv_layers: Number of convolutional layers in the network.
        :param num_of_pool_layers: Number of pooling layers in the network.
        :param num_of_dense_layers: Number of fully connected (dense) layers in the network.
        :param dataset_shape: Shape of the input dataset (e.g., [channels, height, width]).
        :param unique_class_labels: List of unique class labels for classification.
        """
        super().__init__()
        conv_tmp = CONV_NEURONS_CONST
        conv_tmp_old = conv_tmp
        dense_tmp = DENSE_NEURONS_CONST
        dense_tmp_old = dense_tmp
        self.num_of_conv_layers = num_of_conv_layers
        self.num_of_pool_layers = num_of_pool_layers
        self.num_of_dense_layers = num_of_dense_layers
        self.layers = nn.ModuleList()
        kernel_size = 3

        # Part I: Convolutional part of our network
        if num_of_conv_layers > num_of_pool_layers:
            for i in range(0, num_of_conv_layers - num_of_pool_layers):
                if i == 0:
                    self.layers.append(
                        nn.Conv2d(dataset_shape[0], CONV_NEURONS_CONST, kernel_size=kernel_size, stride=1, padding=CONV_PADDING))
                    conv_tmp_old = conv_tmp
                    conv_tmp = conv_tmp * 2
                else:
                    if conv_tmp <= CONV_NEURONS_BOUND:
                        self.layers.append(
                            nn.Conv2d(conv_tmp_old, conv_tmp, kernel_size=kernel_size, stride=1, padding=CONV_PADDING))
                        conv_tmp_old = conv_tmp
                        conv_tmp = conv_tmp * 2
                    else:
                        self.layers.append(
                            nn.Conv2d(conv_tmp_old, CONV_NEURONS_BOUND, kernel_size=kernel_size, stride=1, padding=CONV_PADDING))
                        conv_tmp_old = CONV_NEURONS_BOUND
            for i in range(num_of_conv_layers - num_of_pool_layers, num_of_conv_layers):
                if conv_tmp <= CONV_NEURONS_BOUND:
                    self.layers.append(
                        nn.Conv2d(conv_tmp_old, conv_tmp, kernel_size=kernel_size, stride=1, padding=CONV_PADDING))
                    conv_tmp_old = conv_tmp
                    conv_tmp = conv_tmp * 2
                else:
                    self.layers.append(
                        nn.Conv2d(conv_tmp_old, CONV_NEURONS_BOUND, kernel_size=kernel_size, stride=1, padding=CONV_PADDING))
                    conv_tmp_old = CONV_NEURONS_BOUND
                self.layers.append(nn.MaxPool2d(kernel_size=2, stride=2, padding=MAX_POOL_PADDING))

        elif num_of_conv_layers == num_of_pool_layers:
            for i in range(0, num_of_conv_layers):
                if i == 0:
                    self.layers.append(
                        nn.Conv2d(dataset_shape[0], CONV_NEURONS_CONST, kernel_size=kernel_size, stride=1, padding=CONV_PADDING))
                    self.layers.append(nn.MaxPool2d(kernel_size=2, stride=2, padding=MAX_POOL_PADDING))
                    conv_tmp_old = conv_tmp
                    conv_tmp = conv_tmp * 2
                else:
                    if conv_tmp <= CONV_NEURONS_BOUND:
                        self.layers.append(
                            nn.Conv2d(conv_tmp_old, conv_tmp, kernel_size=kernel_size, stride=1, padding=CONV_PADDING))
                        conv_tmp_old = conv_tmp
                        conv_tmp = conv_tmp * 2
                    else:
                        self.layers.append(
                            nn.Conv2d(conv_tmp_old, CONV_NEURONS_BOUND, kernel_size=kernel_size, stride=1, padding=CONV_PADDING))
                        conv_tmp_old = CONV_NEURONS_BOUND
                    self.layers.append(nn.MaxPool2d(kernel_size=2, stride=2, padding=MAX_POOL_PADDING))

        else:
            for i in range(0, num_of_conv_layers):
                if i == 0:
                    self.layers.append(
                        nn.Conv2d(dataset_shape[0], CONV_NEURONS_CONST, kernel_size=kernel_size, stride=1, padding=CONV_PADDING))
                    self.layers.append(nn.MaxPool2d(kernel_size=2, stride=2, padding=MAX_POOL_PADDING))
                    conv_tmp_old = conv_tmp
                    conv_tmp = conv_tmp * 2
                else:
                    if conv_tmp <= CONV_NEURONS_BOUND:
                        self.layers.append(
                            nn.Conv2d(conv_tmp_old, conv_tmp, kernel_size=kernel_size, stride=1, padding=CONV_PADDING))
                        conv_tmp_old = conv_tmp
                        conv_tmp = conv_tmp * 2
                    else:
                        self.layers.append(
                            nn.Conv2d(conv_tmp_old, CONV_NEURONS_BOUND, kernel_size=kernel_size, stride=1, padding=CONV_PADDING))
                        conv_tmp_old = CONV_NEURONS_BOUND
                    self.layers.append(nn.MaxPool2d(kernel_size=2, stride=2, padding=MAX_POOL_PADDING))
            for i in range(num_of_conv_layers, num_of_pool_layers):
                self.layers.append(nn.MaxPool2d(kernel_size=2, stride=2, padding=MAX_POOL_PADDING))

        # Part II: Dense part of our network
        self.flat = nn.Flatten()

        for i in range(0, num_of_dense_layers):
            if i == 0:
                self.layers.append(nn.Linear(self.calculate_flatten_dim(dataset_shape), DENSE_NEURONS_CONST))
                dense_tmp_old = dense_tmp
                dense_tmp = dense_tmp // 2
            else:
                if dense_tmp <= DENSE_NEURONS_BOUND:
                    self.layers.append(nn.Linear(dense_tmp_old, dense_tmp))
                    dense_tmp_old = dense_tmp
                    dense_tmp = dense_tmp // 2
                else:
                    self.layers.append(nn.Linear(dense_tmp_old, DENSE_NEURONS_BOUND))
                    dense_tmp_old = DENSE_NEURONS_BOUND
        if num_of_dense_layers == 0:
            self.output_layer = nn.Linear(self.calculate_flatten_dim(dataset_shape), len(unique_class_labels))
        else:
            self.output_layer = nn.Linear(dense_tmp_old, len(unique_class_labels))

    def forward(self, x):
        """
        Forward pass for the dynamic model

        :param x: Input tensor with shape (batch_size, time_steps, height, width, channels)
        :return: output tensor after passing through all layers
        """
        flat_flag = True
        for i, layer in enumerate(self.layers):
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                if flat_flag and isinstance(layer, nn.Linear):
                    x = self.flat(x)
                    flat_flag = False
                x = F.relu(layer(x))
            else:
                x = layer(x)
        if flat_flag:
            x = self.flat(x)
        x = self.output_layer(x)
        return x

    def calculate_flatten_dim(self, dataset_shape):
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
        with torch.no_grad():
            for i, layer in enumerate(self.layers):
                if isinstance(layer, nn.Conv2d):
                    x = F.relu(layer(x))
                elif isinstance(layer, nn.MaxPool2d):
                    x = layer(x)
                else:
                    continue
        return x.numel()


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

    def __init__(self, rebuild_data, num_of_conv_layers, num_of_pool_layers, num_of_dense_layers, unique_class_labels,
                 class_stats):
        """
        Initializes the Trainer class with network architecture parameters and dataset configurations.

        :param rebuild_data: If False, only resamples the train set.
        :param num_of_conv_layers: Number of convolutional layers.
        :param num_of_pool_layers: Number of pooling layers.
        :param num_of_dense_layers: Number of dense layers.
        :param unique_class_labels: Unique class labels in the dataset.
        :param class_stats: Statistics of class distribution.
        """
        self.class_stats = class_stats
        self.rebuild_data = rebuild_data
        self.num_of_conv_layers = num_of_conv_layers
        self.num_of_pool_layers = num_of_pool_layers
        self.num_of_dense_layers = num_of_dense_layers
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
                train_images = self.received_images_reshaped
                train_labels = self.received_labels_decoded
                # Build initial or updated trainset (be careful Tensors to float32 not float64)
                train_images_tensor = torch.tensor(train_images, dtype=torch.float32)
                train_labels_tensor = torch.tensor(train_labels, dtype=torch.long)
                train_dataset = TensorDataset(train_images_tensor, train_labels_tensor)
                self.testset = torch.utils.data.DataLoader(train_dataset, shuffle=True)
        while self.new_weights_flag is False:
            if rebuild_data:
                self.connect_socket()
                self.start_controller()
                rebuild_data = False
            pass
        self.net = Net(self.num_of_conv_layers, self.num_of_pool_layers, self.num_of_dense_layers, dataset_shape_torch,
                       unique_class_labels)

        self.net.load_state_dict(torch.load('weights_only.pth'))
        self.new_weights_flag = False
        self.net.to(self.device)
        summary(self.net, input_size=tuple(dataset_shape_torch))

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
                    # communicate stats to streamlit
                if self.new_weights_flag is True:
                    break
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
            with open('config.json') as json_file:
                config = json.load(json_file)
        except:
            print("config.json not found")
            exit()
            args = sys.argv[1:]
        unique_class_labels = range(config['num_of_classes'])
        while True:
            # Receive data from the socket
            data = conn.recv(8192)
            if not data:
                break
            print(pickle.loads(data))
            self.num_of_conv_layers, self.num_of_pool_layers, self.num_of_dense_layers = pickle.loads(data)
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
            with open('config.json') as json_file:
                config = json.load(json_file)
        except:
            print("config.json not found")
            exit()
        # Initialize an IPv4 socket with TCP (default) and try to connect to the nn
        self.live_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.live_socket.connect((config["host_address"], config['prediction_live_port']))
        return


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")
    try:
        with open('config.json') as json_file:
            config = json.load(json_file)
    except:
        print("config.json not found")
        exit()
        args = sys.argv[1:]
    tmp_filter_test = config['stream_batch_test']
    sampling_method_id = 1
    unique_class_labels = range(config['num_of_classes'])
    current_offset = 0
    dataset_shape_torch = [3, 50, 50]
    class_stats = [0] * len(unique_class_labels)
    tr = Trainer(True, config['num_of_conv_layers'], config['num_of_pool_layers'], config['num_of_dense_layers'],
                 unique_class_labels, class_stats)
    net = tr.train()
    while net == 1:
        tr.__init__(False, tr.num_of_conv_layers, tr.num_of_pool_layers, tr.num_of_dense_layers, unique_class_labels,
                    tr.class_stats)
        net = tr.train()
