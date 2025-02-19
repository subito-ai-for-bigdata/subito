import json
import pickle
import sys
import time
import numpy
import torch
from threading import Thread
import pandas as pd
from matplotlib import pyplot as plt
from torch.utils.data import SubsetRandomSampler, TensorDataset
from torchsummary import summary
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import socket
from tqdm import tqdm
from kafka import KafkaProducer, KafkaConsumer, TopicPartition
from kafka.errors import KafkaError
import sampling_lib
from torchvision import transforms, datasets
from collections import defaultdict

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
    Reshapes and transposes the input images to match the expected input format for PyTorch models.

    :param res_images: A list or array of images to reshape.
    :param dataset_shape: The shape to reshape each image to.
    :return: A list of reshaped and transposed images in the format (C, H, W).
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
    Calculates the appropriate size of the flattened tensor after a convolutional operation.

    :param w: The width (or height) of the input image.
    :param k: The kernel size of the convolution.
    :param s: The stride of the convolution.
    :param p: The padding applied to the convolution.
    :param m: Boolean flag indicating if max pooling is applied after the convolution (default is True).
    :return: The flattened output shape as a tuple representing the size of the flattened tensor.
    """
    return int((np.floor((w - k + 2 * p) / s) + 1) / 2 if m else 1), k, s, p, m


# Define the structure of the Neural Network
class Net(nn.Module):
    """
    A custom deep neural network architecture that allows dynamic construction
    based on a given list of layers. The network supports various layer types including
    convolutional (conv), dense layers, pooling. The model adapts the architecture by adjusting 
    the number of units in the layers based on user-defined bounds.
    """
    FLAT_SHAPE_SIZE = -1
    def __init__(self, num_of_conv_layers, num_of_pool_layers, num_of_dense_layers, dataset_shape, unique_class_labels):
        """
        Initialize the neural network with convolutional, pooling, and dense layers based on the given configuration.

        The constructor builds the layers of the network by first adding the specified number of convolutional layers,
        followed by pooling layers, and then dense layers. The network's structure is dynamically adjusted based on the 
        provided number of layers and dataset shape. The final output layer is constructed based on the number of unique class labels.

        :param num_of_conv_layers: The number of convolutional layers to include in the network.
        :param num_of_pool_layers: The number of pooling layers to include in the network.
        :param num_of_dense_layers: The number of dense layers to include in the network.
        :param dataset_shape: The shape of the input dataset, typically in the form (channels, height, width).
        :param unique_class_labels: A list of unique class labels to determine the size of the output layer.
        
        :return: None
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
                    self.layers.append(nn.Conv2d(dataset_shape[0], CONV_NEURONS_CONST, kernel_size=kernel_size, stride=1, padding=CONV_PADDING))
                    conv_tmp_old = conv_tmp
                    conv_tmp = conv_tmp * 2
                else:
                    if conv_tmp <= CONV_NEURONS_BOUND:
                        self.layers.append(nn.Conv2d(conv_tmp_old, conv_tmp, kernel_size=kernel_size, stride=1, padding=CONV_PADDING))
                        conv_tmp_old = conv_tmp
                        conv_tmp = conv_tmp * 2
                    else:
                        self.layers.append(nn.Conv2d(conv_tmp_old, CONV_NEURONS_BOUND, kernel_size=kernel_size, stride=1, padding=CONV_PADDING))
                        conv_tmp_old = CONV_NEURONS_BOUND
            for i in range(num_of_conv_layers - num_of_pool_layers, num_of_conv_layers):
                if conv_tmp <= CONV_NEURONS_BOUND:
                    self.layers.append(nn.Conv2d(conv_tmp_old, conv_tmp, kernel_size=kernel_size, stride=1, padding=CONV_PADDING))
                    conv_tmp_old = conv_tmp
                    conv_tmp = conv_tmp * 2
                else:
                    self.layers.append(nn.Conv2d(conv_tmp_old, CONV_NEURONS_BOUND, kernel_size=kernel_size, stride=1, padding=CONV_PADDING))
                    conv_tmp_old = CONV_NEURONS_BOUND
                self.layers.append(nn.MaxPool2d(kernel_size=2, stride=2, padding=MAX_POOL_PADDING))
        
        elif num_of_conv_layers == num_of_pool_layers:
            for i in range(0, num_of_conv_layers):
                if i == 0:
                    self.layers.append(nn.Conv2d(dataset_shape[0], CONV_NEURONS_CONST, kernel_size=kernel_size, stride=1, padding=CONV_PADDING))
                    self.layers.append(nn.MaxPool2d(kernel_size=2, stride=2, padding=MAX_POOL_PADDING))
                    conv_tmp_old = conv_tmp
                    conv_tmp = conv_tmp * 2
                else:
                    if conv_tmp <= CONV_NEURONS_BOUND:
                        self.layers.append(nn.Conv2d(conv_tmp_old, conv_tmp, kernel_size=kernel_size, stride=1, padding=CONV_PADDING))
                        conv_tmp_old = conv_tmp
                        conv_tmp = conv_tmp * 2
                    else:
                        self.layers.append(nn.Conv2d(conv_tmp_old, CONV_NEURONS_BOUND, kernel_size=kernel_size, stride=1, padding=CONV_PADDING))
                        conv_tmp_old = CONV_NEURONS_BOUND
                    self.layers.append(nn.MaxPool2d(kernel_size=2, stride=2, padding=MAX_POOL_PADDING))
 
        else:
            for i in range(0, num_of_conv_layers):
                if i == 0:
                    self.layers.append(nn.Conv2d(dataset_shape[0], CONV_NEURONS_CONST, kernel_size=kernel_size, stride=1, padding=CONV_PADDING))
                    self.layers.append(nn.MaxPool2d(kernel_size=2, stride=2, padding=MAX_POOL_PADDING))
                    conv_tmp_old = conv_tmp
                    conv_tmp = conv_tmp * 2
                else:
                    if conv_tmp <= CONV_NEURONS_BOUND:
                        self.layers.append(nn.Conv2d(conv_tmp_old, conv_tmp, kernel_size=kernel_size, stride=1, padding=CONV_PADDING))
                        conv_tmp_old = conv_tmp
                        conv_tmp = conv_tmp * 2
                    else:
                        self.layers.append(nn.Conv2d(conv_tmp_old, CONV_NEURONS_BOUND, kernel_size=kernel_size, stride=1, padding=CONV_PADDING))
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
        Forward pass of the network. Takes an input tensor and processes it through the layers.
        
        :param x: Input tensor (e.g., image or sequence data).
        
        :return: The output of the network after passing through all layers.
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
        Calculates the flattened dimension size for the given dataset shape.
        
        :param dataset_shape: Shape of the dataset (e.g., (batch_size, height, width, channels)).
        
        :return: The flattened dimension size (integer).
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

    def __init__(self, rebuild_data, epochs, sample_size, batch_size, lr, num_of_conv_layers, num_of_pool_layers, num_of_dense_layers, sample_size_slack, unique_class_labels):
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
        :param sample_size_slack: Slack value for sample size.
        :param unique_class_labels: Number of unique class labels in the dataset.
        """
        self.nas_changed = False
        self.rebuild_data = rebuild_data
        if rebuild_data:
            self.epochs = epochs
            self.sample_size = sample_size
            self.old_sample_size = self.sample_size
            self.batch_size = batch_size
            self.lr = lr
            self.num_of_conv_layers = num_of_conv_layers
            self.num_of_pool_layers = num_of_pool_layers
            self.num_of_dense_layers = num_of_dense_layers
            self.sample_size_slack = sample_size_slack
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
                print("Running on the GPU")
            else:
                self.device = torch.device("cpu")
                print("Running on the CPU")
            if config['USE_KAFKA']:
                # Consumer-part for images
                consumer_images_test = KafkaConsumer("test-topic", group_id = 'group4', bootstrap_servers = ['127.0.0.1:9092'], auto_offset_reset = 'earliest')
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
        self.net = Net(self.num_of_conv_layers, self.num_of_pool_layers, self.num_of_dense_layers, dataset_shape_torch, unique_class_labels)
        self.net.to(self.device)
        summary(self.net, input_size=tuple(dataset_shape_torch), batch_size=self.batch_size)
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
        print('SAMPLE_SIZE:\t', self.sample_size*100, '%')
        print('NUM_OF_CONV:\t', self.num_of_conv_layers)
        print('NUM_OF_POOL:\t', self.num_of_pool_layers)
        print('NUM_OF_DENSE:\t', self.num_of_dense_layers)
        print('LR:\t', self.lr)
        print('BATCH_SIZE:\t', self.batch_size)

    def print_dataset_info(self):
        """
        Show the percentage of each class in trainset
        """
        total = 0
        counter_dict = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0}
        for data in self.trainset:
            Xs, ys = data
            for y in ys:
                counter_dict[int(y)] += 1
                total += 1
        print(counter_dict)
        for i in counter_dict:
            print(f"{i}: {counter_dict[i] / total * 100.0}%")

    def train(self):
        """
        Train the Neural Network with the trainset
        """
        net = self.net
        net.to(self.device)
        loss_function = nn.CrossEntropyLoss()
        optimizer = optim.Adam(net.parameters(), lr=self.lr)
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
            print(f"Epoch: {epoch+1} starting out of {self.epochs}")
            # Train the neural network
            for X, y in tqdm(self.trainset):
                X, y = X.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                # Forward pass
                output = net(X)
                # calculate the batch loss
                loss = loss_function(output, y)
                loss.backward()
                optimizer.step()
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
                    if int(float(data[2]))!=-1:
                        self.num_of_conv_layers = int(float(data[2]))
                        self.num_of_pool_layers = int(float(data[3]))
                        self.num_of_dense_layers = int(float(data[4]))
                        self.nas_changed = True
                    print(f"Updated EPOCHS to: {self.epochs} \n SAMPLE RATE to: {self.sample_size} \n NUM_OF_CONV_LAYERS to: {self.num_of_conv_layers} \n NUM_OF_POOL_LAYERS to: {self.num_of_pool_layers} \n NUM_OF_DENSE_LAYERS to: {self.num_of_dense_layers} \n LR to: {self.lr} \n BATCH_SIZE to: {self.batch_size}")
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
            with open('config.json') as json_file:
                config = json.load(json_file)
        except:
            print("config.json not found")
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
    tr = Trainer(True, config['initial_epochs'], config['initial_sampling_rate'], config['size_of_batch'], config["lr"],
                 config['num_of_conv_layers'], config['num_of_pool_layers'], config['num_of_dense_layers'],
                 config['sample_rate_slack'], unique_class_labels)
    conn, listen_thread = tr.connect_socket()
    tr.start_controller()
    net = tr.train()
    nas_to_change = False
    while net == 1:
        torch.save(tr.net.state_dict(), 'weights_only.pth')
        if nas_to_change:
            tr.prediction_socket_nas.sendall(pickle.dumps([tr.num_of_conv_layers, tr.num_of_pool_layers, tr.num_of_dense_layers]))
            nas_to_change = False
        if tr.nas_changed:
            nas_to_change = True
        else:
            tr.prediction_socket.sendall(pickle.dumps(['2']))
        tr.__init__(False, tr.epochs, tr.sample_size, tr.batch_size, tr.lr,
                    tr.num_of_conv_layers, tr.num_of_pool_layers,
                    tr.num_of_dense_layers, config['sample_rate_slack'],
                    unique_class_labels)
        net = tr.train()
