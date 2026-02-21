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
                # TODO: FIX PADDING
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
                 batch_size, rank):
        """
        Initialize a Trainer object

        :param rebuild_data: if False we only resample the train set
        :param num_of_conv_layers: Number of convolutional layers
        :param num_of_pool_layers: Number of pooling layers
        :param num_of_dense_layers: Number of dense layers
        :param unique_class_labels: unique class labels
        :param batch_size: batch size
        :param rank: predictor_id
        """
        self.rebuild_data = rebuild_data
        self.num_of_conv_layers = num_of_conv_layers
        self.num_of_pool_layers = num_of_pool_layers
        self.num_of_dense_layers = num_of_dense_layers
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
                self.net = Net(self.num_of_conv_layers, self.num_of_pool_layers, self.num_of_dense_layers,
                               dataset_shape_torch,
                               unique_class_labels)
        if self.updated_NAS:
            self.net = Net(self.num_of_conv_layers, self.num_of_pool_layers, self.num_of_dense_layers, dataset_shape_torch,
               unique_class_labels)
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
            self.net.load_state_dict(self.state_dict)
            print("loading weights from socket")
        self.new_weights_flag = False
        self.net.to(self.device)
        # summary(self.net, input_size=tuple(dataset_shape_torch))

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
        import socket

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
                self.num_of_conv_layers, self.num_of_pool_layers, self.num_of_dense_layers = layers
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
        with open('config.json') as json_file:
            config = json.load(json_file)
    except:
        print("config.json not found")
        exit()
    dataset_shape_torch = config['dataset_shape_torch']
    tr = Trainer(True, config['num_of_conv_layers'], config['num_of_pool_layers'], config['num_of_dense_layers'],
                 range(config['num_of_classes']), config['predictor_batch_size'], rank)
    tr.connect_socket()
    while True:
        st = time.time()
        tr.__init__(False, tr.num_of_conv_layers, tr.num_of_pool_layers, tr.num_of_dense_layers, tr.unique_class_labels, tr.batch_size, tr.rank)
        net = tr.train()
        tr_time = time.time() - st
        print(tr_time)