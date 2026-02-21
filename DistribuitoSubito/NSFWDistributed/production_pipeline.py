import atexit
import json
import math
import os
import pickle
import random
import sys
import time
from collections import Counter
from typing import Optional

import pandas as pd
from PIL import Image
from matplotlib import pyplot as plt
from torchvision import datasets

from sphere_exploration.factory import SphereExplorationFactory
import numpy
import ray
import torch
from threading import Thread
from sphere_exploration.gm_utils import GMUtils
from ray.data import Dataset
from torch.utils.data import TensorDataset
from torchsummary import summary
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import socket
from tqdm import tqdm
from kafka import KafkaConsumer, TopicPartition
import dask
from dask.distributed import Client, LocalCluster
import sampling_lib
from sphere_exploration.max_min_on_sphere import ParameterSpaceExplorer

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


# Define the structure of the Neural Network
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
                        nn.Conv2d(dataset_shape[0], CONV_NEURONS_CONST, kernel_size=kernel_size, stride=1,
                                  padding=CONV_PADDING))
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
                            nn.Conv2d(conv_tmp_old, CONV_NEURONS_BOUND, kernel_size=kernel_size, stride=1,
                                      padding=CONV_PADDING))
                        conv_tmp_old = CONV_NEURONS_BOUND
            for i in range(num_of_conv_layers - num_of_pool_layers, num_of_conv_layers):
                if conv_tmp <= CONV_NEURONS_BOUND:
                    self.layers.append(
                        nn.Conv2d(conv_tmp_old, conv_tmp, kernel_size=kernel_size, stride=1, padding=CONV_PADDING))
                    conv_tmp_old = conv_tmp
                    conv_tmp = conv_tmp * 2
                else:
                    self.layers.append(nn.Conv2d(conv_tmp_old, CONV_NEURONS_BOUND, kernel_size=kernel_size, stride=1,
                                                 padding=CONV_PADDING))
                    conv_tmp_old = CONV_NEURONS_BOUND
                # TODO: FIX PADDING
                self.layers.append(nn.MaxPool2d(kernel_size=2, stride=2, padding=MAX_POOL_PADDING))

        elif num_of_conv_layers == num_of_pool_layers:
            for i in range(0, num_of_conv_layers):
                if i == 0:
                    self.layers.append(
                        nn.Conv2d(dataset_shape[0], CONV_NEURONS_CONST, kernel_size=kernel_size, stride=1,
                                  padding=CONV_PADDING))
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
                            nn.Conv2d(conv_tmp_old, CONV_NEURONS_BOUND, kernel_size=kernel_size, stride=1,
                                      padding=CONV_PADDING))
                        conv_tmp_old = CONV_NEURONS_BOUND
                    self.layers.append(nn.MaxPool2d(kernel_size=2, stride=2, padding=MAX_POOL_PADDING))

        else:
            for i in range(0, num_of_conv_layers):
                if i == 0:
                    self.layers.append(
                        nn.Conv2d(dataset_shape[0], CONV_NEURONS_CONST, kernel_size=kernel_size, stride=1,
                                  padding=CONV_PADDING))
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
                            nn.Conv2d(conv_tmp_old, CONV_NEURONS_BOUND, kernel_size=kernel_size, stride=1,
                                      padding=CONV_PADDING))
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


@ray.remote(num_gpus=0.25)
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
                   num_of_dense_layers, n_workers, sample_size_slack, unique_class_labels, controller):
        """
        :param rebuild_data: if False we only resample the train set
        :param epochs: if no update is received
        :param initial_sample_size: if no update is received
        :param batch_size: batch size of the neural network
        """
        print(self.rank, ' started initializing')
        self.rebuild_data = rebuild_data
        self.epochs = epochs
        self.sample_size = sample_size
        self.old_sample_size = self.sample_size
        self.num_of_conv_layers = num_of_conv_layers
        self.num_of_pool_layers = num_of_pool_layers
        self.num_of_dense_layers = num_of_dense_layers
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
                # consumer_images = KafkaConsumer("train-topic", group_id='group3'+self.rank.__str__(), bootstrap_servers=['127.0.0.1:9092'],
                #                                 auto_offset_reset='earliest')
                # try:
                #     received_images = []
                #     received_labels = []
                #     tmp_count = 0
                #     my_print_flag0 = True
                #     my_print_flag1 = True
                #     max0 = -1
                #     max1 = -1
                #     for message in consumer_images:
                #         if my_print_flag0 and message.partition == 0:
                #             print("Partition0")
                #             print(message.offset)
                #             my_print_flag0 = False
                #         if my_print_flag1 and message.partition == 1:
                #             print("Partition1")
                #             print(message.offset)
                #             my_print_flag1 = False
                #         if message.partition == 0:
                #             max0 = message.offset
                #         else:
                #             max1 = message.offset
                #         parts = []
                #         for partition in consumer_images.partitions_for_topic("train-topic"):
                #             parts.append(TopicPartition("train-topic", partition))
                #         end_offsets = consumer_images.end_offsets(parts)
                #         end_offset = list(end_offsets.values())[0]
                #         # print(f"End offset is: {end_offset}")
                #         # print ("%s:%d:%d: key=%s value=%s" % (message.topic, message.partition, message.offset, message.key, message.value))
                #         if message.partition == 0:
                #             decode_img = np.frombuffer(message.value, dtype=np.uint8)
                #             received_images.append(decode_img)
                #             del decode_img
                #         else:
                #             received_labels.append(message.value)
                #         tmp_count = tmp_count + 1
                #         if tmp_count >= 2 * (end_offset):
                #             consumer_images.poll(timeout_ms=1, update_offsets=False)
                #             for partition in consumer_images.assignment():
                #                 consumer_images.seek(partition, 0)
                #             print("Spoiler:")
                #             print(len(received_images))
                #             print(max0)
                #             print(len(received_labels))
                #             print(max1)
                #             break
                # except KeyboardInterrupt:
                #     sys.exit()
                # consumer_images.close()
                # print(f"Receiven Images: {len(received_images)}")
                # print(f"Received Labels: {len(received_labels)}")
                # self.received_images_reshaped = reshape_for_pytorch(received_images)
                # self.received_labels_decoded = []
                # for i in range(0, len(received_labels)):
                #     l = int(received_labels[i].decode("utf-8"))
                #     self.received_labels_decoded.append(l)
                dataset_path = 'nsfw_dataset/train'
                self.received_images_reshaped = []
                self.received_labels_decoded = []
                class_names = sorted(os.listdir(dataset_path))
                class_to_index = {name: idx for idx, name in enumerate(class_names)}
                for class_name in class_names:
                    class_dir = os.path.join(dataset_path, class_name)
                    if not os.path.isdir(class_dir):
                        continue
                    counter = -1
                    for file_name in os.listdir(class_dir):
                        counter += 1
                        if counter % num_trainers != self.rank:
                            continue
                        file_path = os.path.join(class_dir, file_name)
                        try:
                            img = Image.open(file_path).convert('RGB')  # Convert to RGB
                            # img = img.resize((50, 50))  # Resize to 50x50
                            img_array = np.array(img)  # Convert to numpy array (50, 50, 3)
                            if img_array.shape == (50, 50, 3):
                                img_array = np.transpose(img_array, (2, 0, 1))  # Convert to (3, 50, 50)
                                self.received_images_reshaped.append(img_array)
                                self.received_labels_decoded.append(class_to_index[class_name])
                        except Exception as e:
                            print(f"Error loading {file_path}: {e}")
                self.received_images_reshaped = [np.float32(img / 255.0) for img in self.received_images_reshaped]

        else:
            self.cluster.scale(n_workers)
        self.net = Net(self.num_of_conv_layers, self.num_of_pool_layers, self.num_of_dense_layers, dataset_shape_torch,
                       unique_class_labels)
        # self.net.half()
        self.net.to(self.device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
        # summary(self.net, input_size=tuple(dataset_shape_torch), batch_size=self.batch_size)
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
        # # ─── 1.  Display ten images ────────────────────────────────────────────────────
        # num_to_show = min(10, len(train_images))  # don’t exceed dataset size
        # for idx in range(num_to_show):
        #     plt.figure()
        #     plt.imshow(np.transpose(train_images[idx], (1, 2, 0)))
        #     plt.title(f"Label: {int(train_labels[idx])}")
        #     plt.axis("off")
        #     plt.show()
        # # ─── 2.  Distribution (count & %)  ─────────────────────────────────────────────
        # label_counts = Counter(int(lbl) for lbl in train_labels)
        # df = (pd.DataFrame.from_dict(label_counts, orient="index", columns=["Count"])
        #       .sort_index())
        # df["Percentage"] = df["Count"] / len(train_labels) * 100
        #
        # print("Label distribution:")
        # print(df.to_string())
        # print(f"\nTotal number of elements: {len(train_labels)}")
        train_dataset = ray.data.from_items(data)
        self._ray_dataset = train_dataset
        self.trainset = self._ray_dataset.iter_torch_batches(batch_size=self.batch_size)
        self.train_iter = iter(self.trainset)
        del train_images
        del train_labels
        self.sa_end = time.time() - sa_start
        print(self.rank, ' finished initializing')
        return 0

    def train(self, weights, round):
        """
        Train the Neural Network with the trainset
        """

        self.net = Net(self.num_of_conv_layers, self.num_of_pool_layers, self.num_of_dense_layers, dataset_shape_torch,
                       unique_class_labels)
        # self.net.half()
        self.net.to(self.device)
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
            current_weight_grads = [p.clone().detach().cpu() for p in self.net.parameters()]
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

    # def set_rebuild_data(self, rebuild_data):
    #     self.rebuild_data = rebuild_data
    #
    # def set_epochs(self, epochs):
    #     self.epochs = epochs
    #
    # def set_sample_size(self, sample_size):
    #     self.sample_size = sample_size
    #
    # def set_num_of_conv_layers(self, num_of_conv_layers):
    #     self.num_of_conv_layers = num_of_conv_layers
    #
    # def set_num_of_pooling_layers(self, num_pooling_layers):
    #     self.num_pooling_layers = num_pooling_layers
    #
    # def set_num_of_dense_layers(self, num_of_dense_layers):
    #     self.num_of_dense_layers = num_of_dense_layers

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

    def __init__(self, num_trainers, num_of_conv_layers, num_of_pool_layers, num_of_dense_layers, n_workers, epochs,
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
        self.net = Net(self.num_of_conv_layers, self.num_of_pool_layers, self.num_of_dense_layers, dataset_shape_torch,
                       unique_class_labels)
        self.net.to(self.device)
        avg_weights = self.get_weights()
        self.net.train()
        self.old_sample_size = self.sample_size
        round = 1
        steps_total = 0
        time_total = 0
        self.print_training_setup()
        next_epoch_target_steps = self.sample_size * 22400 / self.batch_size
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
                    continue
            self.controller.set_shutdown_flag.remote(False)
            train_loss = np.mean(train_losses)
            train_accuracy = np.mean(train_accuracies)

            avg_weights = self.average_weights(worker_weights)

            # avg_weights = fed_opt.fed_adam(avg_weights, worker_weights)

            round += 1
            # # Evaluate
            self.net.set_weights(avg_weights)
            curr_epoch_time = time.time() - start
            time_total += time.time() - start

            # ray_dataset = ray.get(self.trainers[0].get_data_loader.remote())
            # train_loader = ray_dataset.iter_torch_batches(batch_size=self.batch_size)
            # train_accuracy, train_loss = evaluate(self.net, train_loader)

            if steps_total > next_epoch_target_steps:
                epochs_completed += 1
                next_epoch_target_steps = (epochs_completed+1) * self.sample_size * 22400 / self.batch_size
                epoch_duration_estimation = time_total + (self.epochs-epochs_completed) * (time_total / epochs_completed)
                serialized_df = pickle.dumps([train_accuracy, train_loss, curr_epoch_time, epoch_duration_estimation])
                print("Accuracy:", train_accuracy)
                self.live_socket.sendall(serialized_df)
            if self.nas_changed or self.old_sample_size != self.sample_size:
                if self.nas_changed:
                    print("NAS changed--Loading new net to Actors")
                    return
                else:
                    print("Loading new data to Actors")
                    load_res = [trainer.load_actor.remote(False, self.epochs, self.sample_size, self.batch_size,
                                                          self.lr, self.num_of_conv_layers,
                                                          self.num_of_pool_layers, self.num_of_dense_layers,
                                                          self.n_workers, config['sample_rate_slack'],
                                                          unique_class_labels,
                                                          self.controller) for trainer in self.trainers]
                    res = ray.get(load_res)
                    self.old_sample_size = self.sample_size
                    if res != [0, 0]:
                        print("Loading to Actors failed")
                    self.print_training_setup()
                    print("Loaded new data to Actors")
            if steps_total > self.epochs * self.sample_size * 22400 / self.batch_size:
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
                        self.num_of_conv_layers = int(float(data[2]))
                        self.num_of_pool_layers = int(float(data[3]))
                        self.num_of_dense_layers = int(float(data[4]))
                        self.n_workers = int(float(data[5]))
                        self.nas_changed = True
                    # try:
                    #     self.lr = float(data[5])
                    #     self.batch_size = int(data[6])
                    # except:
                    #     pass
                    print(
                        f"Updated EPOCHS to: {self.epochs} \n SAMPLE RATE to: {self.sample_size} \n NUM_OF_CONV_LAYERS to: {self.num_of_conv_layers} \n NUM_OF_POOL_LAYERS to: {self.num_of_pool_layers} \n NUM_OF_DENSE_LAYERS to: {self.num_of_dense_layers} \n N_WORKERS to: {self.n_workers} \n LR to: {self.lr} \n BATCH_SIZE to: {self.batch_size}")
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
        with open('config.json') as json_file:
            config = json.load(json_file)
    except:
        print("config.json not found")
        exit()
    thread_per_worker = 1
    dask.config.set(scheduler='threads', num_of_workers=config['n_workers'], threads_per_worker=thread_per_worker)
    # cluster = LocalCluster(n_workers=config['n_workers'], threads_per_worker=thread_per_worker,
    #                        dashboard_address=':8887')
    # client = Client(cluster)
    # print(f"Dashboard link: {client.dashboard_link}")
    sampling_method_id = 2
    tmp_filter_test = config['stream_batch_test']
    unique_class_labels = range(config['num_of_classes'])
    current_offset = 0
    num_trainers = config['num_of_trainers']
    dataset_shape_torch = config['dataset_shape_torch']
    context = ray.init(ignore_reinit_error=True)
    print(context.dashboard_url)
    ps = ParameterServer(num_trainers, config['num_of_conv_layers'],
                         config['num_of_pool_layers'], config['num_of_dense_layers'], config['n_workers'],
                         config['initial_epochs'], config['initial_sampling_rate'], config["lr"],
                         config['size_of_batch'])
    conn, listen_thread = ps.connect_socket()
    ps.start_controller()
    trainers = [Trainer.remote(i) for i in range(num_trainers)]
    load_res = [trainer.load_actor.remote(True, ps.epochs, ps.sample_size, ps.batch_size,
                                          ps.lr, ps.num_of_conv_layers,
                                          ps.num_of_pool_layers, ps.num_of_dense_layers,
                                          ps.n_workers, config['sample_rate_slack'], unique_class_labels, ps.controller)
                for trainer in trainers]
    res = ray.get(load_res)
    ps.set_trainers(trainers)
    nas_to_change = False
    ps.train()

    while True:
        if nas_to_change:
            packet = pickle.dumps([ps.num_of_conv_layers, ps.num_of_pool_layers, ps.num_of_dense_layers])
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
                                              ps.lr, ps.num_of_conv_layers,
                                              ps.num_of_pool_layers, ps.num_of_dense_layers,
                                              ps.n_workers, config['sample_rate_slack'], unique_class_labels,
                                              ps.controller)
                    for trainer in trainers]
        res = ray.get(load_res)
        ps.train()
