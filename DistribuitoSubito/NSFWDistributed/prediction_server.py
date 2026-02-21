import copy
import json
import os
import platform
import signal
import sys
import threading
import pickle
import time
import traceback
from multiprocessing import Manager
import torch
import socket


class Server:
    live_socket = None

    def __init__(self, num_of_predictors, num_of_conv_layers, num_of_pool_layers, num_of_dense_layers):
        self.num_of_predictors = num_of_predictors
        self.state_dict = torch.load("weights_only.pth", map_location="cpu")
        self.inf_times = [0] * num_of_predictors
        self.predictor_class_stats = [0] * num_of_predictors
        self.new_weights_flag = False
        self.num_of_conv_layers = num_of_conv_layers
        self.num_of_pool_layers = num_of_pool_layers
        self.num_of_dense_layers = num_of_dense_layers
        for predictor_id in range(num_of_predictors):
            self.inf_times[predictor_id] = []
            self.predictor_class_stats[predictor_id] = []

    def handle_client(self, predictor_id, client_socket):
        """
        Handle each predictor client connection and fetch their data

        :param predictor_id: an id given to each predictor instance
        :param client_socket: the socket that receives data from each predictor instance

        """
        with client_socket:
            print("Client connected.")
            t = 0
            while True:
                data = client_socket.recv(1024)
                t += 1
                if not data:
                    break
                # print(pickle.loads(data))
                new_class_stats = pickle.loads(data)
                if new_class_stats == "stop":
                    print("Client disconnected.")
                    break
                if len(self.inf_times[predictor_id]) >= 10:
                    self.inf_times[predictor_id].pop(0)
                if len(self.predictor_class_stats[predictor_id]) >= 10:
                    self.predictor_class_stats[predictor_id].pop(0)
                self.inf_times[predictor_id].append(new_class_stats[-1])
                self.predictor_class_stats[predictor_id].append(new_class_stats[:-1])
                for i in range(len(new_class_stats) - 1):
                    self.class_stats[i] += new_class_stats[i]
                print("Received:", self.class_stats)

    def accept_connections(self):
        """
        Accept connections from each predictor instance

        :return: client_pids
        """
        class_stats = [0] * (len(unique_class_labels))
        print(class_stats)
        with Manager() as manager:
            self.class_stats = manager.list(class_stats)
            self.client_sockets = []
            threads = []
            client_pids = []
            script = "prediction_client.py"
            python_exe = sys.executable
            for predictor_id in range(self.num_of_predictors):
                pid = os.spawnl(os.P_NOWAIT, python_exe, python_exe, script, str(predictor_id))
                client_pids.append(pid)
                print(f"Spawned process with PID: {pid}")
            print("Waiting for Production Pipeline")
            while self.new_weights_flag is False:
                if self.live_socket is None:
                    self.connect_socket()
                    self.start_controller()
            for predictor_id in range(self.num_of_predictors):
                client_socket, addr = server_socket.accept()
                self.client_sockets.append(client_socket)
                print(f"Accepted connection from {addr}")
                thread = threading.Thread(target=self.handle_client, args=(predictor_id, client_socket,))
                thread.start()
                threads.append(thread)
            for client_socket in self.client_sockets:
                start_sig = 'start'
                client_socket.sendall(start_sig.encode())
            thread = threading.Thread(target=self.send_live_stats)
            thread.start()
            threads.append(thread)
            for thread in threads:
                thread.join()
        return client_pids

    def send_live_stats(self,):
        """
        Send aggregated stats of predictor statistics to the Subito dashboard every 3 seconds
        """
        while True:
            try:
                tps = [0]*self.num_of_predictors
                for predictor_id in range(self.num_of_predictors):
                    if len(self.predictor_class_stats[predictor_id]) == len(self.inf_times[predictor_id]):
                        cum_sum = sum([sum(self.predictor_class_stats[predictor_id][idx]) for idx in range(len(self.predictor_class_stats[predictor_id]))])
                        tps[predictor_id] = cum_sum / sum(self.inf_times[predictor_id])
                tp = sum(tps)
            except Exception as e:
                print('Waiting for stats')
                time.sleep(5)
                continue
            class_stats_packet = copy.deepcopy(self.class_stats)
            class_stats_packet.append(tp)
            packet = pickle.dumps(class_stats_packet)
            self.live_socket.sendall(len(packet).to_bytes(8, 'big'))  # send size first
            self.live_socket.send(packet)
            time.sleep(3)

    def socket_listener_weights(self, conn):
        """
        Handle socket communication and update predictor weights

        :param conn: received from connect_socket()
        """
        while True:
            # Receive data from the socket
            # try:
            length_bytes = conn.recv(8)
            total_length = int.from_bytes(length_bytes, 'big')
            print(f"Received {total_length} bytes of weights update")
            # Then receive the actual data
            buffer = b''
            while len(buffer) < total_length:
                packet = conn.recv(262144)
                if not packet:
                    break
                buffer += packet
            try:
                self.new_weights_flag = True
                print("Sending weights...")
                packet = b'wei'
                packet += buffer
                for client_socket in self.client_sockets:
                    client_socket.sendall(len(packet).to_bytes(8, 'big'))  # send size first
                    client_socket.sendall(packet)
                print("Weights sent.")
                self.new_weights_flag = False
            except ValueError:
                print("Invalid input. Please enter a valid model.")
            # except:
            #   self.disconnect_socket(conn)
            #   conn, addr = self.server_socket.accept()

    def socket_listener_nas(self, conn):
        """
        Handle socket communication and update predictor NAS

        :param conn: received from connect_socket()
        """
        try:
            with open('config.json') as json_file:
                config = json.load(json_file)
        except:
            print("config.json not found")
            exit()
        unique_class_labels = range(config['num_of_classes'])
        while True:
            length_bytes = conn.recv(8)
            total_length = int.from_bytes(length_bytes, 'big')
            print(f"Received {total_length} bytes of nas update")
            buffer = b''
            while len(buffer) < total_length:
                packet = conn.recv(4096)
                if not packet:
                    break
                buffer += packet
            self.num_of_conv_layers, self.num_of_pool_layers, self.num_of_dense_layers = pickle.loads(buffer)
            tmp = (self.num_of_conv_layers, self.num_of_pool_layers, self.num_of_dense_layers)
            packet = b'nas'
            packet += pickle.dumps(tmp)
            for client_socket in self.client_sockets:
                client_socket.sendall(len(packet).to_bytes(8, 'big'))
                client_socket.sendall(packet)
            self.class_stats = [0] * (len(unique_class_labels) + 1)
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
        listen_thread = threading.Thread(target=self.socket_listener_weights, args=(conn,))
        listen_thread.daemon = True
        listen_thread.start()
        self.server_nas_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_nas_socket.bind((config['host_address'], config['prediction_nas_port']))
        self.server_nas_socket.listen(1)  # Allow 1 failed connection
        conn, addr = self.server_nas_socket.accept()
        print("Connected by", addr)
        # Spawn and start a thread to listen for new data
        listen_thread = threading.Thread(target=self.socket_listener_nas, args=(conn,))
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
    try:
        with open('config.json') as json_file:
            config = json.load(json_file)
    except:
        print("config.json not found")
        exit()
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((config["host_predictors"], config["prediction_server_port"]))
    server = Server(config['num_of_predictors'], config['num_of_conv_layers'], config['num_of_pool_layers'], config['num_of_dense_layers'])
    unique_class_labels = range(config['num_of_classes'])
    server_socket.listen()
    print("Socket server is listening...")
    client_pids = server.accept_connections()
    throughput = sum(server.class_stats[:-1]) / max([sum(server.inf_times[x]) for x in range(len(server.inf_times))])
    print('Throughput: ', throughput, ' data_points/s')
    for pid in client_pids:
        try:
            if platform.system() == 'Windows':
                import ctypes
                PROCESS_TERMINATE = 1
                handle = ctypes.windll.kernel32.OpenProcess(PROCESS_TERMINATE, False, pid)
                ctypes.windll.kernel32.TerminateProcess(handle, -1)
                ctypes.windll.kernel32.CloseHandle(handle)
                print(f"Terminated PID (Windows): {pid}")
            else:
                # On Unix-like systems
                os.kill(pid, signal.SIGTERM)
                print(f"Terminated PID (Unix): {pid}")
        except Exception as e:
            print(f"Failed to terminate PID {pid}: {e}")
