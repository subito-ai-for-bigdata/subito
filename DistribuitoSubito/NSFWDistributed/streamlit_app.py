import json
import pickle
import shutil
import socket
import struct
import subprocess
import time
from threading import Thread
import pandas as pd
import streamlit as st
from streamlit.runtime.scriptrunner import get_script_run_ctx
from streamlit.runtime.scriptrunner import add_script_run_ctx
import altair as alt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
import visualkeras
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import streamlit_scrollable_textbox as stx
import paretoset

# Define a CSS style for light and dark modes
if 'initialized' not in st.session_state or not st.session_state.initialized:
    st.session_state['last_arch_selected'] = -1
    st.session_state['sr'] = 100
    st.session_state['ep'] = 2
    original_file = 'gear_icon.png'
    # Loop to create three copies with the desired names
    for i in range(3):
        # Create the new file name
        new_file = f'original{i}.png'
        # Copy the file with the new name
        shutil.copyfile(original_file, new_file)
    st.session_state['sr1'] = 5
    st.session_state['ep1'] = 3
    st.session_state['dw1'] = 4
    st.session_state['sr2'] = 15
    st.session_state['ep2'] = 18
    st.session_state['dw2'] = 4
    st.session_state['sr3'] = 25
    st.session_state['ep3'] = 6
    st.session_state['dw3'] = 4
    print('Changed stats to default')
st.set_page_config(page_title="Pipeline Dashboard",
                   page_icon="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcS3ZCIY5dBOrUJeLuz6aMhO05BRNo58wk-dEg&s",
                   layout="wide")

# Initialize session state variables to manage the state of each toggle
if 'toggle1' not in st.session_state:
    st.session_state.toggle1 = False
if 'toggle2' not in st.session_state:
    st.session_state.toggle2 = False
if 'toggle3' not in st.session_state:
    st.session_state.toggle3 = False
if 'toggle_manual' not in st.session_state:
    st.session_state.toggle_manual = False


def send_new_epochs_to_training_process(socket, res_idx):
    """
    Use this function to send bytes to the training model.

    :param client_socket: socket object used for send
    :param data: pickle dump of new epochs and new sampling rate
    :return: 0 on success 1 on failure
    """
    try:
        if res_idx == -1:
            encoded = st.session_state['ep_manual'].__str__() + ',' + st.session_state[
                'sr_manual'].__str__() + ',-1,-1,-1'
        else:
            # Encode with default UTF-8
            res = st.session_state['bo_res'].iloc[res_idx]
            encoded = res["Epochs"].__str__() + ',' + res["Sample Size"].__str__() + ',' + res["Conv"].__str__() + ',' + \
                      res["Pool"].__str__() + ',' + res["Dense"].__str__() + ',' + res['Dask Workers'].__str__()
        socket.sendall(str(encoded).encode())
        return 0
    except ConnectionRefusedError:
        print("Could not connect to the training script socket. Make sure it is running.")
        return 1


def create_model(conv_number, pool_number, dense_number, CONV_NEURONS_CONST, DENSE_NEURONS_CONST, CONV_NEURONS_BOUND,
                 DENSE_NEURONS_BOUND):
    CONV_PADDING = 'same'
    MAX_POOL_PADDING = 'same'
    dataset_shape = [32, 32, 3]
    unique_class_labels = range(len(st.session_state["class_stats"][0:-2]))

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
                model.add(layers.Conv2D(int(CONV_NEURONS_CONST), (3, 3), activation='relu', input_shape=dataset_shape,
                                        padding=CONV_PADDING))
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
            model.add(layers.MaxPooling2D((2, 2), strides=(2, 2), padding=MAX_POOL_PADDING))

    elif conv_number == pool_number:
        for i in range(0, int(conv_number)):
            if i == 0:
                model.add(layers.Conv2D(int(CONV_NEURONS_CONST), (3, 3), activation='relu', input_shape=dataset_shape,
                                        padding=CONV_PADDING))
                conv_tmp = conv_tmp * 2
                model.add(layers.MaxPooling2D((2, 2), strides=(2, 2), padding=MAX_POOL_PADDING))
            else:
                if conv_tmp <= CONV_NEURONS_BOUND:
                    model.add(layers.Conv2D(conv_tmp, (3, 3), activation='relu', padding=CONV_PADDING))
                    conv_tmp = conv_tmp * 2
                else:
                    model.add(layers.Conv2D(CONV_NEURONS_BOUND, (3, 3), activation='relu', padding=CONV_PADDING))
                model.add(layers.MaxPooling2D((2, 2), strides=(2, 2), padding=MAX_POOL_PADDING))
    else:
        for i in range(0, int(conv_number)):
            if i == 0:
                model.add(layers.Conv2D(int(CONV_NEURONS_CONST), (3, 3), activation='relu', input_shape=dataset_shape,
                                        padding=CONV_PADDING))
                conv_tmp = conv_tmp * 2
                model.add(layers.MaxPooling2D((2, 2), strides=(2, 2), padding=MAX_POOL_PADDING))
            else:
                if conv_tmp <= CONV_NEURONS_BOUND:
                    model.add(layers.Conv2D(conv_tmp, (3, 3), activation='relu', padding=CONV_PADDING))
                    conv_tmp = conv_tmp * 2
                else:
                    model.add(layers.Conv2D(CONV_NEURONS_BOUND, (3, 3), activation='relu', padding=CONV_PADDING))
                model.add(layers.MaxPooling2D((2, 2), strides=(2, 2), padding=MAX_POOL_PADDING))

        for i in range(int(conv_number), int(pool_number)):
            model.add(layers.MaxPooling2D((2, 2), strides=(2, 2), padding=MAX_POOL_PADDING))

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
    model.add(layers.Dense(len(unique_class_labels), activation='softmax', name="dense_last"))

    model.compile(optimizer=Adam(lr),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    return model


def create_model_images():
    # @title Necessary Editing for the illustration of the NN architecture
    CONV_NEURONS_CONST = 32
    CONV_NEURONS_BOUND = 256
    DENSE_NEURONS_CONST = 128
    DENSE_NEURONS_BOUND = 32
    for i in range(3):
        BO_model = create_model(st.session_state['bo_res'].loc[i, 'Conv'], st.session_state['bo_res'].loc[i, 'Pool'],
                                st.session_state['bo_res'].loc[i, 'Dense'], CONV_NEURONS_CONST, DENSE_NEURONS_CONST,
                                CONV_NEURONS_BOUND, DENSE_NEURONS_BOUND)
        # font = ImageFont.load_default()
        # visualkeras.layered_view(BO_model, legend=True, font=font, to_file='original'+i.__str__()+'.png')
        # Generate the image with a specific scale
        color_map = defaultdict(dict)
        # "#ffd166", "#ef476f", "#118ab2", "#073b4c", "#842da1", "#ffbad4", "#fe9775", "#83d483", "#06d6a0", "#0cb0a9"
        color_map[layers.Conv2D]['fill'] = "#ffd166"
        color_map[layers.MaxPooling2D]['fill'] = "#ef476f"
        color_map[layers.Dense]['fill'] = "#842da1"
        color_map[layers.Flatten]['fill'] = "#0cb0a9"
        image = visualkeras.layered_view(BO_model, color_map=color_map,
                                         background_fill=(0, 0, 0, 0))  # Adjust the scale to your preference

        # Resize the image to exact dimensions (e.g., 800x600)
        image = image.resize((3000, 1800))
        image.save('original' + i.__str__() + '.png')
        del BO_model


def run_sbto():
    print('sbto started')
    subprocess.Popen("conda run python sbto.py", shell=True)
    print('sbto finished')


def refresh_sbto_res():
    st.session_state['subito_running'] = True
    # t = Thread(target=run_sbto, args=())
    # get_script_run_ctx(t)
    # add_script_run_ctx(t)
    # t.daemon = True
    # t.start()
    st.session_state['sr1'] = st.session_state['sr1']
    st.session_state['ep1'] = st.session_state['ep1']
    st.session_state['dw1'] = st.session_state['dw1']
    st.session_state['sr2'] = st.session_state['sr2']
    st.session_state['ep2'] = st.session_state['ep2']
    st.session_state['dw2'] = st.session_state['dw2']
    st.session_state['sr3'] = st.session_state['sr3']
    st.session_state['ep3'] = st.session_state['ep3']
    st.session_state['dw3'] = st.session_state['dw3']


# Functions to handle toggle logic
def toggle1_changed():
    if st.session_state.toggle1:
        st.session_state['last_arch_selected'] = 1
        new_file = 'tmp.png'
        # Copy the file with the new name
        shutil.copyfile('original0.png', new_file)
        st.session_state.toggle2 = False
        st.session_state.toggle3 = False
        st.session_state.toggle_manual = False
        send_new_epochs_to_training_process(st.session_state['prod_socket'], 0)


def toggle2_changed():
    if st.session_state.toggle2:
        st.session_state['last_arch_selected'] = 2
        new_file = 'tmp.png'
        # Copy the file with the new name
        shutil.copyfile('original1.png', new_file)
        st.session_state.toggle1 = False
        st.session_state.toggle3 = False
        st.session_state.toggle_manual = False
        send_new_epochs_to_training_process(st.session_state['prod_socket'], 1)


def toggle3_changed():
    if st.session_state.toggle3:
        st.session_state['last_arch_selected'] = 3
        new_file = 'tmp.png'
        # Copy the file with the new name
        shutil.copyfile('original2.png', new_file)
        st.session_state.toggle1 = False
        st.session_state.toggle2 = False
        st.session_state.toggle_manual = False
        send_new_epochs_to_training_process(st.session_state['prod_socket'], 2)


def toggle_manual_changed():
    if st.session_state.toggle_manual:
        st.session_state.toggle1 = False
        st.session_state.toggle2 = False
        st.session_state.toggle3 = False
        send_new_epochs_to_training_process(st.session_state['prod_socket'], -1)
        st.session_state['sr'] = round(st.session_state['sr_manual'] * 100, 1)
        st.session_state['ep'] = st.session_state['ep_manual']


def mode_select():
    if st.session_state['mode'] == "Manual Mode:wrench:":
        st.session_state['manual'] = True
    else:
        st.session_state['manual'] = False


def disable_deploy_manual():
    if st.session_state.toggle_manual:
        st.session_state.toggle_manual = False


def update_config_file(key):
    value = st.session_state[key]
    try:
        with open('config.json') as json_file:
            config = json.load(json_file)
    except:
        st.write("config.json not found")
        exit()
    if key == 'sample_size_low' or key == 'sample_size_high':
        config[key] = 1 - value
    else:
        config[key] = value
    with open('config.json', 'w') as file:
        json.dump(config, file)


def socket_listener_sbto(server_socket):
    """
      Handle socket communication and update object values

      :param conn: received from connect_socket()
      """

    def duplicate_list_elements(lst):
        return lst * 4  # Duplicates the list

    # Receive data from the socket
    server_socket.listen(1)  # Allow 1 failed connection
    conn, addr = server_socket.accept()
    print("Connected by", addr)
    while True:
        try:
            length_bytes = conn.recv(8)
            total_length = int.from_bytes(length_bytes, 'big')
            if total_length == 1:
                print("ping received")
                continue
            print(f"Received {total_length} bytes of optimizer results")
            # Then receive the actual data
            buffer = b''
            while len(buffer) < total_length:
                packet = conn.recv(4096)
                if not packet:
                    break
                buffer += packet
            data = buffer
            sbto_res_df = pd.DataFrame(pickle.loads(data))
            sbto_res_df_sorted = sbto_res_df.sort_values(by='Score', ascending=False).reset_index(drop=True)
            sbto_res_df_sorted['Loss Epoch'] = sbto_res_df_sorted['Loss Epoch'].apply(duplicate_list_elements)
            sbto_res_df_sorted['Acc Epoch'] = sbto_res_df_sorted['Acc Epoch'].apply(duplicate_list_elements)
            print(sbto_res_df_sorted)
            st.session_state['bo_res'] = sbto_res_df_sorted
            st.session_state['sr1'] = round(sbto_res_df_sorted['Sample Size'][0] * 100, 1)
            st.session_state['sr2'] = round(sbto_res_df_sorted['Sample Size'][1] * 100, 1)
            st.session_state['sr3'] = round(sbto_res_df_sorted['Sample Size'][2] * 100, 1)
            st.session_state['ep1'] = sbto_res_df_sorted['Epochs'][0]
            st.session_state['ep2'] = sbto_res_df_sorted['Epochs'][1]
            st.session_state['ep3'] = sbto_res_df_sorted['Epochs'][2]
            st.session_state['dw1'] = sbto_res_df_sorted['Dask Workers'][0]
            st.session_state['dw2'] = sbto_res_df_sorted['Dask Workers'][1]
            st.session_state['dw3'] = sbto_res_df_sorted['Dask Workers'][2]
            print('Changes made')
            st.session_state['subito_running'] = False
            st.session_state['terminal_invoked'] = False
            st.session_state['stdout'] = ''
        except:
            print('Disconnecting')
            conn, addr = server_socket.accept()
            print("Connected by", addr)
    return sbto_res_df


def socket_listener_live_sbto(socket):
    """
      Handle socket communication and update object values

      :param conn: received from connect_socket()
      """

    # Receive data from the socket
    socket.listen(1)  # Allow 1 failed connection
    conn, addr = socket.accept()
    print("Connected by", addr)
    while True:
        try:
            data_length_packed = conn.recv(4)
            if not st.session_state['terminal_invoked']:
                st.session_state['subito_running'] = True
            if not data_length_packed:
                raise RuntimeError("Connection closed or no data received")
            print(data_length_packed)
            data_length = struct.unpack('!I', data_length_packed)[0]
            print(data_length_packed)
            # Now receive the actual pickled data
            pickled_data = b''
            while len(pickled_data) < data_length:
                chunk = conn.recv(data_length - len(pickled_data))
                print("this chunk is:")
                print(len(chunk))
                if not chunk:
                    raise RuntimeError("Connection closed before all data was received")
                pickled_data += chunk
            print("The pickled data is:")
            print(len(pickled_data))
            print(pickled_data)
            data = pickle.loads(pickled_data)
            print(data)
            st.session_state['stdout'] = data + st.session_state['stdout']
            if not st.session_state['terminal_invoked']:
                st.session_state['terminal_invoked'] = True
            if not data:
                break
        except:
            print('Disconnecting live sbto socket')
            disconnect_socket(conn)
            conn, addr = socket.accept()
            print("Connected by", addr)


def socket_listener_live(live_socket):
    """
      Handle socket communication and update object values

      :param conn: received from connect_socket()
      """

    # Receive data from the socket
    live_socket.listen(1)  # Allow 1 failed connection
    conn, addr = live_socket.accept()
    st.session_state['live_metrics'] = pd.DataFrame(columns=['Acc', 'Loss', 'Time', 'Est_Tr_Time'])
    epoch_last = 1
    print("Connected by", addr)
    live_duration = 30
    while True:
        try:
            data = conn.recv(4096)
            df_row = pickle.loads(data)
            if len(st.session_state['live_metrics'].index) < live_duration:
                st.session_state['live_metrics'] = pd.concat(
                    [st.session_state['live_metrics'],
                     pd.DataFrame([df_row], columns=['Acc', 'Loss', 'Time', 'Est_Tr_Time'])],
                    ignore_index=True)
                st.session_state['live_metrics'].index = range(1, epoch_last + 1)
            else:
                st.session_state['live_metrics'] = pd.concat([st.session_state['live_metrics'].iloc[1:],
                                                              pd.DataFrame([df_row], columns=['Acc', 'Loss', 'Time',
                                                                                              'Est_Tr_Time'])],
                                                             ignore_index=False)
                st.session_state['live_metrics'].index = range(epoch_last - live_duration + 1, epoch_last + 1)
            epoch_last += 1
            # print(st.session_state['live_metrics'])
            if not data:
                break
        except:
            print('Disconnecting')
            disconnect_socket(conn)
            conn, addr = live_socket.accept()
            print("Connected by", addr)
    return


def socket_listener_live_prediction(socket):
    """
      Handle socket communication and update object values

      :param conn: received from connect_socket()
      """

    # Receive data from the socket
    try:
        with open('config.json') as json_file:
            config = json.load(json_file)
    except:
        print("config.json not found")
        exit()
    socket.listen(1)  # Allow 1 failed connection
    conn, addr = socket.accept()
    st.session_state["class_stats"] = [0] * (config['num_of_classes'] + 1)
    print("Connected by", addr)
    while True:
        try:
            length_bytes = conn.recv(8)
            total_length = int.from_bytes(length_bytes, 'big')
            # Then receive the actual data
            buffer = b''
            while len(buffer) < total_length:
                packet = conn.recv(4096)
                if not packet:
                    break
                buffer += packet
            new_stats = pickle.loads(buffer)
            print(new_stats)
            st.session_state["class_stats"] = new_stats
        except:
            print('Disconnecting')
            disconnect_socket(conn)
            conn, addr = socket.accept()
            print("Connected by", addr)
    return


def disconnect_socket(conn):
    """
      Just close the connection

      :param conn: connection instance
      """
    conn.close()


def connect_sockets():
    """
      Initialize a server socket as a new thread and wait for connections

      :return: [socket, listening_thread] instances
      """
    try:
        with open('config.json') as json_file:
            config = json.load(json_file)
    except:
        print("config.json not found")
        exit()

    run_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    run_socket.bind((config['host_address'], config['sbto_run_port']))
    run_socket.listen(1)
    conn, addr = run_socket.accept()
    st.session_state['sbto_run_socket'] = conn
    ping_thread = Thread(target=ping_run_socket)
    get_script_run_ctx(ping_thread)
    add_script_run_ctx(ping_thread)
    ping_thread.daemon = True
    ping_thread.start()

    prod_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    prod_socket.connect((config["host_address"], config['production_port']))
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((config['host_address'], config['streamlit_port']))
    live_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    live_socket.bind((config['host_address'], config['production_live_port']))
    sbto_live_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sbto_live_socket.bind((config['host_address'], config['sbto_live_port']))
    prediction_live_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    prediction_live_socket.bind((config['host_address'], config['prediction_live_port']))
    listen_thread_server = Thread(target=socket_listener_sbto, args=(server_socket,))
    get_script_run_ctx(listen_thread_server)
    add_script_run_ctx(listen_thread_server)
    listen_thread_server.daemon = False
    listen_thread_server.start()
    print('sbto thread started')
    listen_thread_live_sbto = Thread(target=socket_listener_live_sbto, args=(sbto_live_socket,))
    get_script_run_ctx(listen_thread_live_sbto)
    add_script_run_ctx(listen_thread_live_sbto)
    listen_thread_live_sbto.daemon = False
    listen_thread_live_sbto.start()
    listen_thread_live = Thread(target=socket_listener_live, args=(live_socket,))
    get_script_run_ctx(listen_thread_live)
    add_script_run_ctx(listen_thread_live)
    listen_thread_live.daemon = True
    listen_thread_live.start()
    listen_thread_live_prediction = Thread(target=socket_listener_live_prediction, args=(prediction_live_socket,))
    get_script_run_ctx(listen_thread_live_prediction)
    add_script_run_ctx(listen_thread_live_prediction)
    listen_thread_live_prediction.daemon = True
    listen_thread_live_prediction.start()
    st.session_state['prod_socket'] = prod_socket
    st.session_state['live_socket'] = live_socket
    return prod_socket


def ping_run_socket():
    while True:
        try:
            st.session_state['sbto_run_socket'].sendall(str("p").encode())
            time.sleep(30)
        except Exception as error:
            print("An exception occurred live socket", error)
            print('Run Socket ping failure')


# @st.experimental_fragment(run_every='1s')
# def draw_live():
#   if 'live_metrics' in st.session_state:
#     placeholder = st.empty()
#     with placeholder.container():
#       while True:
#         try:
#           col1, col2, col3 = st.columns(3)
#           break
#         except:
#           print("Retrying drawing live data")
#       with col1:
#         st.write("Training Accuracy")
#         st.line_chart(st.session_state['live_metrics']['Acc'], use_container_width=True, height=200)
#       with col2:
#         st.write("Training Loss")
#         st.line_chart(st.session_state['live_metrics']['Loss'], use_container_width=True, height=200)
#       with col3:
#         st.write("Epoch Duration (s)")
#         st.line_chart(st.session_state['live_metrics']['Time'], use_container_width=True, height=200)


@st.fragment(run_every='1s')
def draw_live1():
    if 'live_metrics' in st.session_state:
        placeholder = st.empty()
        with placeholder.container():
            st.write("Accuracy")
            st.line_chart(st.session_state['live_metrics']['Acc'], use_container_width=True, height=200,
                          color='#0000FF')


@st.fragment(run_every='1s')
def draw_live2():
    if 'live_metrics' in st.session_state:
        placeholder = st.empty()
        with placeholder.container():
            st.write("Loss")
            st.line_chart(st.session_state['live_metrics']['Loss'], use_container_width=True, height=200,
                          color='#FF0000')


@st.fragment(run_every='1s')
def draw_live3():
    if 'live_metrics' in st.session_state:
        placeholder = st.empty()
        with placeholder.container():
            st.write("Epoch Duration (s)")
            st.line_chart(st.session_state['live_metrics']['Time'], use_container_width=True, height=200,
                          color='#378805')


@st.fragment(run_every='1s')
def show_live_est():
    if 'live_metrics' in st.session_state:
        placeholder = st.empty()
        with placeholder.container():
            try:
                st.metric("Estimated Latency (s)",
                          '{0:.2f}'.format(st.session_state['live_metrics']['Est_Tr_Time'].iloc[-1]))
            except:
                pass


@st.fragment(run_every='2s')
def draw_live_preds():
    if st.session_state['subito_running'] and st.session_state['terminal_invoked']:
        st.session_state['terminal_invoked'] = False
        st.rerun()
    if 'class_stats' in st.session_state:
        placeholder = st.empty()
        with placeholder.container():
            st.write("Prediction Statistics")
            cols = st.columns([0.7, 0.3])
            with cols[0]:
                st.bar_chart(st.session_state['class_stats'][0:-1], x_label='Classes', y_label='Count')
            with cols[1]:
                st.metric("Inference Throughput (data points/s)",
                          '{0:.2f}'.format(st.session_state['class_stats'][-1]))


def conditional_decorator(dec, condition):
    def decorator(func):
        if not condition:
            # Return the function unchanged, not decorated.
            return func
        return dec(func)

    return decorator


if 'initialized' not in st.session_state or not st.session_state.initialized:
    st.session_state['subito_running'] = False
    st.session_state['terminal_invoked'] = False
    st.session_state.run_signal = False


def send_sbto_run_signal():
    print("Trying Sending SBTO signal")
    if st.session_state.run_signal and st.session_state.initialized:
        try:
            print("Sending SBTO signal")
            st.session_state['sbto_run_socket'].sendall(str("start").encode())
            st.session_state.run_signal = False
            return
        except ConnectionRefusedError:
            print("Could not connect to the sbto_run_socket. Make sure it is running.")
            return
        except:
            print("Could not send SBTO signal. Try to reconnect.")


@conditional_decorator(st.fragment(run_every="2s"), st.session_state['subito_running'] is True)
def display_output():
    placeholder = st.empty()
    with placeholder.container():
        stx.scrollableTextbox(st.session_state['stdout'], height=1000)
        if st.session_state['subito_running'] is False:
            print("rerunning ALERTSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSS")
            st.rerun()


def create_pareto_line_df(pareto_df):
    # Set the appropriate mask, i.e., max the accuracy, minimize the training speed
    mask = paretoset.paretoset(pareto_df, sense=["max", "min"])
    # Apply the mask and get the result
    paretodf = pareto_df[mask].copy()
    # Create a scatter plot
    paretodf.reset_index(inplace=True, drop=True)
    initial_cand_i = paretodf['Accuracy'].idxmax()
    initial_cand = paretodf.iloc[initial_cand_i]
    pareto_line_df = paretodf.loc[[initial_cand_i], ['Accuracy', 'Training Speed (sec)']].copy()
    paretodf.drop(initial_cand_i, inplace=True)
    while not paretodf.empty:
        min_neighbour_cand_norm = float('inf')  # Set a high initial value
        min_neighbour_cand_i = -1  # Initialize the index
        for next_i in paretodf.index:
            norm = np.linalg.norm(paretodf.loc[next_i] - initial_cand)
            if norm < min_neighbour_cand_norm:
                min_neighbour_cand_i = next_i
                min_neighbour_cand_norm = norm
        pareto_line_df = pd.concat(
            [pareto_line_df, paretodf.loc[[min_neighbour_cand_i], ['Accuracy', 'Training Speed (sec)']]],
            ignore_index=True,
            axis=0)
        initial_cand = paretodf.loc[min_neighbour_cand_i]
        paretodf.drop(min_neighbour_cand_i, inplace=True)
    return pareto_line_df


try:
    with open('config.json') as json_file:
        config = json.load(json_file)
except:
    print("config.json not found")
    exit()
#prod_pipe_socket = start_controller()
if 'stream_batch_train' not in st.session_state:
    st.session_state['stream_batch_train'] = config['stream_batch_train']  # Default value
if 'stream_batch_test' not in st.session_state:
    st.session_state['stream_batch_test'] = config['stream_batch_test']  # Default value
if 'num_of_conv_layers' not in st.session_state:
    st.session_state['num_of_conv_layers'] = config['num_of_conv_layers']  # Default value
if 'num_of_pool_layers' not in st.session_state:
    st.session_state['num_of_pool_layers'] = config['num_of_pool_layers']  # Default value
if 'num_of_dense_layers' not in st.session_state:
    st.session_state['num_of_dense_layers'] = config['num_of_dense_layers']  # Default value
if 'lr' not in st.session_state:
    st.session_state['lr'] = config['lr']  # Default value
if 'size_of_batch' not in st.session_state:
    st.session_state['size_of_batch'] = config['size_of_batch']  # Default value
if 'bo_data_train' not in st.session_state:
    st.session_state['bo_data_train'] = config['bo_data_train']  # Default value
if 'bo_data_test' not in st.session_state:
    st.session_state['bo_data_test'] = config['bo_data_test']  # Default value
if 'sample_size_low' not in st.session_state:
    st.session_state['sample_size_low'] = 1-config['sample_size_low']  # Default value
if 'sample_size_high' not in st.session_state:
    st.session_state['sample_size_high'] = 1-config['sample_size_high']  # Default value
if 'num_of_conv_layers_low' not in st.session_state:
    st.session_state['num_of_conv_layers_low'] = config['num_of_conv_layers_low']  # Default value
if 'num_of_conv_layers_high' not in st.session_state:
    st.session_state['num_of_conv_layers_high'] = config['num_of_conv_layers_high']  # Default value
if 'num_of_pool_layers_low' not in st.session_state:
    st.session_state['num_of_pool_layers_low'] = config['num_of_pool_layers_low']  # Default value
if 'num_of_pool_layers_high' not in st.session_state:
    st.session_state['num_of_pool_layers_high'] = config['num_of_pool_layers_high']  # Default value
if 'num_of_dense_layers_low' not in st.session_state:
    st.session_state['num_of_dense_layers_low'] = config['num_of_dense_layers_low']  # Default value
if 'num_of_dense_layers_high' not in st.session_state:
    st.session_state['num_of_dense_layers_high'] = config['num_of_dense_layers_high']  # Default value
if 'num_of_epochs_low' not in st.session_state:
    st.session_state['num_of_epochs_low'] = config['num_of_epochs_low']  # Default value
if 'num_of_epochs_high' not in st.session_state:
    st.session_state['num_of_epochs_high'] = config['num_of_epochs_high']  # Default value
if 'num_of_dask_workers_low' not in st.session_state:
    st.session_state['num_of_dask_workers_low'] = config['num_of_dask_workers_low']  # Default value
if 'num_of_dask_workers_high' not in st.session_state:
    st.session_state['num_of_dask_workers_high'] = config['num_of_dask_workers_high']  # Default value
if 'acquisition_f' not in st.session_state:
    st.session_state['acquisition_f'] = config['acquisition_f']  # Default value
if 'bo_call_number' not in st.session_state:
    st.session_state['bo_call_number'] = config['bo_call_number']  # Default value
if 'theta_parameter' not in st.session_state:
    st.session_state['theta_parameter'] = config['theta_parameter']  # Default value
if 'lamda_acc' not in st.session_state:
    st.session_state['lamda_acc'] = config['lamda_acc']  # Default value
if 'num_of_trainers' not in st.session_state:
    st.session_state['num_of_trainers'] = config['num_of_trainers']  # Default value
if 'num_of_predictors' not in st.session_state:
    st.session_state['num_of_predictors'] = config['num_of_predictors']  # Default value

if 'initialized' not in st.session_state or not st.session_state.initialized:
    st.session_state['prod_socket'] = connect_sockets()
    st.session_state.initialized = True
    st.session_state['stdout'] = ''
    print("Initialization Finished")

# Define the path to your logo image
logo_url = "subitoLogo.svg"
st.logo(logo_url)
with st.sidebar:
    st.header("Production Pipeline:")
    cols = st.columns(2)
    with cols[0]:
        stream_batch_train = st.number_input("Stream Train Size:", step=1, value=st.session_state['stream_batch_train'], key='stream_batch_train',
                                             on_change=update_config_file('stream_batch_train'))
    with cols[1]:
        stream_batch_test = st.number_input("Stream Test Size:", step=1, value=st.session_state['stream_batch_test'], key='stream_batch_test',
                                            on_change=update_config_file('stream_batch_test'))
    num_of_conv_layers = st.number_input("Default Number of Conv Layers:", step=1, value=st.session_state['num_of_conv_layers'], key='num_of_conv_layers',
                                         on_change=update_config_file('num_of_conv_layers'))
    num_of_pool_layers = st.number_input("Default Number of Pool Layers:", step=1, value=st.session_state['num_of_pool_layers'], key='num_of_pool_layers',
                                         on_change=update_config_file('num_of_pool_layers'))
    num_of_dense_layers = st.number_input("Default Number of Dense Layers:", step=1, value=st.session_state['num_of_dense_layers'], key='num_of_dense_layers',
                                          on_change=update_config_file('num_of_dense_layers'))
    cols = st.columns(2)
    with cols[0]:
        lr = st.number_input("Learning Rate:", value=st.session_state['lr'], step=0.001, format="%0.3f", key='lr',
                             on_change=update_config_file('lr'))
    with cols[1]:
        size_of_batch = st.number_input("Batch Size:", value=st.session_state['size_of_batch'], step=32, min_value=1, max_value=stream_batch_train,
                                        key='size_of_batch', on_change=update_config_file('size_of_batch'))
    cols = st.columns(2)
    with cols[0]:
        num_of_trainers = st.number_input("Trainers:", value=st.session_state['num_of_trainers'], step=1, key='num_of_trainers', min_value=1,
                                         on_change=update_config_file('num_of_trainers'))
    with cols[1]:
        num_of_predictors = st.number_input("Predictors:", value=st.session_state['num_of_predictors'], step=1, min_value=1,
                                        key='num_of_predictors', on_change=update_config_file('num_of_predictors'))
    st.header("SuBiTO:")
    cols = st.columns(2)
    with cols[0]:
        bo_data_train = st.number_input("SuBiTO Train Size:", step=1, value=st.session_state['bo_data_train'], key='bo_data_train',
                                        on_change=update_config_file('bo_data_train'))
    with cols[1]:
        bo_data_test = st.number_input("SuBiTO Test Size:", step=1, value=st.session_state['bo_data_test'], key='bo_data_test',
                                       on_change=update_config_file('bo_data_test'))

    cols = st.columns(2)
    with cols[0]:
        sample_size_high = st.number_input("Compression R. Low:", step=0.05, value=st.session_state['sample_size_high'], min_value=0.6, max_value=st.session_state['sample_size_low'],
                                           key='sample_size_high',
                                           on_change=update_config_file('sample_size_high'))
    with cols[1]:
        sample_size_low = st.number_input("Compression R. High:", step=0.05, value=st.session_state['sample_size_low'], min_value=sample_size_high, max_value=1.0,
                                          key='sample_size_low',
                                          on_change=update_config_file('sample_size_low'))
    cols = st.columns(2)
    with cols[0]:
        num_of_conv_layers_low = st.number_input('Conv Low:', step=1, value=st.session_state['num_of_conv_layers_low'], min_value=1,
                                                 key='num_of_conv_layers_low',
                                                 on_change=update_config_file('num_of_conv_layers_low'))
    with cols[1]:
        num_of_conv_layers_high = st.number_input('Conv High:', step=1, value=st.session_state['num_of_conv_layers_high'], min_value=num_of_conv_layers_low,
                                                  key='num_of_conv_layers_high',
                                                  on_change=update_config_file('num_of_conv_layers_high'))

    cols = st.columns(2)
    with cols[0]:
        num_of_pool_layers_low = st.number_input('Pool Low:', step=1, value=st.session_state['num_of_pool_layers_low'], min_value=0,
                                                 key='num_of_pool_layers_low',
                                                 on_change=update_config_file('num_of_pool_layers_low'))
    with cols[1]:
        num_of_pool_layers_high = st.number_input('Pool High:', step=1, value=st.session_state['num_of_pool_layers_high'], min_value=num_of_pool_layers_low,
                                                  key='num_of_pool_layers_high',
                                                  on_change=update_config_file('num_of_pool_layers_high'))

    cols = st.columns(2)
    with cols[0]:
        num_of_dense_layers_low = st.number_input('Dense Low:', step=1, value=st.session_state['num_of_dense_layers_low'], min_value=0,
                                                  key='num_of_dense_layers_low',
                                                  on_change=update_config_file('num_of_dense_layers_low'))
    with cols[1]:
        num_of_dense_layers_high = st.number_input('Dense High:', step=1, value=st.session_state['num_of_dense_layers_high'], min_value=num_of_dense_layers_low,
                                                   key='num_of_dense_layers_high',
                                                   on_change=update_config_file('num_of_dense_layers_high'))

    cols = st.columns(2)
    with cols[0]:
        num_of_lstm_layers_low = st.number_input('LSTM Low:', step=1, value=0, min_value=0,
                                                 key='num_of_lstm_layers_low')
    with cols[1]:
        num_of_lstm_layers_high = st.number_input('LSTM High:', step=1, value=0, min_value=0,
                                                  key='num_of_lstm_layers_high')

    cols = st.columns(2)
    with cols[0]:
        num_of_gru_layers_low = st.number_input('GRU Low:', step=1, value=0, min_value=0, key='num_of_gru_layers_low')
    with cols[1]:
        num_of_gru_layers_high = st.number_input('GRU High:', step=1, value=0, min_value=0,
                                                 key='num_of_gru_layers_high')

    cols = st.columns(2)
    with cols[0]:
        num_of_rnn_layers_low = st.number_input('RNN Low:', step=1, value=0, min_value=0, key='num_of_rnn_layers_low')
    with cols[1]:
        num_of_rnn_layers_high = st.number_input('RNN High:', step=1, value=0, min_value=0,
                                                 key='num_of_rnn_layers_high')

    cols = st.columns(2)
    with cols[0]:
        num_of_dropout_layers_low = st.number_input('Dropout Low:', step=1, value=0, min_value=0,
                                                    key='num_of_dropout_layers_low')
    with cols[1]:
        num_of_dropout_layers_high = st.number_input('Dropout High:', step=1, value=0, min_value=0,
                                                     key='num_of_dropout_layers_high')

    cols = st.columns(2)
    with cols[0]:
        num_of_epochs_low = st.number_input('Epochs Low:', step=1, value=st.session_state['num_of_epochs_low'], min_value=1, key='num_of_epochs_low',
                                            on_change=update_config_file('num_of_epochs_low'))
    with cols[1]:
        num_of_epochs_high = st.number_input('Epochs High:', step=1, value=st.session_state['num_of_epochs_high'], min_value=num_of_epochs_low,
                                             max_value=20, key='num_of_epochs_high',
                                             on_change=update_config_file('num_of_epochs_high'))
    cols = st.columns(2)
    with cols[0]:
        num_of_dask_workers_low = st.number_input('Workers Low:', step=1, value=st.session_state['num_of_dask_workers_low'], min_value=2, key='num_of_dask_workers_low',
                                            on_change=update_config_file('num_of_dask_workers_low'))
    with cols[1]:
        num_of_dask_workers_high = st.number_input('Workers High:', step=1, value=st.session_state['num_of_dask_workers_high'], min_value=num_of_dask_workers_low,
                                             max_value=20, key='num_of_dask_workers_high',
                                             on_change=update_config_file('num_of_dask_workers_high'))
    acquisition_f = st.selectbox(
        "Acquisition Function",
        ("gp_hedge", "LCB", "EI", "PI"), key='acquisition_f', on_change=update_config_file('acquisition_f'))

    bo_call_number = st.slider("Optimizer Calls:", step=1, value=st.session_state['bo_call_number'], min_value=11, max_value=50, key='bo_call_number',
                               on_change=update_config_file('bo_call_number'))
    theta_parameter = st.number_input('Theta Parameter:', step=1, value=st.session_state['theta_parameter'], key='theta_parameter',
                                      on_change=update_config_file('theta_parameter'))
    lamda_acc = st.number_input('Lambda Accuracy:', step=0.1, value=st.session_state['lamda_acc'], min_value=0.01, max_value=1.0, key='lamda_acc',
                                on_change=update_config_file('lamda_acc'))

# # Use HTML and CSS to display the logo in the bottom right corner
# st.markdown(
#   f"""
#     <style>
#     .top-right-logo {{
#         position: fixed;
#         top: 50px;
#         left: 50px;
#         width: 200px;  /* Adjust the width as needed */
#     }}
#     </style>
#     <img src="{logo_url}" class="top-right-logo">
#     """,
#   unsafe_allow_html=True
# )
sbt, prod_pipe = st.columns(2)
with sbt:
    #st.header('SuBiTO :rocket:')
    st.header('SuBiTO Optimizer')
    st.toggle(label='Run', key='run_signal', help='Click to run Optimizer', on_change=send_sbto_run_signal(),
              disabled=st.session_state['subito_running'])
    #st.button(label='Run')
    if st.session_state['subito_running'] is False:
        if 'bo_res' in st.session_state:
            create_model_images()
        cols = st.columns(3)
        with cols[0]:
            if 'bo_res' in st.session_state:
                st.metric("Compression Ratio", (100 - st.session_state['sr1']).__str__() + '%')
                st.metric("Epochs", st.session_state['ep1'].__str__())
                st.metric("Workers", st.session_state['dw1'].__str__())
            link1 = 'original0.png'
            st.image(link1,
                     use_container_width =True)
        with cols[1]:
            if 'bo_res' in st.session_state:
                st.metric("Compression Ratio", (100 - st.session_state['sr2']).__str__() + '%')
                st.metric("Epochs", st.session_state['ep2'].__str__())
                st.metric("Workers", st.session_state['dw2'].__str__())
            link2 = 'original1.png'
            st.image(link2,
                     use_container_width =True)
        with cols[2]:
            if 'bo_res' in st.session_state:
                st.metric("Compression Ratio", (100 - st.session_state['sr3']).__str__() + '%')
                st.metric("Epochs", st.session_state['ep3'].__str__())
                st.metric("Workers", st.session_state['dw3'].__str__())
            link3 = 'original2.png'
            st.image(link3,
                     use_container_width =True)
        st.markdown("""<div style="text-align: center;">
      <span style="color: #ffd166;">Conv2D▮</span>&nbsp;
      <span style="color: #ef476f;">MaxPool2D▮</span>&nbsp;
      <span style="color: #0cb0a9;">Flatten▮</span>&nbsp;
      <span style="color: #842da1;">Dense▮</span>
      </div>""", unsafe_allow_html=True)
        cols = st.columns(3)
        color_acc = 'tab:blue'
        color_loss = 'tab:red'
        plt.style.use('default')
        fs = 14
        if 'bo_res' in st.session_state:
            loss_max = 0
            loss_min = 1000
            epochs_max = 0
            for res_index in range(3):
                loss_tmp_max = max(st.session_state['bo_res']['Loss Epoch'].iloc[res_index])
                loss_tmp_min = min(st.session_state['bo_res']['Loss Epoch'].iloc[res_index])
                #epochs_max_tmp = st.session_state['bo_res']['Epochs'].iloc[res_index]
                if loss_max <= loss_tmp_max:
                    loss_max = loss_tmp_max
                if loss_min >= loss_tmp_min:
                    loss_min = loss_tmp_min
                # if epochs_max <= epochs_max_tmp:
                #   epochs_max = epochs_max_tmp
        with cols[1]:
            if 'bo_res' in st.session_state:
                loss_data = st.session_state['bo_res']['Loss Epoch'].iloc[1]
                acc_data = st.session_state['bo_res']['Acc Epoch'].iloc[1]
                df = pd.DataFrame(list(zip(loss_data, acc_data)), columns=['Loss', 'Acc'])
                epochs_scale = np.arange(1, len(loss_data) + 1)
                fig, ax1 = plt.subplots()
                fig.patch.set_alpha(0.0)
                ax1.patch.set_alpha(0.0)
                ax1.set_xlabel('Epochs', fontsize=fs)
                ax1.set_ylabel('Accuracy', color=color_acc, fontsize=fs)
                ax1.set_ylim([0, 1])
                # ax1.set_xlim([1, epochs_max])
                ax1.plot(epochs_scale, acc_data, color=color_acc, linewidth=6)
                ax1.tick_params(axis='y', labelcolor=color_acc, labelsize=fs)
                ax1.tick_params(axis='x', labelsize=fs)
                ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
                ax2 = ax1.twinx()
                ax2.set_ylabel('Loss', color=color_loss, fontsize=fs)
                ax2.set_ylim([loss_min, loss_max])
                ax2.plot(epochs_scale, loss_data, color=color_loss, linewidth=6)
                ax2.tick_params(axis='y', labelcolor=color_loss, labelsize=fs)
                st.pyplot(fig)
            on2 = st.toggle(":blue[Deploy Architecture]", key='toggle2', on_change=toggle2_changed,
                            disabled=not ('bo_res' in st.session_state) or st.session_state['manual'] == True)
            if on2:
                with st.spinner('Loading...'):
                    time.sleep(2)
        with cols[0]:
            if 'bo_res' in st.session_state:
                loss_data = st.session_state['bo_res']['Loss Epoch'].iloc[0]
                acc_data = st.session_state['bo_res']['Acc Epoch'].iloc[0]
                df = pd.DataFrame(list(zip(loss_data, acc_data)), columns=['Loss', 'Acc'])
                epochs_scale = np.arange(1, len(loss_data) + 1)
                fig, ax1 = plt.subplots()
                fig.patch.set_alpha(0.0)
                ax1.patch.set_alpha(0.0)
                ax1.set_xlabel('Epochs', fontsize=fs)
                ax1.set_ylabel('Accuracy', color=color_acc, fontsize=fs)
                ax1.set_ylim([0, 1])
                # ax1.set_xlim([1, epochs_max])
                ax1.plot(epochs_scale, acc_data, color=color_acc, linewidth=6)
                ax1.tick_params(axis='y', labelcolor=color_acc, labelsize=fs)
                ax1.tick_params(axis='x', labelsize=fs)
                ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
                ax2 = ax1.twinx()
                ax2.set_ylabel('Loss', color=color_loss, fontsize=fs)
                ax2.set_ylim([loss_min, loss_max])
                ax2.plot(epochs_scale, loss_data, color=color_loss, linewidth=6)
                ax2.tick_params(axis='y', labelcolor=color_loss, labelsize=fs)
                st.pyplot(fig)
            on1 = st.toggle(":green[Deploy Architecture]", key='toggle1', on_change=toggle1_changed,
                            disabled=not ('bo_res' in st.session_state) or st.session_state['manual'] == True)
            if on1:
                with st.spinner('Loading...'):
                    time.sleep(2)
        with cols[2]:
            if 'bo_res' in st.session_state:
                loss_data = st.session_state['bo_res']['Loss Epoch'].iloc[2]
                acc_data = st.session_state['bo_res']['Acc Epoch'].iloc[2]
                df = pd.DataFrame(list(zip(loss_data, acc_data)), columns=['Loss', 'Acc'])
                epochs_scale = np.arange(1, len(loss_data) + 1)
                fig, ax1 = plt.subplots()
                fig.patch.set_alpha(0.0)
                ax1.patch.set_alpha(0.0)
                ax1.set_xlabel('Epochs', fontsize=fs)
                ax1.set_ylabel('Accuracy', color=color_acc, fontsize=fs)
                ax1.set_ylim([0, 1])
                # ax1.set_xlim([1, epochs_max])
                ax1.plot(epochs_scale, acc_data, color=color_acc, linewidth=6)
                ax1.tick_params(axis='y', labelcolor=color_acc, labelsize=fs)
                ax1.tick_params(axis='x', labelsize=fs)
                ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
                ax2 = ax1.twinx()
                ax2.set_ylabel('Loss', color=color_loss, fontsize=fs)
                ax2.set_ylim([loss_min, loss_max])
                ax2.plot(epochs_scale, loss_data, color=color_loss, linewidth=6)
                ax2.tick_params(axis='y', labelcolor=color_loss, labelsize=fs)
                st.pyplot(fig)
                # st.image('https://www150.statcan.gc.ca/edu/power-pouvoir/c-g/c-g05-2-1-eng.png', use_column_width=True)
            on3 = st.toggle(":orange[Deploy Architecture]", key='toggle3', on_change=toggle3_changed,
                            disabled=not ('bo_res' in st.session_state) or st.session_state['manual'] == True)
            if on3:
                with st.spinner('Loading...'):
                    time.sleep(2)
        if 'bo_res' in st.session_state:
            st.write('SuBiTO Options and Pareto Optimal Solutions')
            chart_data = st.session_state['bo_res'][['Accuracy', 'Training Speed (sec)']]
            pareto_df = chart_data.copy()
            pareto_line_df = create_pareto_line_df(chart_data)
            chart_data.insert(2, 'Color', [''] * len(chart_data), True)
            chart_data.insert(2, 'Marker', [''] * len(chart_data), True)
            # chart_data['Color'] = ['']*len(chart_data)
            for i in range(len(chart_data)):
                if i == 0:
                    chart_data.at[i, 'Color'] = 'green'
                    chart_data.at[i, 'Marker'] = 'triangle'
                    chart_data.at[i, 'Marker_Size'] = 300
                elif i == 1:
                    chart_data.at[i, 'Color'] = 'blue'
                    chart_data.at[i, 'Marker'] = 'triangle'
                    chart_data.at[i, 'Marker_Size'] = 300
                elif i == 2:
                    chart_data.at[i, 'Color'] = 'orange'
                    chart_data.at[i, 'Marker'] = 'triangle'
                    chart_data.at[i, 'Marker_Size'] = 300
                else:
                    chart_data.at[i, 'Color'] = 'red'
                    chart_data.at[i, 'Marker'] = 'circle'
                    chart_data.at[i, 'Marker_Size'] = 100
            # Create a scatter plot
            scatter = alt.Chart(chart_data).mark_point(filled=True).encode(
                x=alt.X('Accuracy', title='Accuracy'),  # X-axis label for clarity
                y=alt.Y('Training Speed (sec)', title='Training Time (s)'),  # Rename y-axis label
                color=alt.Color('Color:N', scale=None),  # Use the Color column to define the color
                shape=alt.Shape('Marker', scale=None),
                size=alt.Size('Marker_Size', scale=None),
                tooltip=['Accuracy', 'Training Speed (sec)'])  # Add tooltip for better interactivity
            pareto_line = alt.Chart(pareto_line_df).mark_line(
                color='#0000ff50',
                size=3,
                point=alt.OverlayMarkDef(color='#ffffff00', filled=True)).encode(
                x='Accuracy',
                y='Training Speed (sec)',
                tooltip=['Accuracy', 'Training Speed (sec)'])
            combined_chart = scatter + pareto_line
            st.altair_chart(combined_chart.interactive(), use_container_width=True)
    else:
        # Display the terminal output
        display_output()

with prod_pipe:
    st.header('Training Pipeline')
    sbt_sel = ''
    link = 'tmp.png'
    if st.session_state['last_arch_selected'] == 1:
        st.image(link, width=450)
        if st.session_state.toggle1:
            sbt_sel = 'scen1'
            st.session_state['sr'] = st.session_state['sr1']
            st.session_state['ep'] = st.session_state['ep1']
            st.session_state['dw'] = st.session_state['dw1']
    elif st.session_state['last_arch_selected'] == 2:
        st.image(link, width=450)
        if st.session_state.toggle2:
            sbt_sel = 'scen2'
            st.session_state['sr'] = st.session_state['sr2']
            st.session_state['ep'] = st.session_state['ep2']
            st.session_state['dw'] = st.session_state['dw2']
    elif st.session_state['last_arch_selected'] == 3:
        st.image(link, width=450)
        if st.session_state.toggle3:
            sbt_sel = 'scen3'
            st.session_state['sr'] = st.session_state['sr3']
            st.session_state['ep'] = st.session_state['ep3']
            st.session_state['dw'] = st.session_state['dw3']
    else:
        st.image('default_nn.png', width=450)
    cols = st.columns(4)
    with cols[0]:
        genre = st.radio("Mode", ["SuBiTO Mode:rocket:", "Manual Mode:wrench:"], on_change=mode_select, key='mode')
    if genre == "SuBiTO Mode:rocket:":
        st.session_state['manual'] = False
        with cols[1]:
            st.metric("Compression Ratio", (100 - st.session_state['sr']).__str__() + '%')
        with cols[2]:
            st.metric("Epochs", st.session_state['ep'].__str__())
    else:
        st.session_state['manual'] = True
        with cols[1]:
            st.metric("Compression Ratio", (100 - st.session_state['sr']).__str__() + '%')
            sr = 1 - st.number_input("Compression Ratio:", step=0.05, value=0.9, min_value=0.00, max_value=1.0,
                                     on_change=disable_deploy_manual)
        with cols[2]:
            st.metric("Epochs", st.session_state['ep'].__str__())
            ep = st.number_input("Epochs:", step=1, value=2, min_value=1, max_value=50, on_change=disable_deploy_manual)
        with cols[2]:
            st.session_state['sr_manual'] = sr
            st.session_state['ep_manual'] = ep
            st.toggle(":gray[Deploy]", key='toggle_manual', on_change=toggle_manual_changed,
                      help='Click to deploy manual settings')
    with cols[3]:
        show_live_est()
    st.markdown("Training Metrics")
    # if 'live_metrics' in st.session_state:
    # placeholder = st.empty()
    # for seconds in range(200):
    #   with placeholder.container():
    #     col1, col2, col3 = st.columns(3)
    #     with col1:
    #       st.write("Training Accuracy")
    #       st.line_chart(st.session_state['live_metrics']['Acc'], use_container_width=True)
    #     with col2:
    #       st.write("Training Loss")
    #       st.line_chart(st.session_state['live_metrics']['Loss'], use_container_width=True)
    #     with col3:
    #       st.write("Epoch Duration (s)")
    #       st.line_chart(st.session_state['live_metrics']['Time'], use_container_width=True)
    #     time.sleep(1)
    cols = st.columns(3)
    with cols[0]:
        draw_live1()
    with cols[1]:
        draw_live2()
    with cols[2]:
        draw_live3()
    st.header('Prediction Pipeline')
    draw_live_preds()
