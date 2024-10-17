import socket
import struct
import numpy as np

from numpy import ndarray


def setup_socket(n: int):
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect(("127.0.0.1", 2000))
    # send config to remote classifier
    try:
        ndim_packed = struct.pack("!q", n)
        client.sendall(ndim_packed)

        msg = client.recv(1024).decode("utf-8")
        print(f"Received msg: '{msg}'")
        if msg != "OK\n":
            raise Exception("Invalid number of dimensions?")
    except Exception as e:
        client.close()
        raise e

    return client


def receive_request(client: socket.socket, n: int) -> ndarray:
    data_size = n * 8
    data = client.recv(data_size)
    return np.array(struct.unpack(f"{n}d", data))


def send_response(client: socket.socket, cls: bool):
    bool_byte = int(cls).to_bytes(1, byteorder="big")
    client.sendall(bool_byte)
