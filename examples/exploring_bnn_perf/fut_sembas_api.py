import socket
import struct
from time import sleep

import torch


def wait_until_open(
    client: socket.socket, max_attempts: int | None = 10, delay: float = 0.1
):
    success = False
    i = 0
    while not success and (max_attempts is None or i < max_attempts):
        try:
            client.connect(("127.0.0.1", 2000))
            success = True
        except:
            sleep(delay)

        i += 1


def setup_socket(ndim: int, fail_on_refuse=False):
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    if fail_on_refuse:
        client.connect(("127.0.0.1", 2000))
    else:
        wait_until_open(client)

    # send config to remote classifier
    try:
        ndim_packed = struct.pack("!q", ndim)
        client.sendall(ndim_packed)

        msg = client.recv(1024).decode("utf-8")
        print(f"Received msg: '{msg}'")
        if msg != "OK\n":
            raise Exception("Invalid number of dimensions?")
    except Exception as e:
        client.close()
        raise e

    return client


def receive_request(client: socket.socket, ndim: int) -> torch.Tensor:
    data_size = ndim * 8  # ndim * size(f64)
    data = client.recv(data_size)
    return torch.tensor(struct.unpack(f"{ndim}d", data))


def send_response(client: socket.socket, cls: bool):
    bool_byte = int(cls).to_bytes(1, byteorder="big")
    client.sendall(bool_byte)
