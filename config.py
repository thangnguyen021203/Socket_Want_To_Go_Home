import socket

CONFIG = {
    "num_clients": 20,
    "batch_size": 32,
    "epochs": 20,
    "lr": 0.01,
    "dataset": "MNIST",
    "use_secret_sharing": True,  # Sử dụng PySyft để bảo mật
    "rounds": 2,
    "convergence_threshold": 0.01,
    "num_client_training": 2,
    "server_host": "localhost",
    "server_port": 5555,
    "trusted_server_host": "localhost",
    "trusted_server_port": 5556
}
