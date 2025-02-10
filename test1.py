import torch
import pickle
import socket
from models.base_model import CNNModel


# 🔹 Code của model (không cần gửi tên class)
model_code = """
import torch.nn as nn
import torch.nn.functional as F

class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()

    def forward(self, x):
        raise NotImplementedError("Must be implemented in subclass")

class CNNModel(BaseModel):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
"""

# 🔹 Khởi tạo model
model = CNNModel()
state_dict = model.state_dict()

# 🔹 Đóng gói và gửi qua socket
data = pickle.dumps((model_code, state_dict))

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind(("localhost", 5555))
server.listen(1)

conn, addr = server.accept()
conn.sendall(data)
conn.close()
server.close()
