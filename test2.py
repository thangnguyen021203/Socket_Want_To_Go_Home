import torch
import torch.nn as nn
import socket
import pickle

# ✅ Bước 1: Nhận dữ liệu từ server
client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect(("localhost", 5555))

received_data = b""
while True:
    packet = client.recv(4096)  
    if not packet:
        break
    received_data += packet

client.close()

# ✅ Bước 2: Giải mã dữ liệu
model_code, state_dict = pickle.loads(received_data)

# ✅ Bước 3: Thực thi code nhận được
exec(model_code, globals())

# ✅ Bước 4: Tìm class con của BaseModel
model_class = None
for obj in globals().values():
    if isinstance(obj, type) and issubclass(obj, nn.Module) and obj != nn.Module:
        if issubclass(obj, globals()["BaseModel"]) and obj != globals()["BaseModel"]:
            model_class = obj
            break

if model_class is None:
    raise ValueError("Không tìm thấy class nào kế thừa từ BaseModel!")

# ✅ Bước 5: Khởi tạo model
model = model_class()
model.load_state_dict(state_dict)
model.train()

print(f"✅ Model [{model_class.__name__}] đã được khôi phục và sẵn sàng huấn luyện!")
