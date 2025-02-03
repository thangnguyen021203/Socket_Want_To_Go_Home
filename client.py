import torch
import socket
from utils.parser import args_parser
from models.cnn_model import CNNModel
from utils.data_utils import get_dataloaders, choose_dataset
from utils.communication import send_message, receive_message
from config import CONFIG

class Client:
    def __init__(self, model, dataset, client_id, client_host = "localhost", client_port = 0):
        self.model = model
        self.dataset = dataset
        self.client_id = client_id
        self.client_host = client_host
        self.client_port = client_port
        self.server_host = CONFIG["server_host"]
        self.server_port = CONFIG["server_port"]
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client_socket.bind((self.client_host, self.client_port))
        self.client_port = self.client_socket.getsockname()[1]  # Lấy port mà client đang sử dụng
        self.client_socket.listen(1)
        

    def set_model(self, model):
        """Cập nhật mô hình cho client từ server."""
        self.model.load_state_dict(model)

    def train(self):
        """
        Huấn luyện mô hình trên dữ liệu của client.
        """
        self.model.train()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=CONFIG["lr"])
        loss_fn = torch.nn.CrossEntropyLoss()

        for epoch in range(CONFIG["epochs"]):
            # print(self.dataset.dataset)
            loss_epoch = 0
            for data, target in self.dataset:
                optimizer.zero_grad()
                output = self.model(data)
                loss_epoch = loss_fn(output, target)
                loss_epoch.backward()
                optimizer.step()
            print(f"Epoch {epoch+1}/{CONFIG['epochs']} - Loss: {loss_epoch.item()}")
        # Trả về trạng thái của mô hình sau khi huấn luyện
        return self.model.state_dict()
    
    def send_to_server(self, conn, message):
        """
        Gửi message tới server.
        """
        try:
            send_message(conn, message)
        except Exception as e:
            print(f"Client {self.client_id} failed to send message to server.")
            
    
    def receive_model_from_server(self, conn):
        """
        Nhận mô hình từ server.
        """
        try:
            model = receive_message(conn)
        except Exception as e:
            print(f"Client {self.client_id} failed to receive model from server.")
            return None
        return model
    
    def connect_to_server(self): 
        """
        Kết nối tới server.
        """
        try:
            server_socket = socket.create_connection((self.server_host, self.server_port), timeout=2)
        except Exception as e:
            print(f"Client {self.client_id} failed to connect to server.")
            return None
        return server_socket
    
    def disconnect_from_server(self, conn):
        """
        Ngắt kết nối tới server.
        """
        try:
            conn.close()
        except Exception as e:
            print(f"Client {self.client_id} failed to disconnect from server.")
            return False
        return True
    
    def start(self):
        """
        Bắt đầu client.
        """
        
        """
        Đăng ký client với server.
        """
        server_conn = self.connect_to_server()
        if server_conn:
            print("Connected to server.")
            self.send_to_server(server_conn,"REGIST")
            # print("Sent Regist")
            self.send_to_server(server_conn,{self.client_id: (self.client_host, self.client_port)})
            print("Sent client info")
            server_conn.close()
        
        """
        Server ping.
        """
        # print("Truoc khi tạo connect ping")
        client_conn, addr = self.client_socket.accept()
        # print("Sau khi tạo connect ping")
        
        try:
            data = receive_message(client_conn)
            print(f"Client {self.client_id} received: {data}")
        except Exception as e:
            print(f"Error receiving message on Client {self.client_id}: {e}")
        print(data)
        if data == "PING":
            self.send_to_server(client_conn,"PONG")
            # print("Pong sent")
        else:
            self.send_to_server(client_conn,"ERROR")
            print("Something went wrong with the server.")
        client_conn.close()
        client_conn, addr = self.client_socket.accept()
        self.set_model(self.receive_model_from_server(client_conn))
        self.train()
        self.send_to_server(client_conn,self.model.state_dict())
        self.disconnect_from_server(client_conn)
            
        print(f"Client {self.client_id} finished training and disconnected.")

if __name__ == "__main__":
    args = args_parser()
    train_dataset, test_dataset = choose_dataset(args.dataset)
    train_dataloader = get_dataloaders(train_dataset, 
                                      CONFIG["num_clients"], 
                                      CONFIG["batch_size"], 
                                      args.iid)
    client = Client(client_id=args.client_id, model=CNNModel(),dataset= train_dataloader[args.client_id])
    client.start()