import torch.nn.functional as F
from federated.fedavg import aggregate
import torch
import socket
import pickle
import threading
from utils.data_utils import get_dataloaders, get_test_dataloader, choose_dataset
from utils.parser import args_parser
from utils.communication import send_message, receive_message
from models.cnn_model import CNNModel
from config import CONFIG

#Server có 2 thread 
# 1 thread để nghe các client regist
# 1 thread để ping và thực thi FL
class Server:
    def __init__(self, global_model, test_dataloader=None):
        self.global_model = global_model
        self.test_dataloader = test_dataloader
        self.clients = {}
        self.clients_connect = {}
        self.host = CONFIG["server_host"]
        self.port = CONFIG["server_port"]
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(CONFIG["num_clients"])
        self.conn = None

    def update_model(self, client_updates):
        """
        Gọi hàm aggregate để tổng hợp mô hình toàn cục.
        """
        return aggregate(self.global_model, client_updates)

    def evaluate(self):
        """
        Đánh giá mô hình toàn cục.
        """
        self.global_model.eval()
        total_loss = 0
        correct = 0
        total = 0

        # Giả sử dữ liệu validation có sẵn trên một worker (hoặc tập trung)
        for data, labels in self.test_dataloader:
            output = self.global_model(data)
            loss = F.cross_entropy(output, labels)
            total_loss = total_loss + loss.item()
            _, predicted = torch.max(output, 1)
            correct = correct + (predicted == labels).sum().item()
            total = total + labels.size(0)

        total_loss = total_loss / total
        accuracy = 100.0 * correct / total
        return loss, accuracy
    
    def ping_clients(self):
        """Ping tất cả client để kiểm tra kết nối. Loại bỏ client mất kết nối."""
        print(self.clients)
        disconnected_clients = []
        for client_id, (client_host, client_port) in self.clients.items():
            print((client_host, client_port))
            try:
                # print("Vô đây chưa?")
                client_conn = socket.create_connection((client_host, client_port)) 
                # print("Chưa create connection được")
                send_message(client_conn, "PING")
                print("Sent PING")
                response = receive_message(client_conn)
                if response != "PONG":
                    disconnected_clients.append(client_id)
                client_conn.close()
            except:
                disconnected_clients.append(client_id)
        print(disconnected_clients)
        self.clients_connect = self.clients.copy()
        # Loại bỏ client mất kết nối
        for client_id in disconnected_clients:
            del self.clients_connect[client_id]
        print(f"Active clients after ping: {list(self.clients_connect.keys())}")
    
    def send_model(self, model):
        """Gửi mô hình đến client để huấn luyện."""
        client_conns = []
        for client_id in self.clients_connect:
            client_ip, client_port = self.clients_connect[client_id] 

            try:
                client_conn = socket.create_connection((client_ip, client_port)) 
                send_message(client_conn, model)
                print(f"Sent model to {client_id}")
                client_conns.append(client_conn)
            except:
                print(f"Failed to send model to {client_id}")
        return client_conns


    def receive_model_updates(self, client_conns):
        """Nhận tham số mô hình từ client."""
        aggregated_params = []
        for client_conn in client_conns:
            model_params = receive_message(client_conn)
            try:
                aggregated_params.append(model_params)
            except pickle.UnpicklingError:
                print(f"Failed to unpickle model params from client.")
            client_conn.close()
        return aggregated_params
        # for _ in range(len(self.clients)): 
        #     # self.conn, addr = self.server_socket.accept()
        #     model_params = receive_message(conn)
        #     try:
        #         aggregated_params.append(model_params)
        #     except pickle.UnpicklingError:
        #         print(f"Failed to unpickle model params from client.")

        #     conn.close()
        # return aggregated_params    


    def listen_clients(self):
        """Lắng nghe các client."""
        print("Listening for clients...")
        while True:
            self.conn, addr = self.server_socket.accept()
            print(f"Connection from {addr}")

            try:
                message = receive_message(self.conn)
                print(message)
                if message == "REGIST":
                    client_info = receive_message(self.conn)
                    client_id, (client_host, client_port) = list(client_info.items())[0]
                    self.clients[client_id] = (client_host, client_port)
                    print(f"Client {client_id} registered from {(client_host, client_port)}")

            except Exception as e:
                print(f"Error receiving client data: {e}")


    def start(self):
        """Bắt đầu server."""
        print(f"Server started on {self.host}:{self.port}")

        threading.Thread(target=self.listen_clients, daemon=True).start()

        while True:
            input("Press Enter to start Federated Learning...\n")
            self.ping_clients()
            client_conns = self.send_model(self.global_model.state_dict())
            client_updates = self.receive_model_updates(client_conns)
            updated_model = self.update_model(client_updates)
            self.global_model.load_state_dict(updated_model)
            loss, accuracy = self.evaluate()
            print(f"Server evaluation: Loss = {loss:.4f}, Accuracy = {accuracy:.2f}%")

if __name__ == "__main__":
    args = args_parser()
    train_dataset, test_dataset = choose_dataset(args.dataset)
    # train_dataloader = get_dataloaders(train_dataset, 
    #                                   CONFIG["num_clients"], 
    #                                   CONFIG["batch_size"], 
    #                                   args.iid)
    test_dataloader = get_test_dataloader(test_dataset, 
                                          CONFIG["batch_size"])

    server = Server(CNNModel(),test_dataloader=test_dataloader)
    server.start()