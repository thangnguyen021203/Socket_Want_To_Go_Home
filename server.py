import torch.nn.functional as F
import torch
import socket
import pickle
import threading
from utils.data_utils import get_test_dataloader, choose_dataset
from utils.parser import args_parser
from utils.communication import send_message, receive_message
from utils.plot import plot_metrics
from federated.cipher_utils import generate_prime, random_number, aes_ctr_prg
from models.cnn_model import CNNModel
from config import CONFIG
from federated.fedavg import aggregate


#Server có 2 thread 
# 1 thread để nghe các client regist
# 1 thread để ping và thực thi FL
class Server:
    def __init__(self, global_model, test_dataloader=None):
        self.global_model = global_model
        self.test_dataloader = test_dataloader
        self.clients = {}
        self.client_active = {}
        self.host = CONFIG["server_host"]
        self.port = CONFIG["server_port"]
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(CONFIG["num_clients"])
    
    def ping_clients(self):
        """Ping tất cả client để kiểm tra kết nối. Loại bỏ client mất kết nối."""
        print(self.clients)
        disconnected_clients = []
        for client_id, (client_host, client_port) in self.clients.items():
            # print((client_host, client_port))
            try:
                # print("Vô đây chưa?")
                client_conn = socket.create_connection((client_host, client_port)) 
                # print(;"Chưa create connection được")
                send_message(client_conn, "PING")
                print("Sent PING")
                response = receive_message(client_conn)
                if response != "PONG":
                    disconnected_clients.append(client_id)
                client_conn.close()
            except:
                disconnected_clients.append(client_id)
        # print(disconnected_clients)
        self.client_active = self.clients.copy()
        # Loại bỏ client mất kết nối
        for client_id in disconnected_clients:
            del self.client_active[client_id]
        print(f"Active clients after ping: {list(self.client_active.keys())}")
    
    def setup(self):
        """
        Server tạo ngẫu nhiên số g và tạo ngẫu nhiên số nguyên tố p1.
        Server gửi danh sách thông tin client_active và g&p1 cho từng client đang active.
        """
        g = random_number()
        p1 = generate_prime(random_number())

        for client in self.client_active:
            client_ip, client_port = self.client_active[client]
            client_conn = socket.create_connection((client_ip, client_port))
            send_message(client_conn, (self.client_active, g, p1))
            client_conn.close()
        
        




    def send_model(self, model):
        """Gửi mô hình đến client để huấn luyện."""
        client_conns = []
        for client_id in self.client_active:
            client_ip, client_port = self.client_active[client_id] 

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

    def listen_clients(self):
        """Lắng nghe các client REGIST."""
        print("Listening for clients...")
        while True:
            conn, addr = self.server_socket.accept()
            # print(f"Connection from {addr}")

            try:
                message = receive_message(conn)
                # print(message)
                if message == "REGIST":
                    client_info = receive_message(conn)
                    client_id, (client_host, client_port) = list(client_info.items())[0]
                    self.clients[client_id] = (client_host, client_port)
                    print(f"Client {client_id} registered from {(client_host, client_port)}")

            except Exception as e:
                print(f"Error receiving client data: {e}")
            conn.close()


    def start(self):
        """Bắt đầu server."""
        print(f"Server started on {self.host}:{self.port}")

        threading.Thread(target=self.listen_clients, daemon=True).start()

        loss_history = []
        accuracy_history = []
        ######################################
        while input("Press Enter to start Federated Learning...\n") != "exit":
            # Ping clients xác định clients nào còn active trước khi thực hiện train
            self.ping_clients()

            # Server gửi thông tin client cho các client và truyền thêm tham số chung g và p1 cho mỗi client
            


            # Gửi mô hình cho các client và nhận các bản cập nhật từ chúng
            client_conns = self.send_model(self.global_model.state_dict())
            client_updates = self.receive_model_updates(client_conns)
            
            # Tổng hợp các bản cập nhật từ client và cập nhật mô hình toàn cục
            updated_model = self.update_model(client_updates)
            
            # Cập nhật mô hình toàn cục và đánh giá
            self.global_model.load_state_dict(updated_model)
            loss, accuracy = self.evaluate()
            loss_history.append(loss)
            accuracy_history.append(accuracy)
            print(f"Server evaluation: Loss = {loss:.4f}, Accuracy = {accuracy:.2f}%")
        
        plot_metrics(loss_history, accuracy_history)

if __name__ == "__main__":
    args = args_parser()
    train_dataset, test_dataset = choose_dataset(args.dataset)
    test_dataloader = get_test_dataloader(test_dataset, 
                                          CONFIG["batch_size"])

    server = Server(CNNModel(),test_dataloader=test_dataloader)
    server.start()