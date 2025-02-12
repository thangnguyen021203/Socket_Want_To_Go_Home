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


class Server:
    def __init__(self, global_model, test_dataloader=None):
        self.global_model = global_model

        self.test_dataloader = test_dataloader

        self.clients_active = {}

        self.server_host = CONFIG["server_host"]
        self.server_port = CONFIG["server_port"]
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind((self.server_host, self.server_port))
        self.server_socket.listen(CONFIG["num_clients"])
    
    def recieve_activeClients(self):
        """
        Nhận danh sách clients actrive từ Trusted Server
        """
        try:
            server_conn, _ = self.server_socket.accept()
            self.clients_active = receive_message(server_conn)
        except:
            print("Fail to recieve clients active.")
    
    def train(self):
        """
        Gửi global model, nhận lại local model, tổng hợp và gửi model đã tổng hợp.
        """
        connections = {}
        client_updates = []
        for client_id, client_host, client_port in self.clients_active:
            try:
                client_conn = socket.create_connection((client_host, client_port))
                connections[client_id] = client_conn
                send_message(client_conn, self.global_model.state_dict())
            except:
                print(f"Fail to connect to client {client_id} to send global model.")
        
        for client_id in connections:
            try:
                local_model, num_data = receive_message(connections[client_id])
                client_updates.append((local_model, num_data))
            except:
                print(f"Fail to recieve local model from client {client_id}")

        self.global_model.load_state_dict(aggregate(client_updates))

        for client_id in connections:
            try:
                send_message(connections[client_id], self.global_model.state_dict())
                connections[client_id].close()
            except:
                print(f"Fail to send aggregate param to client {client_id}")

    
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


    def start(self):
        """Bắt đầu server."""
        print(f"Server started on {self.host}:{self.port}")

        loss_history = []
        accuracy_history = []
        ######################################  
        while input("Press Enter to start Federated Learning...\n") != "exit":
           # Nhận danh sách clients_active từ Trusted Server
            self.recieve_activeClients()

            # bắt đầu quá trình train FL
            self.train()

            # đánh giá model
            loss, accuracy = self.evaluate()
            loss_history.append(loss)
            accuracy_history.append(accuracy)
            print(f"Server evaluation: Loss = {loss:.4f}, Accuracy = {accuracy:.2f}%")
        plot_metrics(loss_history, accuracy_history)



if __name__ == "__main__":
    args = args_parser()
    _, test_dataset = choose_dataset(args.dataset)
    test_dataloader = get_test_dataloader(test_dataset, 
                                          CONFIG["batch_size"])

    server = Server(CNNModel(),test_dataloader=test_dataloader)
    server.start()