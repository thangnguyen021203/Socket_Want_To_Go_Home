import torch
import socket
from utils.parser import args_parser
from models.cnn_model import CNNModel
from utils.data_utils import get_dataloaders, choose_dataset
from utils.communication import send_message, receive_message
from models.utils import compare_state_dicts, modify_state_dicts
from federated.cipher_utils import generate_prime, random_number, aes_ctr_prg
from config import CONFIG

class Client:
    def __init__(self, model, dataset, client_id, client_host = "localhost", client_port = 0):
        self.model = model
        self.dataset = dataset
        self.dataset_size = len(dataset.dataset)

        self.client_id = client_id
        self.client_host = client_host
        self.client_port = client_port
        
        self.g = None
        self.p1 = None
        self.private = random_number()
        self.pair_private = random_number()
        self.public = None
        self.pair_PRG = None
        self.self_PRG = None

        self.neighbors = None

        self.server_host = CONFIG["server_host"]
        self.server_port = CONFIG["server_port"]
        self.TrustedServer_host = CONFIG["trusted_server_host"]
        self.TrustedServer_port = CONFIG["trusted_server_port"]
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client_socket.bind((self.client_host, self.client_port))
        self.client_port = self.client_socket.getsockname()[1]  # Lấy port mà client đang sử dụng
        self.client_socket.listen(1)
        

    def set_model(self, model):
        """Cập nhật mô hình cho client từ server."""
        self.model.load_state_dict(model)

    def train_local(self):
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
    
    def regist(self):
        """
        Client đăng kí thông tin với Trusted Server.
        """
        TrustedServer_conn = socket.create_connection((self.TrustedServer_host,self.TrustedServer_port))
        try:
            send_message(TrustedServer_conn, {self.client_id: (self.client_host, self.client_port)})
            print("Sent info to Trusted Server.")
            self.g, self.p1 = receive_message(TrustedServer_conn)
            print("Receiving g and p1")
            TrustedServer_conn.close()
            self.public = (self.g % self.p1)**(self.pair_private % self.p1)
        except:
            print(f"Regist Error.")

    def wait_ping(self):
        """
        Chờ Trusted Server ping và gửi public cho Trusted Server.
        """
        try:
            client_conn, addr = self.client_socket.accept()
            send_message(client_conn, (self.public, self.pair_PRG))
            client_conn.close()
        except: 
            print("Error in sending public to Trusted Server.")
    
    def recieve_neighbors(self):
        """
        Nhận danh sách các hàng xóm ({clientid: (client_host, client_port, public, pair_PRG)}) từ Trusted Server
        """
        try:
            client_conn, addr = self.client_socket.accept()
            self.neighbors = receive_message(client_conn)
            client_conn.close()
        except: 
            print("Error in recieving neighbors.")

    def train(self):
        """
        Nhận mô hình, train, gửi lại mô hình sau khi train.
        """
        try:
            client_conn, addr = self.client_socket.accept()
            # Nhận tham số mô hình từ server
            self.set_model(receive_message(client_conn))
            # Huấn luyện mô hình trên dữ liệu private
            self.train_local()
            # Modify tham số mô hình
            modify_state_dicts(self.model.state_dict(), self.client_id, self.neighbors, self.self_PRG)
            # Gửi local model cùng với số lượng dữ liệu
            send_message(client_conn, (self.model.state_dict(), self.dataset_size))
            # Nhận tham số mô hình sau khi tổng hợp
            updated_model = receive_message(client_conn)
            client_conn.close()
            # Trả về mô hình sau khi cập nhật để kiểm tra điều kiện vòng mới
            return updated_model
        except: 
            print("Error in sending public to Trusted Server.")
    
    def condition_NewTraining(self, updated_model):
        """
        Xem xét có nhận về đúng mô hình từ server không để bắt đầu vòng tổng hợp mới.
        """
        try:
            TrustedServer_conn = socket.create_connection((self.TrustedServer_host, self.TrustedServer_port))
            if compare_state_dicts(self.model.state_dict(), updated_model):
                send_message(TrustedServer_conn, "Done")
            else:
                send_message(TrustedServer_conn, "Server do wrong things!")
            TrustedServer_conn.close()
        except:
            print("Fail Connection to Trusted Server!")

    def start(self):
        """
        Bắt đầu client.
        """
        
        # Đăng ký client với Trusted Server.
        self.regist()
        print("Regist successfully.")
        # Chờ Trusted Server ping.
        self.wait_ping()
        # Nhận danh sách neighbor.
        self.recieve_neighbors()
        print("Recieving neighbors successfully.")
        # Bắt đầu quá trình training FL.
        updated_model = self.train()
        print("Training FL success fully")
        # Kiểm tra mô hình nhận về từ server coi có đúng không
        self.condition_NewTraining(updated_model)
            
        print(f"Client {self.client_id} finished training and disconnected.")

if __name__ == "__main__":
    args = args_parser()
    train_dataset, _ = choose_dataset(args.dataset)
    train_dataloader = get_dataloaders(train_dataset, 
                                      CONFIG["num_clients"], 
                                      CONFIG["batch_size"], 
                                      args.iid)
    client = Client(client_id=args.client_id, model=CNNModel(),dataset= train_dataloader[args.client_id])
    # print(len(train_dataloader[args.client_id].dataset))
    client.start()