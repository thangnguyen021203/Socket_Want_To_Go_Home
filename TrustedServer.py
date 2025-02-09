import socket
from config import CONFIG
from utils.communication import receive_message, send_message
from federated.cipher_utils import generate_prime, random_number, aes_ctr_prg


class TrustedServer:
    def __init__(self):
        self.TrustedServer_host = CONFIG["trusted_server_host"]
        self.TrustedServer_port = CONFIG["trusted_server_port"]
        self.TrustedServer_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.TrustedServer_socket.bind((self.TrustedServer_host, self.TrustedServer_port))
        self.TrustedServer_socket.listen(CONFIG["num_clients"])

        self.server_host = CONFIG["server_host"]
        self.server_port = CONFIG["server_port"]

        self.clients = {}
        self.clients_active = {}
        self.g = random_number()
        self.p1 = generate_prime(10) #sinh số nguyên tố ngẫu nhiên với số bit cho trước

    
    def listen_clients(self):
        """Lắng nghe các client REGIST."""
        print("Listening for clients...")
        while True:
            conn, addr = self.TrustedServer_socket.accept()
            # print(f"Connection from {addr}")
            
            # nhận client info regist từ client
            try:
                client_info = receive_message(conn)
                client_id, (client_host, client_port) = list(client_info.items())[0]
                self.clients[client_id] = (client_host, client_port)
                print(f"Client {client_id} registered from {(client_host, client_port)}")
            except Exception as e:
                print(f"Error receiving client info: {e}")
            
            # gửi public g và p1 cho client
            try:
                send_message(conn, (self.g,self.p1))
            except Exception as e:
                print(f"Error send g and p1 to client {client_id}")

            conn.close()

    
    def ping_clients(self):
        """Ping tất cả client để kiểm tra kết nối. Loại bỏ client mất kết nối."""
        print(self.clients)
        for client_id, (client_host, client_port) in self.clients.items():
            # print((client_host, client_port))
            try:
                # print("Vô đây chưa?")
                client_conn = socket.create_connection((client_host, client_port)) 
                # print(;"Chưa create connection được")
                send_message(client_conn, "PING")
                print("Sent PING")
                response = receive_message(client_conn)
                if not public:
                    print(f"Client {client_id} not respone Ping")
                else: 
                    public = response
                    self.clients_active[client_id] = (client_host, client_port, public)
                client_conn.close()
            except:
                print("Something wrong when connect to client to Ping")
        # Loại bỏ client mất kết nối
        print(f"Active clients after ping: {list(self.clients_active.keys())}")
    

    def sendServer_ClientsActive(self):
        """
        Gửi thông tin client đã chọn cho server (client_id, client_host, client_port)
        """

        server_conn = socket.create_connection((self.server_host, self.server_port))

        list_chosen_clients = []
        for client_id in self.clients_active:
            client_host, client_port, public = self.clients_active[client_id]
            list_chosen_clients.append((client_id, client_host, client_port))

        send_message(server_conn, list_chosen_clients)
        print("Send list chosen clients to Server")

        server_conn.close()
    
    def sendClient_ClientsActive(self):
        """
        Gửi danh sách client active cho từng client
        """