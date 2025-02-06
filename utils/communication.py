import struct
import pickle

def send_message(conn, msg):
    serialized_msg = pickle.dumps(msg)
    msg_length = struct.pack('!I', len(serialized_msg))  # Đóng gói kích thước (4 byte)
    conn.sendall(msg_length + serialized_msg)  # Gửi kích thước trước, sau đó dữ liệu

def receive_message(conn):
    # Nhận đúng 4 byte đầu tiên (chứa kích thước dữ liệu)
    msg_length_data = conn.recv(4)
    if not msg_length_data:
        return None
    msg_length = struct.unpack('!I', msg_length_data)[0]  # Giải mã kích thước

    # Nhận toàn bộ dữ liệu
    received_data = b""
    while len(received_data) < msg_length:
        packet = conn.recv(msg_length - len(received_data))  # Nhận phần còn thiếu
        if not packet:
            return None
        received_data += packet

    return pickle.loads(received_data)  # Giải mã object
