import pickle 
import sys

def send_message(conn, msg):
    conn.send(pickle.dumps(f"{sys.getsizeof(pickle.dumps(msg))}"))
    conn.sendall(pickle.dumps(msg))

def receive_message(conn):
    received_data = pickle.loads(conn.recv(1024))
    mgs=b''
    while not(mgs):
        mgs=conn.recv(int(received_data))
    res = pickle.loads(mgs)
    return res
