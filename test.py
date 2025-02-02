clients={"123":(9,10), "456": (19,20)}
for i in clients:
    client_ip, client_port = clients[i]
    print(client_ip, client_port)