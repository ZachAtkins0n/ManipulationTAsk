"""
This code is to emulate the receving of the joint pos messages in urscript
"""


import socket

def receive_data_from_server(device_b_ip, device_b_port):
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    
    # Bind and listen on the device B's IP and port
    server_socket.bind((device_b_ip, device_b_port))
    server_socket.listen(5)
    print(f"Device B listening on {device_b_ip}:{device_b_port}...")
    
    while True:
        client_socket, client_address = server_socket.accept()
        #print(f"Connection received from {client_address}")
        with client_socket:
        # Receive data from the server
            data = client_socket.recv(1024).decode()
            if data:
                print(f"Received from server: {data}")
        
    #client_socket.close()

# Example usage
device_b_ip = ''  # Device B's IP address
device_b_port = 23456         # Port for Device B

receive_data_from_server(device_b_ip, device_b_port)