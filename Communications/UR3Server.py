import socket
import threading

# This function will handle forwarding data to Device B
def forward_to_device_b(data, device_b_ip, device_b_port):
    try:
        # Create a socket for Device B
        device_b_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        device_b_socket.connect((device_b_ip, device_b_port))
        
        # Send the data to Device B
        device_b_socket.sendall(data.encode())
        #device_b_socket.close()
    except Exception as e:
        print(f"Error forwarding data to Device B: {e}")

# Function to handle incoming connections from Device A
def handle_device_a_connection(client_socket, device_b_ip, device_b_port):
    
    try:
        while True:
            data = client_socket.recv(1024).decode()
            ur3_mov_cmd = "movel p["+ data + "], a=0.2, v=0.1"
            if not data:
                break  # client closed connection
            print(f"[Server] Received from Device A: {data}")
            forward_to_device_b(ur3_mov_cmd, device_b_ip, device_b_port)
    except Exception as e:
        print(f"[Server] Error: {e}")
    finally:
        client_socket.close()

# Set up the server to listen for incoming connections from Device A
def start_server(server_ip, server_port, device_b_ip, device_b_port):
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((server_ip, server_port))
    server_socket.listen(5)
    print(f"Server listening on {server_ip}:{server_port}...")
    
    while True:
        # Accept connections from Device A
        client_socket, client_address = server_socket.accept()
        print(f"Connection received from {client_address}")
        
        # Handle Device A connection in a new thread
        threading.Thread(target=handle_device_a_connection, args=(client_socket, device_b_ip, device_b_port)).start()

# Example usage
server_ip = 'zach.j.atkinson-andes.nord'  # Server's IP address
server_port = 65432         # Port for listening for Device A connections
device_b_ip = ''  # Device B's IP address
device_b_port = 23456        # Port for Device B to receive data

start_server(server_ip, server_port, device_b_ip, device_b_port)
