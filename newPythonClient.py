import socket

# Define the server (Unity/C#) address and port
server_address = ('127.0.0.1', 5555)  # Change to Unity's server IP and port

# Create a socket object
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Connect to the server
client_socket.connect(server_address)

# Send a number
number_to_send = 42  # Change to the number you want to send
client_socket.sendall(str(number_to_send).encode())

# Close the connection
client_socket.close()