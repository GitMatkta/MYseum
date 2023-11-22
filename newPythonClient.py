import socket
import struct
import time

def start_client():
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect(('127.0.0.1', 5555))  # Connect to the Unity server's IP and port

    print("Connected to Unity server")

    last_sent_value = -1  # Initialize with a value that is different from the expected range

    try:
        while True:
            # Get a new custom value from user input
            custom_value = int(input("Enter a new value (integer): "))

            # Check if the value is different from the last one
            if custom_value != last_sent_value:
                # Send a custom message to the server in the form of an integer
                client.send(struct.pack('!I', custom_value))  # Convert integer to bytes

                # Receive the response from the server
                #response_message = client.recv(4)  # Assuming messages are 4 bytes (int size)
                #response_data = struct.unpack('!I', response_message)[0]  # Convert bytes to integer
                #print(f"Received response from Unity: {response_data}")

                # Update the last sent value
                last_sent_value = custom_value

            # Wait for a moment before checking/sending the next message
            time.sleep(1)
    except KeyboardInterrupt:
        print("Connection closed by user.")
    finally:
        client.close()

if __name__ == '__main__':
    start_client()
