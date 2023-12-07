import socket
import struct
import time
from Iteration_2 import imageNumber

def start_client():
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect(('127.0.0.1', 5555))  # Connect to the Unity server's IP and port

    print("Connected to Unity server")

    last_sent_value = -1  # Initialize with a value that is different from the expected range

    try:
        while True:
            # Get the current value of imageNumber
            current_image_number = get_image_number_from_iteration_2()

            # Check if the value is different from the last one
            if current_image_number != last_sent_value:
                # Send the current value of imageNumber to the server in the form of an integer
                client.send(struct.pack('!I', current_image_number))  # Convert integer to bytes

                # Update the last sent value
                last_sent_value = current_image_number

            # Wait for a moment before checking/sending the next message
            time.sleep(1)
    except KeyboardInterrupt:
        print("Connection closed by user.")
    finally:
        client.close()

def get_image_number_from_iteration_2():
    # Access the global variable 'imageNumber' from Iteration2.py
    return imageNumber

if __name__ == '__main__':
    start_client()
