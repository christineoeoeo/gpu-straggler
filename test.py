import socket
import os

print("Hostname:", socket.gethostname())
print("User:", os.getenv("USER"))
