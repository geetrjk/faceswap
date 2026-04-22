import socket
import select
import threading
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from scripts.simplepod import connect

def forward_handler(local_socket, remote_host, remote_port, ssh_transport):
    remote_socket = ssh_transport.open_channel(
        "direct-tcpip", (remote_host, remote_port), local_socket.getpeername()
    )
    if remote_socket is None:
        print("Failed to open remote port")
        local_socket.close()
        return

    while True:
        r, w, x = select.select([local_socket, remote_socket], [], [])
        if local_socket in r:
            data = local_socket.recv(1024)
            if len(data) == 0:
                break
            remote_socket.send(data)
        if remote_socket in r:
            data = remote_socket.recv(1024)
            if len(data) == 0:
                break
            local_socket.send(data)
    remote_socket.close()
    local_socket.close()

def main():
    local_port = 8000
    remote_host = "127.0.0.1"
    remote_port = 8000

    print("Connecting to SimplePod...")
    client = connect()
    transport = client.get_transport()

    local_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    local_server.bind(("127.0.0.1", local_port))
    local_server.listen(5)
    print(f"Tunnel open on 127.0.0.1:{local_port} -> {remote_host}:{remote_port}")

    try:
        while True:
            local_socket, addr = local_server.accept()
            thr = threading.Thread(
                target=forward_handler,
                args=(local_socket, remote_host, remote_port, transport)
            )
            thr.daemon = True
            thr.start()
    except KeyboardInterrupt:
        print("Stopping tunnel.")
    finally:
        local_server.close()
        client.close()

if __name__ == "__main__":
    main()
