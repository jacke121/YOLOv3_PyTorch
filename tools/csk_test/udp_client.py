from socket import *

import time

HOST = '192.168.30.152'
PORT = 9999
s = socket(AF_INET, SOCK_DGRAM)

conn =False
while True:
    if conn:
        message =b'send message:>>'
        s.sendall(message)
        try:
            data = s.recv(1024)
            print(data)
        except Exception as E:
            print("recv error",E)
            time.sleep(2)
            conn=False
    else:
        try:
            # s.close()
            s.connect((HOST, PORT))
            conn = True
        except Exception as E:
            print("conn error", E)
            time.sleep(2)
    time.sleep(0.5)

s.close()  