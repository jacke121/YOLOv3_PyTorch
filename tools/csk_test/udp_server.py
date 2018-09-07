# -*- coding: utf-8 -*-
from socket import *

HOST = ''
PORT = 9999
s = socket(AF_INET, SOCK_DGRAM)
s.bind((HOST, PORT))
print('...waiting for message..')
count=0
while True:
    data, address = s.recvfrom(1024)
    print(data, address)
    if count<100:
        s.sendto(b'ok', address)
        count+=1
s.close()