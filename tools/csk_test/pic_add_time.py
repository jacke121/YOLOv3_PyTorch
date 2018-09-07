
#!/usr/bin/env python
# -*- coding: utf-8 -*-
from socket import *  
import time
# HOST = '<broadcast>'
HOST = '192.168.25.255'
PORT = 21567
BUFSIZE = 20 
ADDR = (HOST, PORT)  
udpCliSock = socket(AF_INET, SOCK_DGRAM)
#设置阻塞
udpCliSock.setblocking(1)
#设置超时时间
udpCliSock.settimeout(2)
udpCliSock.bind(('', 0))  
udpCliSock.setsockopt(SOL_SOCKET, SO_BROADCAST, 1)  
while True:
    data=b"Robot Online!"
    print("sending -> %s" %data)
    udpCliSock.sendto(data,ADDR)
    try:
        data,ADDR = udpCliSock.recvfrom(BUFSIZE)
        if  data:
            print(data)
        time.sleep(0.2)
    except Exception as E:
        continue
udpCliSock.close()