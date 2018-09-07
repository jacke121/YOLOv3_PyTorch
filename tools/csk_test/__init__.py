# !/usr/bin/env python2.7
# -*- coding: utf-8 -*-


from scapy.all import srp, Ether, ARP, conf

lan = '192.168.25.1/24'

ans, unans = srp(Ether(dst="FF:FF:FF:FF:FF:FF") / ARP(pdst=lan), timeout=1)
for snd, rcv in ans:
    cur_mac = rcv.sprintf("%Ether.src%")
    cur_ip = rcv.sprintf("%ARP.psrc%")
    print(cur_mac + ' - ' + cur_ip)
