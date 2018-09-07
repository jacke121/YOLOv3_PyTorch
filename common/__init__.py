import multiprocessing
from ctypes import c_char_p

from multiprocessing import Value


def aaa(b):
    print(b.value)

if __name__ == '__main__':
    ss = Value(c_char_p, b'ss')
    p= multiprocessing.Process(target=aaa, args=(ss,))
    p.start()