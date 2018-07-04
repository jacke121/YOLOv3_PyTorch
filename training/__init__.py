#/usr/bin/env python
#coding:utf-8
import datetime
import multiprocessing
import time

#定义Event全局对象
e=multiprocessing.Event()

#扩展的进程类
class MyProcessEx(multiprocessing.Process):
    def __init__(self,name,event):
        super(MyProcessEx,self).__init__(name=name)
        self.__event=event

    def run(self):
        print('process({name}) work...'.format(name=self.name))
        time.sleep(1)
        print('process({name}) sleep...'.format(name=self.name))
        #进入等待状态
        while 1:
            self.__event.wait()
            print('process({name}) awake'.format(name=self.name),datetime.datetime.now())


if __name__=='__main__':
    p=MyProcessEx(name='Model1',event=e)
    p.start()

    for i in range(1000):
        time.sleep(1)
        #唤醒
        e.set()
        e.clear()#加这一句就可以重复使用了
        print('Event status:{s}'.format(s=e.is_set()))