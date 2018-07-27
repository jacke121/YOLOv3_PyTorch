# -*- coding: utf-8 -*-
import os
import datetime
from elasticsearch import Elasticsearch
es = Elasticsearch(['192.168.55.90:9200'],timeout=50000)

def update_data(pic_name,value,startday,endday):

    updateBody = {
                 "query": {"bool": {
                     "must": [
                         {"range": {
                           "ymd": {
                               "gte": startday,
                               "lte": endday
                           }}  },
                 {
                 "match": {"fileName.keyword":pic_name},
                 }
                ]}
        },
        "script": {
            "inline": "ctx._source.photo.capturenum =params.tags",
            "params": {
                "tags":value
            },
            "lang":"painless"
        }
    }
    res= es.update_by_query(index='ccat1', body=updateBody, doc_type='demo0',request_timeout=600)
    if res["updated"] is not None and res["updated"]>0:
        return res["updated"]
    else:
        return 0

if __name__ == '__main__':#把没有老鼠的更新为-1,参数是图片文件夹
    today = datetime.date.today()
    yesterday = today - datetime.timedelta(days=1)
    yesterday=int(yesterday.strftime("%Y%m%d"))
    today=int(today.strftime("%Y%m%d"))

    path=r"\\192.168.55.38\Team-CV\cam2pick\camera_pic_0725\bj_800\mouse_null_all\JPEGImages"
    files=os.listdir(path)
    update_count=0
    for index, filenmae in enumerate(files):
        res=update_data(filenmae,-1,yesterday,today)
        if res>0 and res<3:
            print("update ok",filenmae)
            update_count+=1
        else:
            print(" update error", filenmae)
            break
    print("更新条数",update_count)