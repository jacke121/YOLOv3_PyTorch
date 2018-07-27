

import datetime
import json

import sys
from elasticsearch import Elasticsearch

es = Elasticsearch(['192.168.55.90:9200'])
source_arr = ["uri","photo.tempTime",#"coordinateList.height",
                  "camera.deviceId",
                  "photo.capturenum",
                  "fileName","flag","captureInfo.currentDistinguishNum"]
source_arr = ["fileName"]

def update_uri(uri,value,startday,endday):

    updateBody = {
                 "query": {"bool": {
                     "must": [
                         {"range": {
                           "ymd": {
                               "gte": startday,
                               "lte": endday
                           }}  },
                 {
                 "match": {"uri.keyword":uri},
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
def query_param(uri):
    res = es.search(index="ccat1",doc_type='demo0',from_=0,size=10000,
                    body={
                        "_source": source_arr,"query":  {"bool": {
       "must": [
           {"match": {"uri.keyword": uri}},
           # {"match": {"captureInfo.currentDistinguishNum": 0}},
           {"range": {
                   "photo.capturenum": {
                       "gte":0,
                       "lte": 10
                   }
               }
           }
       ]}
       }})
    # query = {'query': {'match_all': {}}}# 查找所有文档
    count=len(res["hits"]["hits"])
    return count

if __name__ == '__main__':#把有重复覆盖的更新为-2,需要一个txt
    today = datetime.date.today()
    yesterday = today - datetime.timedelta(days=1)
    yesterday = int(yesterday.strftime("%Y%m%d"))
    today = int(today.strftime("%Y%m%d"))
    file=r"C:\Users\sbdya\Desktop\pic.txt"
    with open(file, "r") as r_txt:
        lines = r_txt.readlines()
    update_count=0
    for index,uri in enumerate(lines):
        uri=uri.replace("\n","")
        count= query_param(uri)
        if count>1:
            res=update_uri(uri, -2,yesterday,today)
            if res > 0 and res < 3:
                print("update ok", uri)
                update_count += 1
            else:
                print(" update error", uri)
                break
    print("更新条数", update_count)