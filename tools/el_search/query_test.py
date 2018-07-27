import datetime
import json

from elasticsearch import Elasticsearch

es = Elasticsearch(['192.168.55.90:9200'])
source_arr = ["uri","photo.tempTime",#"coordinateList.height",
                  "camera.deviceId",
                  "photo.capturenum",
                  "fileName","flag","captureInfo.currentDistinguishNum"]
source_arr = ["fileName"]
def query_param():
    time1=datetime.datetime.now()
    res = es.search(index="ccat1",doc_type='demo0',from_=0,size=1000000,
                    body={
                        "_source": source_arr,"query":  {"bool": {
       "must": [
           # {"match": {"camera.deviceId.keyword": "74e47c41ae714653982b02f7cedfecfa"}},
           # {"match": {"captureInfo.currentDistinguishNum": 0}},
           {"range": {
                   "photo.capturenum": {
                       "gte":0,
                       "lte": 10
                   }
               }
           },
           {"range": {
               "ymdh": {
                   "gte": 2018072320,
                   "lte": 2018072406
               }
           }
           }
       ]}
       }})
    # query = {'query': {'match_all': {}}}# 查找所有文档
    print(len(res["hits"]["hits"]))
    datas=[]
    for data in res["hits"]["hits"]:
        pass
        datas.append(data["_source"])
        # print("http://192.168.55.110:7070/image/",data["_source"]["uri"],data["_source"]["photo"]["capturenum"])
        # print("http://192.168.55.110:7070/image/"+data["_source"]["uri"],data["_source"]["photo"]["tempTime"],data["_source"]["captureInfo"]["currentDistinguishNum"])
    for data in datas:
        if datas.count(data)>1:
            print(data)
    print("time",(datetime.datetime.now()-time1).microseconds)

if __name__ == '__main__':
    query_param()

