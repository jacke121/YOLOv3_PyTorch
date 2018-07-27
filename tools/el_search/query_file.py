import datetime
import json

from elasticsearch import Elasticsearch

es = Elasticsearch(['192.168.55.90:9200'])
source_arr = ["uri","photo.tempTime",#"coordinateList.height",
                  "camera.deviceId",
                  "photo.capturenum",
                  "fileName","flag","captureInfo.currentDistinguishNum"]

def query_param(yesterday,today):
    time1=datetime.datetime.now()
    res = es.search(index="ccat1",doc_type='demo0',from_=0,size=100000, body={"_source": source_arr,"sort":{"photo.tempTime":{"order": 'desc'}},"query":  {"bool": {
       "must": [
           {"match": {"camera.deviceId.keyword": "61449692ec2e4de98baf502d24bfbeab"}},
           # {"match": {"captureInfo.currentDistinguishNum": 0}},
           {"range": {
                   "photo.capturenum": {
                       "gte":1,
                       "lte": 10
                   }
               }
           },
           {"range": {
               "ymdh": {
                   "gte": yesterday,
                   "lte": today
               }
           }
           }
       ]}
       }})
    # query = {'query': {'match_all': {}}}# 查找所有文档
    print(len(res["hits"]["hits"]))
    for data in res["hits"]["hits"]:
        pass
        # print(data["_source"]["uri"])
        # print("http://192.168.55.110:7070/image/",data["_source"]["uri"],data["_source"]["photo"]["capturenum"])
        print("http://192.168.55.110:7070/image/"+data["_source"]["uri"])
        # print("http://192.168.55.110:7070/image/"+data["_source"]["uri"],data["_source"]["photo"]["tempTime"],data["_source"]["captureInfo"]["currentDistinguishNum"])
    print("time",(datetime.datetime.now()-time1).microseconds)

if __name__ == '__main__':
    today = datetime.date.today()
    yesterday = today - datetime.timedelta(days=7)
    yesterday = int(yesterday.strftime("%Y%m%d")+"20")
    today = int(today.strftime("%Y%m%d")+"08")
    query_param(yesterday,today)