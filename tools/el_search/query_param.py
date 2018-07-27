import json

from elasticsearch import Elasticsearch

es = Elasticsearch(['192.168.55.90:9200'])
source_arr = ["uri",
                  "camera.deviceId",
                  "photo.capturenum",
                  "fileName","flag"]

def query(param):
    res = es.search(index="ccat1",doc_type='demo0', body={"_source": source_arr,"query":  {"bool": {
       "must": [
           {"match": {"uri.keyword": param}}
           # {"match": {"photo.capturenum": 1}},
           # {"match": {"camera.deviceId": "a567000a138f4e6589b5a4d9e4f1e410"}}
       ]}
       }, "from": 0,
     "size": 10000,                                                      })
    # query = {'query': {'match_all': {}}}# 查找所有文档
    print(len(res["hits"]["hits"]))
    for data in res["hits"]["hits"]:
        # print(data["_source"]["camera"]["deviceId"],"http://192.168.55.110:7070/image/"+data["_source"]["uri"])
        print(data["_source"]["uri"],"capturenum",data["_source"]["photo"]["capturenum"])

query("2018/07/20/0720_230143_407866.jpg")