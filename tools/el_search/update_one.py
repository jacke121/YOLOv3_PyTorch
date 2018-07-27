from elasticsearch import Elasticsearch
es = Elasticsearch(['192.168.55.90:9200'],timeout=50000)

def update_data(uri,value):
    updateBody = {
                 "query": {"bool": {
                     "must": [
                         {"range": {
                           "ymdh": {
                               "gte": 2018062115,
                               "lte": 2018072201
                               # "gte": 20180717,
                               # "lte": 20180720

                           }}  },
                         {"match": {"camera.deviceId.keyword": "a567000a138f4e6589b5a4d9e4f1e410"}},
                         {"match": {"captureInfo.currentDistinguishNum": 0}},
                 # {
                 # "match": {"uri.keyword":uri},
                 # }
                ]}
        },
        "script": {
            "inline": "ctx._source.photo.capturenum =ctx._source.captureInfo.currentDistinguishNum",
            # "inline": "ctx._source.photo.capturenum =params.tags",
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

print(update_data("2018/07/17/0717_232304_927870.jpg",-1))