
# -*- coding: utf-8 -*-
import datetime

import cv2
from elasticsearch import Elasticsearch
es = Elasticsearch(['192.168.55.90:9200'])

source_arr = ["uri",#"coordinateList.height",
                  "camera.deviceId",
                  "photo.capturenum",
                  "fileName","flag","captureInfo.currentDistinguishNum"]
page = es.search(
    index="ccat1", doc_type='demo0',
    scroll ='2m',
    # search_type ='scan',
    size =10000,
    body={"_source": source_arr,"query":  {"bool": {
       "must": [
           # {"match": {"camera.deviceId.keyword": "e521da68922f470a9af492b34fd89b6e"}},
           # {"match": {"captureInfo.currentDistinguishNum": 0}},
           {"range": {
                   "photo.capturenum": {
                       "gte": 1,
                       "lte": 10
                   }
               }
           },
           {"range": {
               "ymdh": {
                   "gte": 2018062300,
                   "lte": 2018072317
               }
           }
           }
       ]}
       }}

)

sid = page['_scroll_id']
scroll_size = page['hits']['total']

# sid="DnF1ZXJ5VGhlbkZldGNoBQAAAAAAA22gFmpxdGVPbFpCUmhpLUVZUVVQV09qcWcAAAAAAAbJLxY2LXFZNlFmQ1RLQzhEclhnRGRPY0tRAAAAAAAGyTAWNi1xWTZRZkNUS0M4RHJYZ0RkT2NLUQAAAAAABsyBFlE4cFdJa1d4U0RhZ2JnbHBrTXo3MFEAAAAAAAbN4xYzYVVJczMtelFBcW5xeDNFNlpLRjhn"
# Start scrolling
while(scroll_size > 0):
    time1=datetime.datetime.now()
    page = es.scroll(scroll_id=sid, scroll='2m')
    # Update the scroll ID
    sid = page['_scroll_id']
    print(sid)
    # Get the number of results that we returned in the last scroll
    scroll_size = len(page['hits']['hits'])
    print("scroll size: " + str(scroll_size),(datetime.datetime.now()-time1).microseconds)
# Do something with the obtained page
