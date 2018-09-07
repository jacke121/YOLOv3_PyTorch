# -*- coding: utf-8 -*-
import cv2
from elasticsearch import Elasticsearch
es = Elasticsearch(['192.168.55.90:9200'])
import sys

def update_data(uri,value):
    updateBody = {
         "query": {"bool": {
             "must": {
                 "match": {"uri.keyword":uri},
                 }}
        },
        "script": {
            "inline": "ctx._source.photo.capturenum =params.tags",
            "params": {
                "tags":value
            },
            "lang":"painless"
        }
    }
    res= es.update_by_query(index='ccat1', body=updateBody, doc_type='demo0')
    if res["updated"] is not None and res["updated"]>0:
        if res["updated"]>3:
            print("更新出现异常",res["updated"],uri)
            sys.exit(1)
            return -1
        return res["updated"]
    else:
        return 0
source_arr = ["uri",
                  "camera.deviceId",
                  "photo.capturenum",
                  "fileName","flag"]
def query(param):
    pass
    # es2 = Elasticsearch(['192.168.55.90:9200'])
    # res = es2.search(index="ccat1",doc_type='demo0', body={"_source": source_arr,"query":  {"bool": {
    #    "must": [
    #        {"match": {"uri.keyword": param}}
    #        # {"match": {"camera.deviceId": "a567000a138f4e6589b5a4d9e4f1e410"}}
    #    ]}
    #    },
    # })
    # print(len(res["hits"]["hits"]))
    # for data in res["hits"]["hits"]:
    #     # print(data["_source"]["camera"]["deviceId"],"http://192.168.55.110:7070/image/"+data["_source"]["uri"])
    #     print(data["_source"]["uri"],"capturenum",data["_source"]["photo"]["capturenum"])

if __name__ == '__main__':
  text_name="data/bd_0030724.txt"
  with open(text_name,"r") as r_txt:
      lines=r_txt.readlines()
  index=0

  while(True):
      if index>=len(lines):
          break
      data=lines[index]
      print(data.replace("\n",""))
      if len(data)<1:
          index+=1
          continue
      d_list=data.replace("\n","").split(" ")
      uri=d_list[0]+d_list[1]
      cap = cv2.VideoCapture(uri)
      ret = cap.isOpened()
      _, img = cap.read()
      if img is None:
          print("img is None", uri)
          if update_data(d_list[1],-2):
              print("update 成功")
          else:
             print("update 失败")
          continue
      if len(d_list)==4:
          if d_list[3]=="-2":
            cv2.putText(img, "img null", (800, 30),cv2.FONT_HERSHEY_SIMPLEX,1.2, (0, 0, 255),2)
          elif d_list[3] == "-1":
              cv2.putText(img, "no", (800, 30), cv2.FONT_HERSHEY_SIMPLEX,1.2, (0, 0, 255), 2)
          else:
              cv2.putText(img, "yes", (800, 30), cv2.FONT_HERSHEY_SIMPLEX,1.2, (0, 0, 255),2)
      cv2.imshow('photo', img)
      waitkey_num = cv2.waitKeyEx()
      if waitkey_num == 2490368:
          index-=1
          continue
      if waitkey_num == 2621440:
          index += 1
          continue
      if waitkey_num == 13:
          print("yes enter")
          if len(d_list) == 3:
              d_list.append(d_list[2])
          if len(d_list) == 4 and d_list[3]!=d_list[2] :
              if update_data(d_list[1], int(d_list[2])):
                  d_list[3] = d_list[2]
                  print("update 成功")
              else:
                  print(print("update 失败"))
          with open(text_name, "w") as f_txt:
            lines[index] = ' '.join(d_list) + "\n"
            f_txt.writelines(lines)
          index += 1
          # continue
      if waitkey_num == 32:
          print("space no")
          if update_data(d_list[1], -1):
              if len(d_list) == 3:
                  d_list.append("-1")
              if len(d_list) == 4:
                  d_list[3] = "-1"
              with open(text_name, "w") as f_txt:
                  lines[index] = ' '.join(d_list) + "\n"
                  f_txt.writelines(lines)
              print("update 成功")
          else:
            print(print("update 失败"))
          index+=1




