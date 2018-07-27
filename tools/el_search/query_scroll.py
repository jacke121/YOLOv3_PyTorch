# -*- coding: utf-8 -*-
from elasticsearch import Elasticsearch

es = Elasticsearch(['192.168.55.90:9200'],timeout=50000)


# -*- coding: utf-8 -*-
from elasticsearch import helpers

def search():
    es_search_options ={
       "query":  {"bool": {
       "must": [
           # {"match": {"camera.deviceId.keyword": "a567000a138f4e6589b5a4d9e4f1e410"}},
           # {"match": {"uri.keyword": "2018/07/22/0722_011235_906952.jpg"}},
           {"range": {
                   "photo.capturenum": {
                       "gte": 1,
                       "lte": 10
                   }
               }
           },
           {"range": {
               "ymd": {
                   "gte": 20180620,
                   "lte": 20180720

               }
           }
           }
           # {"match": {"photo.capturenum": 1}},
           # {"match": {"camera.deviceId": "a567000a138f4e6589b5a4d9e4f1e410"}}
       ]},
       }
    }
    es_result = get_search_result(es_search_options)
    final_result = get_result_list(es_result)
    return final_result


def get_result_list(es_result):
    final_result = []
    for item in es_result:
        final_result.append(item['_source'])
    return final_result


def get_search_result(es_search_options, scroll='5m', index="ccat1",doc_type='demo0', timeout="1m"):
    es_result = helpers.scan(
        client=es,
        query=es_search_options,
        scroll=scroll,
        index=index,
        size=1000,
        doc_type=doc_type,
        timeout=timeout,preserve_order=False
    )
    return es_result

if __name__ == '__main__':
    final_results = search()
    print(len(final_results))
