from kafka import KafkaConsumer
from kafka.structs import TopicPartition

host='192.168.55.80:9092,192.168.55.81:9092,192.168.55.82:9092,192.168.55.83:9092,192.168.55.84:9092'
consumer = KafkaConsumer('demo0',
                         bootstrap_servers=['192.168.55.60:2181,192.168.55.61:2181,192.168.55.62:2181'])

print(consumer.partitions_for_topic("demo0"))  # 获取test主题的分区信息
print(consumer.topics())  # 获取主题列表
print(consumer.subscription())  # 获取当前消费者订阅的主题
print(consumer.assignment())  # 获取当前消费者topic、分区信息
print(consumer.beginning_offsets(consumer.assignment()))  # 获取当前消费者可消费的偏移量
consumer.seek(TopicPartition(topic=u'demo0', partition=0), 100)  # 重置偏移量，从第5个偏移量消费
for message in consumer:
    print("%s:%d:%d: key=%s value=%s" % (message.topic, message.partition,
                                         message.offset, message.key,
                                         message.value))
