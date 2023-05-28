#Data Mining Assignment 1
#task2

from pyspark import SparkContext

import sys
import json
#importing time to measure time differences between the two methods
import time

#Partition function
def partitioner(key):
    # defining the partition index as the hash function of the businessid
    return hash(key)

if __name__ == '__main__':

    #assiging input values to variables
    review_filepath = sys.argv[1]
    output_filepath = sys.argv[2]
    n_partition = int(sys.argv[3])

    #Loading the json file contents into RDD
    sc = SparkContext('local[*]', 'Task 2')
    reviews_data = sc.textFile(review_filepath)
    reviewsRDD = reviews_data.map(json.loads)

    start_time = time.time()
    business_review_mapreduce = reviewsRDD.map(lambda x: (x['business_id'], 1)).reduceByKey(lambda x, y: x + y)
    top_10_businesses = business_review_mapreduce.takeOrdered(10, key=lambda x: (-x[1], x[0]))
    default_time = time.time() - start_time

    # Show the number of partitions for the RDD used for finding the top 10 businesses
    deafult_partition= business_review_mapreduce.getNumPartitions()
    default_items_per_partition = business_review_mapreduce.glom().map(len).collect()


    start_time = time.time()
    partitionedRDD = reviewsRDD.map(lambda x: (x['business_id'], 1)).reduceByKey(lambda x, y: x + y, numPartitions=n_partition).partitionBy(n_partition, partitionFunc=partitioner)
    top_10_businesses_partitioned = partitionedRDD.takeOrdered(10, key=lambda x: (-x[1], x[0]))
    custom_time = time.time() - start_time

    # partitionedRDD = reviewsRDD.map(lambda x: (x['business_id'], 1)).reduceByKey(lambda x, y: x + y, numPartitions=n_partition).map(lambda x: (partitioner(x[0]), x)).repartitionAndSortWithinPartitions(n_partition, keyfunc=lambda x: x[0]).map(lambda x: x[1])
    custom_partition= partitionedRDD.getNumPartitions()
    custom_items_per_partition = partitionedRDD.glom().map(len).collect()

    # write the results to a json file
    with open(output_filepath, 'w') as f:
        json.dump({"default":{"n_partition": deafult_partition, "n_items": default_items_per_partition, "exe_time": default_time },
            "customized":{"n_partition": custom_partition, "n_items": custom_items_per_partition, "exe_time": custom_time}} , f)

