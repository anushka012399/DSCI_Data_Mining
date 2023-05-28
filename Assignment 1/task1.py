#Data Mining Assignment 1
#task1

from pyspark import SparkContext
import json
import sys


def task1_analysis(reviewsRDD):
    #A total no of reviews
    total_no_of_reviews = reviewsRDD.count()
    #B no of reviews in 2018
    reviews_in_2018 = reviewsRDD.filter(lambda x: x['date'].startswith('2018')).count()
    #C unique number of users that gave reviews
    unique_no_of_users = reviewsRDD.map(lambda x: x['user_id']).distinct().count()
    #D top 10 users with the number of reviews that they gave
    user_review_mapreduce = reviewsRDD.map(lambda x: (x['user_id'], 1)).reduceByKey(lambda x, y: x + y)
    top_10_users = user_review_mapreduce.takeOrdered(10, key=lambda x: (-x[1], x[0]))
    #E The number of distinct businesses
    unique_businesses = reviewsRDD.map(lambda x: x['business_id']).distinct().count()
    #F The top 10 businesses that had the largest numbers of review
    business_review_mapreduce = reviewsRDD.map(lambda x: (x['business_id'], 1)).reduceByKey(lambda x, y: x + y)
    top_10_businesses = business_review_mapreduce.takeOrdered(10, key=lambda x: (-x[1], x[0]))
    
    return {
        'n_review': total_no_of_reviews,
        'n_review_2018': reviews_in_2018,
        'n_user': unique_no_of_users,
        'top10_user': top_10_users,
        'n_business': unique_businesses,
        'top10_business': top_10_businesses,
    }

#Loading the json file contents into RDD
sc = SparkContext("local", "task 1")
reviews_data = sc.textFile(sys.argv[1])
reviewsRDD = reviews_data.map(json.loads)

results = task1_analysis(reviewsRDD)

#write the results in the json
with open(sys.argv[2], 'w') as f:
    json.dump(results, f)


# /opt/spark/spark-3.1.2-bin-hadoop3.2/bin/spark-submit --executor-memory 4G --driver-memory 4G
# task1.py <review_filepath> <output_filepath