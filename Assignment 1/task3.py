#Data Mining Assignment 1
#task3

from pyspark import SparkContext
import json
import sys
import time

# Create a Spark context
sc = SparkContext("local", "Task 3")

review_filepath = sys.argv[1]
business_filepath = sys.argv[2]
output_filepath_question_a = sys.argv[3]
output_filepath_question_b = sys.argv[4]

#start load time
start_time_loading = time.time()

# Load the review data into an RDD
reviewsRDD = sc.textFile(review_filepath).map(json.loads)

# Load the business data into an RDD
businessRDD = sc.textFile(business_filepath).map(json.loads)

# Join the two RDDs on the business_id field
joinedRDD = reviewsRDD.map(lambda x: (x['business_id'], x['stars'])) \
  .join(businessRDD.map(lambda x: (x['business_id'], x['city'])))

#load time calculation
end_time_loading = time.time()

loading_time = end_time_loading-start_time_loading

start_time_create = time.time()

# Calculate the average stars for each city
avg_stars_by_city = joinedRDD.map(lambda x: (x[1][1], x[1][0])) \
  .aggregateByKey((0, 0), 
                  lambda acc, value: (acc[0] + value, acc[1] + 1), 
                  lambda acc1, acc2: (acc1[0] + acc2[0], acc1[1] + acc2[1])) \
  .mapValues(lambda x: x[0] / x[1])

end_time_create = time.time()

time_to_create_averages = end_time_create-start_time_create

avg_stars_by_city.take(10)

#sort and print using Python: Method 1
start_time_m1 = time.time()
sorted_results_m1 = sorted(avg_stars_by_city.collect(), key=lambda x: (-x[1], x[0]))[:10]
for result in sorted_results_m1:
  print("City: %s, Average Stars: %f" % (result[0], result[1]))
sorting_printing_time_m1 = time.time()-start_time_m1

avg_stars_by_city.take(10)

#sort and print using Spark: Method 2
start_time_m2 = time.time()
sorted_results_m2 = avg_stars_by_city.sortBy(lambda x: (-x[1], x[0])).take(10)
for result in sorted_results_m2:
  print("City: %s, Average Stars: %f" % (result[0], result[1]))
sorting_printing_time_m2 = time.time()-start_time_m2

avg_stars_by_city.take(10)

#execution time calculation
m1_total_time = loading_time + time_to_create_averages + sorting_printing_time_m1
m2_total_time = loading_time + time_to_create_averages + sorting_printing_time_m2

sorted_results = avg_stars_by_city.sortBy(lambda x: (-x[1], x[0])).collect()

# Write the results to a text file
with open(output_filepath_question_a, "w") as f:
    f.write("city,stars\n")
    for result in sorted_results:
        f.write("%s,%.2f\n" % (result[0], result[1]))

with open(output_filepath_question_b, 'w') as f:
        json.dump({"m1":m1_total_time,"m2":m2_total_time,"reason":"Method 2 which uses Spark to sort the data is more effecient than the Method 1 which uses Python. But the time taken by Spark is more than that of python beacuse Spark RDD uses SortBy() clause which is used to return the result rows sorted within each partition. This inturn increases the overall time and makes it slower than python."} , f)
