#Task 1
# command to run: spark-submit --packages graphframes:graphframes:0.8.2-spark3.1-s_2.12 task1.py <filter threshold> <input_file_path> <community_output_file_path>

from pyspark.sql import SparkSession
from pyspark.sql.functions import collect_set, explode
from graphframes import GraphFrame
import sys
import time

def create_spark_session():
    return SparkSession.builder \
        .appName("CommunityDetection") \
        .config("spark.default.parallelism", "200") \
        .getOrCreate()

def load_data(spark, file_path):
    df = spark.read.csv(file_path, header=True, inferSchema=True)
    return df

def construct_graph(df, threshold):
    df = df.repartition(200)
    
    # Generate user pairs with common business counts
    user_pairs = df.groupBy("user_id", "business_id") \
        .count() \
        .groupBy("business_id") \
        .agg(collect_set("user_id").alias("user_ids")) \
        .select(explode("user_ids").alias("user1"), "user_ids") \
        .select("user1", explode("user_ids").alias("user2")) \
        .filter("user1 != user2") \
        .groupBy("user1", "user2").count() \
        .filter(f"count >= {threshold}")

    # Create vertices and edges
    vertices = user_pairs.select("user1").union(user_pairs.select("user2")).distinct().withColumnRenamed("user1", "id")
    edges = user_pairs.withColumnRenamed("user1", "src").withColumnRenamed("user2", "dst")

    return GraphFrame(vertices, edges)

def detect_communities(graph):
    communities = graph.labelPropagation(maxIter=5)
    return communities

def save_communities(communities, output_file):
    sorted_communities = communities.groupBy("label") \
        .agg(collect_set("id").alias("community")) \
        .select("community") \
        .rdd.map(lambda r: sorted(r[0])) \
        .sortBy(lambda c: (len(c), c[0])) \
        .collect()

    with open(output_file, "w") as f:
        for community in sorted_communities:
            f.write(', '.join(community) + '\n')

if __name__ == "__main__":
    start_time = time.time()
    spark = create_spark_session()
    spark.sparkContext.setLogLevel("WARN")
    
    filter_threshold = sys.argv[1]
    input_file_path = sys.argv[2] #ub_sample_data.csv
    community_output_file_path = sys.argv[3] #task1_commmunities.txt

    df = load_data(spark, input_file_path)
    graph = construct_graph(df, filter_threshold)
    communities = detect_communities(graph)
    save_communities(communities, community_output_file_path)
    end_time = time.time()
    print(end_time - start_time)

    spark.stop()