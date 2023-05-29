from pyspark import SparkContext
from itertools import combinations
import sys
import time
import random
import csv

#Start recording time
start_time = time.time()
    
#Loading Data
input_file_path = sys.argv[1]
output_file_path = sys.argv[2]
sc = SparkContext()

# Read the data
data_rdd = sc.textFile(input_file_path)
header = data_rdd.first()
data_rdd = data_rdd.filter(lambda x: x!= header).map(lambda x: x.split(',')).persist()

# Get unique business IDs and user IDs
business_ids = data_rdd.map(lambda x: x[1]).distinct().collect()
user_ids = data_rdd.map(lambda x: x[0]).distinct().collect()
business_ids.sort()
user_ids.sort()

business_index = {}
index_business = {}
user_index = {}
index = 0

for business in business_ids:
    business_index[business] = index
    index_business[index] = business
    index += 1

index2 = 0 

for user in user_ids:
    user_index[user] = index2
    index2 += 1

#defining number of bands and rows
num_hashes = 200
num_bands = 100
num_rows = num_hashes // num_bands

#Given threshold
threshold = 0.5

# Convert the data to a characteristic matrix (business: [users])
char_matrix_list = data_rdd.map(lambda x: (business_index[x[1]], user_index[x[0]])).combineByKey(lambda x: {x}, lambda a, b: a | {b}, lambda a, b: a | b).persist()
char_matrix_set = dict(char_matrix_list.mapValues(set).collect())

# Generate business signatures using the minhashing algorithm
def make_hash(business_id, user_list, num_hashes, num_users):
    b = random.randint(0, 100)
    def hash_func(a, x, b):
        return ((x * a) + b) % num_users
    
    signature = [min(hash_func(i, user, i) for user in user_list) for i in range(num_hashes)]
    
    return business_id, signature

# Generating business signatures
signature_matrix = char_matrix_list.map(lambda x: make_hash(x[0], x[1], num_hashes, len(user_index)))

# Locality-Sensitive Hashing (LSH)
def lsh(row, num_bands, num_rows):
    # Iterate over the range of num_bands and construct a hash bucket for each band
    return [(tuple([i] + row[1][i*num_rows:(i+1)*num_rows]), row[0]) for i in range(num_bands)]
    # For each band:
    # - slice the signature list to extract the corresponding band of num_rows values
    # - add the band ID as the first element of the resulting list
    # - convert the list to a tuple and pair it with the business_id in a tuple
    # - add the tuple to the output list of hash buckets
    
    # The output is a list of hash buckets, each containing a band of the row signature data and the business_id that corresponds to that signature.

# Group similar businesses using LSH
candidates = signature_matrix.flatMap(lambda x: lsh(x, num_bands, num_rows)).groupByKey().filter(lambda x: len(x[1])>1)
candidates = candidates.flatMap(lambda x: [y for y in combinations(x[1],2)]).distinct().persist()

# Jaccard similarity calculation between two businesses
def jaccard_similarity(business_pair, char_mat_set, threshold):
    result = []
    business1, business2 = char_mat_set[business_pair[0]], char_mat_set[business_pair[1]]
    
    intersection = len(business1 & business2)
    union = len(business1.union(business2))
    similarity = intersection / union

    if similarity >= threshold:
        result = ((business_pair[0], business_pair[1]), similarity)

    return result

# Calculate Jaccard similarity for candidate business pairs
similar_businesses = candidates.map(lambda x: jaccard_similarity(x, char_matrix_set, threshold)).filter(lambda x: x!=[]).collectAsMap()

# Reorganize the results to a sorted list
final_pairs = []
for key, value in similar_businesses.items():
    key_sorted = sorted(key)
    final_pairs.append((key_sorted, value))

final_pairs_sorted = sorted(final_pairs)

with open(output_file_path, 'w') as output:
    writer = csv.writer(output, quoting=csv.QUOTE_NONE,escapechar=',')

    writer.writerow(["business_id_1", " business_id_2", " similarity"])
    for pair in final_pairs_sorted:
        writer.writerow([index_business[pair[0][0]], index_business[pair[0][1]], str(pair[1])])
    output.close()

# End timer and print the elapsed time
end_time = time.time()
#print("time: ", str(end_time - start_time))