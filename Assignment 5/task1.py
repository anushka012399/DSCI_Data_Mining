import binascii
import random
import csv
from blackbox import BlackBox
import sys
import time

# Set a random seed for reproducibility
random.seed(42)

# Define the Bloom Filter class
class BloomFilter:
    def __init__(self, size, num_hash_functions):
        self.size = size
        self.num_hash_functions = num_hash_functions
        self.bit_array = [False] * size
        self.params = [(random.randint(1, 1000), random.randint(1, 1000), random.randint(1000, 100000)) for _ in range(num_hash_functions)]

    def _hash_functions(self, user_id_int):
        hash_values = []
        for a, b, p in self.params:
            hash_value = ((a * user_id_int + b) % p) % self.size
            hash_values.append(hash_value)
        return hash_values

    def add(self, user_id):
        user_id_int = int(binascii.hexlify(user_id.encode("utf8")), 16)
        for h in self._hash_functions(user_id_int):
            self.bit_array[h] = True

    def check(self, user_id):
        user_id_int = int(binascii.hexlify(user_id.encode("utf8")), 16)
        return all(self.bit_array[h] for h in self._hash_functions(user_id_int))

def myhashs(user_id):
    num_hash_functions = 7
    bloom_filter_size = 69997
    bloom_filter = BloomFilter(bloom_filter_size, num_hash_functions)
    user_id_int = int(binascii.hexlify(user_id.encode("utf8")), 16)
    return bloom_filter._hash_functions(user_id_int)

def main():
    bx = BlackBox()
    input_file = sys.argv[1]
    num_of_asks = int(sys.argv[3])
    stream_size = int(sys.argv[2])
    result_file = sys.argv[4]
    bloom_filter_size = 69997
    num_hash_functions = 7
    results = []

    for i in range(num_of_asks):
        current_stream = bx.ask(input_file, stream_size)
        bloom_filter = BloomFilter(bloom_filter_size, num_hash_functions)
        previous_users = set()
        false_positives = 0
        for user_id in current_stream:
            if bloom_filter.check(user_id) and user_id not in previous_users:
                false_positives += 1
            bloom_filter.add(user_id)
            previous_users.add(user_id)

        fpr = false_positives / stream_size
        results.append((i, fpr))

    with open(result_file, "w", newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(["Time", "FPR"])
        for result in results:
            csvwriter.writerow(result)

if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print(end_time-start_time)

 