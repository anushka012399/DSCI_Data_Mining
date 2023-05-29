import time
import csv
import math
from blackbox import BlackBox
import sys

# Set a random seed for reproducibility
import random
random.seed(42)

class FlajoletMartin:
    def __init__(self, num_hash_functions, max_value):
        self.num_hash_functions = num_hash_functions
        self.max_value = max_value
        self.params = [(random.randint(1, 1000), random.randint(1, 1000), random.randint(1000, 100000)) for _ in range(num_hash_functions)]
        self.max_zeros = [0] * num_hash_functions

    def _hash_functions(self, user_id):
        hash_values = []
        for a, b, p in self.params:
            hash_value = ((a * user_id + b) % p) % self.max_value
            hash_values.append(hash_value)
        return hash_values

    def _trailing_zeros(self, n):
        count = 0
        while n > 0 and (n & 1) == 0:
            count += 1
            n >>= 1
        return count

    def add(self, user_id):
        for idx, h in enumerate(self._hash_functions(user_id)):
            self.max_zeros[idx] = max(self.max_zeros[idx], self._trailing_zeros(h))

    def estimate(self):
        num_groups = 3
        group_size = self.num_hash_functions // num_groups
        group_averages = []

        for i in range(0, self.num_hash_functions, group_size):
            group_averages.append(2 ** (sum(self.max_zeros[i:i + group_size]) / group_size))

        group_averages.sort()
        return group_averages[len(group_averages) // 2]

def flajolet_martin(bx, input_file, num_of_asks, num_hash_functions, max_value, window_size):
    estimations = []
    ground_truths = []
    start_time = time.time()

    for i in range(num_of_asks):
        current_stream = bx.ask(input_file, window_size)
        flajolet_martin = FlajoletMartin(num_hash_functions, max_value)
        unique_users = set()
        for user_id in current_stream:
            flajolet_martin.add(hash(user_id))
            unique_users.add(user_id)
        
        ground_truth = len(unique_users)
        estimation = int(flajolet_martin.estimate())
        estimations.append(estimation)
        ground_truths.append(ground_truth)

    end_time = time.time()
    print(f"Execution time: {end_time - start_time}")

    return estimations, ground_truths

def save_results_to_csv(estimations, ground_truths, output_file_name):
    with open(output_file_name, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Time', 'Ground Truth', 'Estimation'])

        for i, (ground_truth, estimation) in enumerate(zip(ground_truths, estimations)):
            writer.writerow([i, ground_truth, estimation])

def main():
    bx = BlackBox()
    input_file = sys.argv[1]
    num_of_asks = int(sys.argv[3])
    stream_size = int(sys.argv[2])
    result_file = sys.argv[4]
    num_hash_functions = 6
    max_value = 2 ** 32 - 1

    estimations, ground_truths = flajolet_martin(bx,input_file, num_of_asks, num_hash_functions, max_value, stream_size)
    save_results_to_csv(estimations, ground_truths, result_file)

if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print(end_time-start_time)


